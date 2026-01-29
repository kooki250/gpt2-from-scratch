import os
from torch.distributed import init_process_group, destroy_process_group
import time
import inspect
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        # GPT-2's c_attn and c_proj usually have bias=True
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=True)

        # output projection - essentially combining the information from all attention heads into a single representation of the expected dimension.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        tril = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "bias", tril.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        # x = [batch_size, sequence_length, n_embd]
        batch_size, sequence_length, n_embd = x.shape

        # Calculate Q, K, V
        kqv = self.c_attn(x)
        q, k, v = kqv.split(self.n_embd, dim=2)

        # Split to heads
        # This results in a 4-dimensional tensor
        # [batch_size, sequence_length, n_head, n_embd // n_head
        k = k.view(
            batch_size, sequence_length, self.n_head, self.n_embd // self.n_head
        ).transpose(1, 2)
        q = q.view(
            batch_size, sequence_length, self.n_head, self.n_embd // self.n_head
        ).transpose(1, 2)
        v = v.view(
            batch_size, sequence_length, self.n_head, self.n_embd // self.n_head
        ).transpose(1, 2)

        # Calculate attention
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Add causal attention mask
        # Ensure the mask matches the sequence length. Mask out future tokens (upper triangle of the attention scores).
        # The `bias` tensor is float, so we need to compare it to 0 to get a boolean mask.
        # attn = attn.masked_fill(self.bias[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
        # Calculate attention
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # Converting to Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Concatenate heads

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.n_embd)
        )
        # Multiply by the output projection
        y = self.c_proj(y)

        # return the result
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        # MLP output projection should go from 4*n_embd back to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config)
                                for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # GPT-2's lm_head usually has bias=True
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx [batch_size, sequence_length]
        batch_size, sequence_length = idx.shape
        pos = torch.arange(0, sequence_length,
                           dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        token_emb = self.transformer.wte(idx)
        x = token_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {
            "openai-community/gpt2",
            "openai-community/gpt2-medium",
            "openai-community/gpt2-large",
            "openai-community/gpt2-xl",
        }
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            "openai-community/gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            "openai-community/gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            "openai-community/gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            "openai-community/gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # always 50257 for GPT model checkpoints
        config_args["vocab_size"] = 50257
        # always 1024 for GPT model checkpoints
        config_args["block_size"] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device.startswith("cuda")
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer


def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = 'mps'
        print(f"using device: {device}")


    def load_tokens(filename):
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt


    class DataLoaderLite:
        def __init__(self, B, T, process_rank, num_processes, split):
            self.B = B
            self.T = T
            self.process_rank = process_rank
            self.num_processes = num_processes
            assert split in {'train', 'val'}

            # get the shard filenames
            data_root = "edu_fineweb10B"
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            if master_process:
                print(f"found {len(shards)} shards for split {split}")
            self.reset()

        def reset(self):
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        def next_batch(self):
            B, T = self.B, self.T
            buf = self.tokens[self.current_position: self.current_position+B*T+1]
            x = (buf[:-1]).view(B, T)  # inputs
            y = (buf[1:]).view(B, T)  # targets
            # advance the position in the tensor
            self.current_position += B * T * self.num_processes
            # if loading the next batch would be out of bounds, advance to next shard
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
            return x, y


    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    #  Gradient accumulation
    total_batch_size = 524288
    B = 16
    T = 1024  # no sohev

    assert total_batch_size % (B * T*ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T*ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Added this line to explicitly set matmul precision
    # reducing percision
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Power of two
    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)  # Comment this line out
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    max_steps = 19073
    warmup_steps = 10  # 715


    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    device_type = "cuda" if device.startswith("cuda") else device
    max_steps = 50000
    last_step = max_steps - 1
    log_file = "log.txt"
    log_dir = "checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0  # Reset loss_accum for each step

        for micro_steps in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps  # compensate on the mean.
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_steps == grad_accum_steps - 1
            loss.backward()

        if ddp:
            torch.distributed.all_reduce(
                loss_accum, op=torch.distributed.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        token_processed = train_loader.B * train_loader.T * \
            grad_accum_steps * ddp_world_size
        token_per_sec = token_processed / dt

        if master_process:
            print(
                f"step {step}, loss: {loss_accum.item()} | lr {lr:.4e} | norm: {norm:.4f} | dt:{dt:.2f}sec | tok/sec: {token_per_sec}"
            )


        # once in a while evaluate our validation loss
        if step % 250 == 0 or step == last_step and step != 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                torch.distributed.all_reduce(
                    val_loss_accum, op=torch.distributed.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                # optionally write model checkpoints
                if (step % 5000 == 0 or step == last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)
            model.train()

        # optimizer.zero_grad()
        # loss_accum = 0.0  # Reset loss_accum for each step
        # for micro_steps in range(grad_accum_steps):
        #     x, y = train_loader.next_batch()
        #     x = x.to(device)
        #     y = y.to(device)

        #     with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #         logits, loss = model(x, y)
        #     loss = loss / grad_accum_steps  # compensate on the mean.
        #     loss_accum += loss.detach()
        #     if ddp:
        #         model.require_backward_grad_sync = micro_steps == grad_accum_steps - 1
        #     loss.backward()


    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()