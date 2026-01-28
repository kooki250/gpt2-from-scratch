import torch
from train4 import GPT, GPTConfig
from transformers import GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "checkpoints/model_01999.pt"

# Allow GPTConfig for safe loading
torch.serialization.add_safe_globals([GPTConfig])

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint['config']

# Create model
model = GPT(config)

# Fix DDP prefix in state_dict
state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_state_dict[k[len("_orig_mod."):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sampling function
def sample(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=None):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_length):
        with torch.no_grad():
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Main
if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    output = sample(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=100)
    print("\nGenerated Text:")
    print(output)
