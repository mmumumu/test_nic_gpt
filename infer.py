# infer.py
import torch
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig

def generate(model, idx, max_new_tokens, block_size, temperature=1.0):
    """
    Autoregressively generate tokens.
    idx: Tensor of shape (B, T) representing initial tokens.
    Returns the generated sequence.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        # Focus on the last time step.
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    block_size = 128
    config = GPTConfig(n_layer=2, n_embd=128, n_head=4, block_size=block_size)
    model = GPT(config).to(device)
    
    # Load trained model checkpoint.
    state_dict = torch.load("model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    prompt = "Economic sanctions as a tool of foreign"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Ensure input is within block_size.
    input_ids = input_ids[:, :block_size]
    
    # Generate new tokens.
    generated = generate(model, input_ids, max_new_tokens=20, block_size=block_size, temperature=1.0)
    
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print("Generated text:")
    print(output_text)

if __name__ == '__main__':
    main()