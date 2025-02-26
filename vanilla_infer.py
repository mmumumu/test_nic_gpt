# vanilla_infer.py
import torch
from transformers import GPT2Tokenizer
from vanilla_model import GPT, GPTConfig

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load GPT-2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    block_size = 128
    config = GPTConfig(n_layer=2, n_embd=128, n_head=4, block_size=block_size)
    model = GPT(config).to(device)
    
    # Load trained model checkpoint.
    state_dict = torch.load("vanilla_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    prompt = "Economic sanctions as a tool of foreign"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids[:, :block_size]
    
    # Generate new tokens.
    generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0, top_k=40)
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    print("Generated text:")
    print(output_text)

if __name__ == '__main__':
    main()