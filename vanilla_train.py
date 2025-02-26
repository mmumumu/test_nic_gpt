# vanilla_train.py
import torch
import torch.optim as optim
from transformers import GPT2Tokenizer
from vanilla_model import GPT, GPTConfig
import torch.nn.functional as F

def prepare_dataset(tokenizer, sentences, block_size):
    """Tokenize a list of sentences with padding/truncation."""
    tokenized = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=block_size,
        return_tensors="pt"
    )
    return tokenized.input_ids

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize GPT-2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 从 dataset.txt 文件中读取句子，每行视为一个句子
    with open("dataset.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    block_size = 128
    input_ids = prepare_dataset(tokenizer, sentences, block_size)
    print("Tokenized input IDs shape:", input_ids.shape)  # (num_sentences, block_size)
    
    # Create model configuration and instantiate the model.
    config = GPTConfig(n_layer=2, n_embd=128, n_head=4, block_size=block_size)
    model = GPT(config).to(device)
    
    input_ids = input_ids.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 50
    num_samples = input_ids.size(0)
    
    print("Starting training on dataset.txt...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i in range(num_samples):
            idx = input_ids[i:i+1]  # (1, block_size)
            logits, loss = model(idx, targets=idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1:02d}: Average Loss = {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "vanilla_model.pt")
    print("Training complete. Model saved as vanilla_model.pt")

if __name__ == '__main__':
    main()