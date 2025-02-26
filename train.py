import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from model import GPT, GPTConfig
import torch.nn.functional as F

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        # 从文件中按行读取句子
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 去除空行，并去掉首尾空白
        self.sentences = [line.strip() for line in lines if line.strip()]
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_ids = []
        for sentence in self.sentences:
            tokenized = tokenizer(
                sentence,
                padding="max_length",
                truncation=True,
                max_length=block_size,
                return_tensors="pt"
            )
            # squeeze 掉 batch 维度，得到 (block_size,) 的 tensor
            self.input_ids.append(tokenized.input_ids.squeeze(0))
        # 堆叠成一个 tensor，形状为 (num_sentences, block_size)
        self.input_ids = torch.stack(self.input_ids)

    def __len__(self):
        return self.input_ids.size(0)
    
    def __getitem__(self, idx):
        return self.input_ids[idx]

def main():
    # 使用 MPS（Apple Silicon）或 CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 初始化 GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    block_size = 128  # 每条句子 tokenize 后控制在 128 个 token
    dataset = TextDataset("dataset.txt", tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建模型配置和模型
    config = GPTConfig(n_layer=2, n_embd=128, n_head=4, block_size=block_size)
    model = GPT(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50
    
    print("Starting training on dataset.txt...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch = batch.to(device)
            # 这里简单地使用输入作为目标（toy LM 任务）
            logits, loss = model(batch, targets=batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1:02d}: Average Loss = {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "model.pt")
    print("Training complete. Model saved as model.pt")

if __name__ == '__main__':
    main()