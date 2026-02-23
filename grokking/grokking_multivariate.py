import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# 1. 实验配置
# ===========================
# 为每个 n 设定适合的 p，保证既能训练又能发生 Grokking
EXPERIMENT_CONFIGS = [
    {'n': 2, 'p': 13, 'epochs': 5000, 'batch_size': 512},   # n=2 需要大一点的 p
    {'n': 3, 'p': 13, 'epochs': 5000, 'batch_size': 512}
]

D_MODEL = 128   
N_HEAD = 4      
N_LAYER = 1     # 保持单层以获得平滑曲线
LR = 1e-3       
TRAIN_RATIO = 0.5 
SEED = 42
LOG_INTERVAL = 10 # 每50轮记录一次，过滤高频震荡

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. 模型定义 (通用版)
# ===========================
class ModularAdder(nn.Module):
    def __init__(self, p, d_model, n_head, n_layer, n_nums):
        super().__init__()
        self.token_emb = nn.Embedding(p, d_model)
        # 位置编码长度适配 n_nums
        self.pos_emb = nn.Parameter(torch.randn(1, n_nums, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model * 4, 
            dropout=0.0,
            activation='relu',
            batch_first=True,
            norm_first=True # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p, bias=False)

    def forward(self, idx):
        x = self.token_emb(idx) + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1) # Mean Pooling
        x = self.ln_f(x)
        return self.head(x)

# ===========================
# 3. 训练与数据生成函数
# ===========================
def run_experiment(config):
    n = config['n']
    p = config['p']
    epochs = config['epochs']
    bs = config['batch_size']
    
    print(f"\n>>> 开始实验: n={n}, p={p} (Total Epochs: {epochs})")
    
    # --- 生成数据 ---
    print(f"    生成数据中 (p^{n})...")
    ranges = [torch.arange(p) for _ in range(n)]
    inputs = torch.cartesian_prod(*ranges)
    targets = inputs.sum(dim=1) % p
    
    # --- 划分数据集 ---
    total_samples = len(inputs)
    train_size = int(total_samples * TRAIN_RATIO)
    indices = torch.randperm(total_samples)
    
    train_ds = TensorDataset(inputs[indices[:train_size]], targets[indices[:train_size]])
    test_ds = TensorDataset(inputs[indices[train_size:]], targets[indices[train_size:]])
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
    
    # --- 初始化模型 ---
    model = ModularAdder(p, D_MODEL, N_HEAD, N_LAYER, n).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    # --- 训练循环 ---
    history = {'step': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            optimizer.step()
            
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        # --- 记录数据 (低频) ---
        if epoch % LOG_INTERVAL == 0 or epoch == epochs - 1:
            train_acc = correct / total
            
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    logits_t = model(x_test)
                    test_correct += (logits_t.argmax(-1) == y_test).sum().item()
                    test_total += y_test.size(0)
            test_acc = test_correct / test_total
            
            history['step'].append(epoch)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            if epoch % 500 == 0:
                print(f"    Epoch {epoch:4d} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")
                
            # 提前停止 (如果完全收敛)
            if test_acc > 0.995 and train_acc > 0.995:
                print(f"    *** Grokking 达成于 Epoch {epoch} ***")
                break
                
    return history

# ===========================
# 4. 主程序与绘图
# ===========================
results = {}
for conf in EXPERIMENT_CONFIGS:
    res = run_experiment(conf)
    results[conf['n']] = res

print("\n>>> 所有实验结束，开始绘图...")

# 设置绘图风格
plt.figure(figsize=(12, 7), dpi=120)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝, 橙, 绿

for i, (n, hist) in enumerate(results.items()):
    c = colors[i % len(colors)]
    steps = hist['step']
    
    # 绘制训练曲线 (虚线)
    plt.plot(steps, hist['train_acc'], 
             linestyle='--', color=c, alpha=0.5, label=f'Train(n={n})')
    
    # 绘制测试曲线 (实线, 加粗)
    plt.plot(steps, hist['test_acc'], 
             linestyle='-', color=c, linewidth=2.5, label=f'Test(n={n})')

plt.title("Grokking Comparison: Different n", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.xscale('log')
plt.ylabel("Accuracy", fontsize=14)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()