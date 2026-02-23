import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# 1. 实验配置
# ===========================
SEARCH_CONFIGS = [
    {'n': 2, 'p': 17},  # 41^2 = 1681 样本
    {'n': 3, 'p': 17},  # 13^3 = 2197 样本
    {'n': 4, 'p': 17},   # 7^4  = 2401 样本
]

# 比例生成：从 0.4 开始，向下递减，步长 0.025 (2.5%)
# 范围: [0.400, 0.375, 0.350, ..., 0.025]
START_RATIO = 0.025
END_RATIO = 0.01
STEP = 0.005
TEST_RATIOS = np.arange(START_RATIO, END_RATIO, -STEP).tolist()

# 固定参数
D_MODEL = 128   
N_HEAD = 4      
N_LAYER = 1     
LR = 1e-3       
MAX_EPOCHS = 3000
TARGET_ACC = 0.95
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. 模型定义
# ===========================
class ModularAdder(nn.Module):
    def __init__(self, p, d_model, n_head, n_layer, n_nums):
        super().__init__()
        self.token_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_nums, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model * 4, 
            dropout=0.0,
            activation='relu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p, bias=False)

    def forward(self, idx):
        x = self.token_emb(idx) + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1) 
        x = self.ln_f(x)
        return self.head(x)

# ===========================
# 3. 训练函数
# ===========================
def train_and_evaluate(n, p, ratio):
    # --- 生成数据 ---
    ranges = [torch.arange(p) for _ in range(n)]
    inputs = torch.cartesian_prod(*ranges)
    targets = inputs.sum(dim=1) % p
    
    total_samples = len(inputs)
    train_size = int(total_samples * ratio)
    
    # 极小样本保护
    if train_size < 5:
        return False, train_size, total_samples

    indices = torch.randperm(total_samples)
    train_ds = TensorDataset(inputs[indices[:train_size]], targets[indices[:train_size]])
    test_ds = TensorDataset(inputs[indices[train_size:]], targets[indices[train_size:]])
    
    bs = min(512, train_size)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
    
    # --- 模型与优化 ---
    model = ModularAdder(p, D_MODEL, N_HEAD, N_LAYER, n).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # --- 训练循环 ---
    success = False
    
    # 进度条显示可以简化，只打印关键点
    print(f"    [训练] 样本数: {train_size} | ", end="", flush=True)

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()

        # 检测泛化
        if (epoch + 1) % 200 == 0 or epoch == MAX_EPOCHS - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    pred = model(x_test).argmax(-1)
                    correct += (pred == y_test).sum().item()
                    total += y_test.size(0)
            acc = correct / total
            
            # 达到目标，提前成功
            if acc >= TARGET_ACC:
                print(f"✔ 成功 (Epoch {epoch}, Acc {acc:.1%})")
                success = True
                break
    
    if not success:
        print(f"✘ 失败 (Max Acc {acc:.1%})")
        
    return success, train_size, total_samples

# ===========================
# 4. 主搜索循环
# ===========================
final_results = []

print(f"=== 开始搜索最小样本量 (Start Ratio={START_RATIO}, Step={STEP}, MaxEpochs={MAX_EPOCHS}) ===\n")

for config in SEARCH_CONFIGS:
    n = config['n']
    p = config['p']
    print(f">>> 正在测试 n={n}, p={p} (Total Data: {p**n})")
    
    best_result = None
    
    for ratio in TEST_RATIOS:
        print(f"  > 尝试比例 {ratio:.3f}: ", end="")
        
        is_success, train_size, total = train_and_evaluate(n, p, ratio)
        
        if is_success:
            # 成功了，记录当前为“已知最小值”，并继续尝试更小的
            best_result = {
                'n': n,
                'p': p,
                'min_ratio': ratio,
                'min_samples': train_size,
                'total_samples': total
            }
        else:
            # 失败了，说明已经跌破了所需的最小数据量
            # 停止当前 n 的搜索，因为再小肯定也不行
            print(f"  > 比例 {ratio:.3f} 失败，停止向下搜索。")
            break
    
    if best_result:
        final_results.append(best_result)
        print(f"  *** n={n} 的最小比例锁定为: {best_result['min_ratio']:.3f} (样本数 {best_result['min_samples']}) ***\n")
    else:
        # 如果 0.4 起步就失败
        final_results.append({
            'n': n, 'p': p, 'min_ratio': f">{START_RATIO}", 'min_samples': "N/A", 'total_samples': p**n
        })
        print(f"  *** n={n} 在起始比例 {START_RATIO} 即失败 ***\n")

# ===========================
# 5. 结果汇总输出
# ===========================
print("\n" + "="*60)
print(f"{'n':<5} | {'p':<5} | {'Total':<8} | {'Min Ratio':<12} | {'Min Samples':<12}")
print("-" * 60)
for res in final_results:
    if isinstance(res['min_ratio'], float):
        ratio_str = f"{res['min_ratio']:.1%}"
    else:
        ratio_str = str(res['min_ratio'])
        
    print(f"{res['n']:<5} | {res['p']:<5} | {res['total_samples']:<8} | {ratio_str:<12} | {res['min_samples']:<12}")
print("-" * 60)