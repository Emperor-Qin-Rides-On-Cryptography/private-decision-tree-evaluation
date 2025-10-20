import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Try to pick a Chinese-capable font available on the system
preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'PingFang SC']
available = {f.name: f for f in font_manager.fontManager.ttflist}
chosen = None
for name in preferred:
    if name in available:
        chosen = name
        break

if chosen is not None:
    plt.rcParams['font.sans-serif'] = [chosen]
else:
    # fallback to SimHei name (may still fail if no font installed)
    plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False

# 1. 设定X轴：操作数的位长 (l)
# 我们选择从 8位 到 512位，与论文中的实验范围类似
l = np.array([8, 16, 32, 64, 128, 256, 512])

# 2. 计算 Folklore 运算符的乘法深度
# 深度 = ceil(log2(l))
depth_folklore = np.ceil(np.log2(l))+1

# 3. 计算常权重运算符在不同 k 值下的乘法深度
# 论文中探讨了 k 和 l 的不同关系，我们在此复现几种
# 深度 = ceil(log2(k))

# Case 1: k = l / 2
k1 = l / 2
depth_cw_k1 = np.ceil(np.log2(k1))+1

# Case 2: k = l / 4
k2 = l / 4
depth_cw_k2 = np.ceil(np.log2(k2))+1

# Case 3: k = l / 8
k3 = l / 8
# 对于 k=1, log2(1)=0, ceil(0)=0. np.log2(1)会正确处理
with np.errstate(divide='ignore'): # 忽略 log2(0) 的警告
    depth_cw_k3 = np.ceil(np.log2(k3))+1
# 当 l=8, k=1, log2(1)=0. 当 l<8 时 k<1, log2(k)为负, 但我们不关心这个范围.

# 4. 绘图
plt.style.use('seaborn-v0_8-whitegrid')

# seaborn style may override rcParams; re-apply chosen font setting to ensure Chinese support
if 'chosen' in globals() and chosen is not None:
    plt.rcParams['font.sans-serif'] = [chosen]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 Folklore 运算符的折线（使用 raw string mathtext）
ax.plot(l, depth_folklore, marker='o', linestyle='-', label=r'传统等值运算符 $f_{AF}$ ($\lceil \log_2(l) \rceil$)')

# 绘制常权重运算符的折线（k = l/2, l/4, l/8）
ax.plot(l, depth_cw_k1, marker='s', linestyle='--', label=r'常权重等值运算符 $f_{ACW}$ ($k = l/2$)')
ax.plot(l, depth_cw_k2, marker='^', linestyle='--', label=r'常权重等值运算符 $f_{ACW}$ ($k = l/4$)')
ax.plot(l, depth_cw_k3, marker='d', linestyle='--', label=r'常权重等值运算符 $f_{ACW}$ ($k = l/8$)')

# 设置图表标题和标签
ax.set_title('传统等值运算符与常权重等值运算符在不同比特精度下消耗的乘法深度对比', fontsize=16)
ax.set_xlabel('比特精度', fontsize=12)
ax.set_ylabel('乘法深度', fontsize=12)

ax.set_xscale('log', base=2)
ax.set_xticks(l)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# 显示图例
ax.legend()

# 显示图表
plt.show()