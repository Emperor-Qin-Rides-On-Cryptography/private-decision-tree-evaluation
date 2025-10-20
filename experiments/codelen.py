import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import matplotlib
from tqdm import tqdm
import os

from matplotlib.font_manager import FontProperties

# 1. 下载字体文件 (例如 SimHei.ttf) 并放置在你的项目目录下或系统字体目录

# 2. 创建 FontProperties 对象，指定字体文件路径和大小
font_path = 'C:\Windows\Fonts\Microsoft YaHei UI.ttf' # 替换为你的字体文件路径 (可以是相对路径或绝对路径)
font = FontProperties(fname=font_path, size=14)


# 需要编码的不同值的数量
n = 2**16

# 测试的汉明权重 (k) 范围
k_values = np.arange(2, 100)

# 用于存储计算出的最小码长 (m)
m_values = []

print(f"正在为 {n} ({int(n)}) 个不同值计算最小码长...")
for k in tqdm(k_values, desc="计算进度"):
    m = k
    while True:
        # 使用scipy.special.comb计算组合数 C(m, k)
        # 使用 float64 进行计算以避免溢出，因为 n 很大
        num_combinations = comb(m, k, exact=False)
        
        # 检查组合数是否足够表示 n 个值
        if num_combinations >= n:
            m_values.append(m)
            break
        # 加速 m 的增长，因为 m 会变得很大
        # 当组合数与 n 相差超过两个数量级时，步长增大
        if n > 100 * num_combinations > 0:
             step = int((m-k)/10) + 1
             m += step
        else:
            m += 1


print("计算完成，正在生成图表...")

# --- 绘图 ---
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制折线图
ax.plot(k_values, m_values, marker=None, linestyle='-', color='b', label=f'n={n}时所需的最小码长')

# 设置图表的标题和坐标轴标签
ax.set_title(f'在需要编码 {n} 个不同值的情况下\n汉明权重(k)与最小码长(m)的关系', fontsize=16)
ax.set_xlabel('汉明权重 (k)', fontsize=12)
ax.set_ylabel('常权重编码后的最小码长 (m)', fontsize=12)

ax.grid(False)
ax.legend()

# 显示图表
plt.show()

