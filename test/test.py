import torch as th
import numpy as np
import matplotlib.pyplot as plt

# 定义 Q 函数
def compute_q(x, means, std=0.05):
    q_terms = [ -((x - m)**2).sum(dim=1) / (2 * std**2) for m in means ]
    stacked = th.stack(q_terms, dim=1)  # [B, num_modes]
    return th.logsumexp(stacked, dim=1)

# 生成网格点
x_vals = np.linspace(-1.5, 1.5, 200)
y_vals = np.linspace(-1.5, 1.5, 200)
xx, yy = np.meshgrid(x_vals, y_vals)
grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
x_tensor = th.tensor(grid_points, dtype=th.float32)

# 设置两个 mode 的均值
mean1 = th.tensor([0.7, 0.7])
mean2 = th.tensor([-0.7, -0.7])
means = [mean1, mean2]

# 计算 Q 值
q_vals = compute_q(x_tensor, means, std=0.05)
q_vals_np = q_vals.detach().numpy().reshape(xx.shape)

# 绘图
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, q_vals_np, levels=50, cmap='viridis')
plt.colorbar(label='Q(x)')
plt.scatter([0.7, -0.7], [0.7, -0.7], color='red', label='Q peaks')
plt.title('Heatmap of Q(x) = logsumexp of two Gaussians')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
