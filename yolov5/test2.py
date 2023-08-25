import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 原始数据
x = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.3, 1.4])
y_data = np.array([
    [0.8768, 0.8491, 0.8495, 0.9081, 0.9017, 0.9121, 0.9141, 0.9141, 0.9141, 0.9141],
    [0.9331, 0.9361, 0.9472, 0.9487, 0.9333, 0.9436, 0.9567, 0.9631, 0.961, 0.9614],
    [0.9028, 0.8972, 0.9222, 0.9194, 0.9056, 0.8917, 0.875, 0.8611, 0.8611, 0.8611],
    [0.9497, 0.9552, 0.9677, 0.9721, 0.9667, 0.9623, 0.9565, 0.9511, 0.9511, 0.9511],
    [0.7495, 0.7379, 0.8109, 0.8107, 0.7915, 0.8141, 0.7665, 0.7524, 0.7524, 0.7461]
])

# 平滑数据
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = np.array([make_interp_spline(x, y)(x_smooth) for y in y_data])

# 创建曲线图
plt.plot(x_smooth, y_smooth.T)

# 添加标题和标签
plt.title('Smoothed Curve Chart')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加图例
plt.legend(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5'])

# 设置X轴坐标刻度为对数刻度

# 显示曲线图
plt.show()
