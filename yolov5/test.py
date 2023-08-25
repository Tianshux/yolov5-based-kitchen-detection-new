import matplotlib.pyplot as plt

# 数据
x = [0, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.3, 'all']
y1 = [0.8768, 0.8491, 0.8495, 0.9081, 0.9017, 0.9121, 0.9141, 0.9141, 0.9141, 0.9141]
y2 = [0.9331, 0.9361, 0.9472, 0.9487, 0.9333, 0.9436, 0.9567, 0.9631, 0.961, 0.9614]
y3 = [0.9028, 0.8972, 0.9222, 0.9194, 0.9056, 0.8917, 0.875, 0.8611, 0.8611, 0.8611]
y4 = [0.9497, 0.9552, 0.9677, 0.9721, 0.9667, 0.9623, 0.9565, 0.9511, 0.9511, 0.9511]
y5 = [0.7495, 0.7379, 0.8109, 0.8107, 0.7915, 0.8141, 0.7665, 0.7524, 0.7524, 0.7461]

# 创建折线图
plt.plot(x, y1, marker='o', label='carpet')
plt.plot(x, y2, marker='x', label='cable')
plt.plot(x, y3, marker='s', label='toothbrush')
plt.plot(x, y4, marker='^', label='leather')
plt.plot(x, y5, marker='D', label='zipper')

# 添加标题和标签
plt.title('The distance Influence of Client1 with Avg_distance Algrithm')
plt.xlabel('ratio')
plt.ylabel('performance')

# 添加图例
plt.legend()

# 设置X轴坐标刻度为对数刻度

# 显示折线图
plt.savefig('curve_chart.png')
plt.show()
