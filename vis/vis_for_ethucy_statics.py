import matplotlib.pyplot as plt
import numpy as np

# 数据
scenes = ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']

colors = ['#32d3eb','#5bc49f','#feb64d' ,'#ff7c7c','#9287e7']
AN = [2.586, 3.498, 25.096, 3.743, 6.333]
AV = [0.437, 0.178, 0.205, 0.369, 0.206]
AA = [0.131, 0.06, 0.035, 0.039, 0.026]
# '#65AE65'绿 ‘#63A0CB’ 蓝色 ‘#FFA657’ 橙色
# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制AN柱状图
bars = ax1.bar(scenes, AN, color='#63A0CB', width=0.6, label='AN')
ax1.set_ylabel('AN',fontsize=19)
#ax1.set_title('Comparison of AN, AV, AA in ETH-UCY datasets')
ax1.set_ylim(0, 30)
ax1.tick_params(axis='x', labelsize=19)
ax1.tick_params(axis='y', labelsize=19)
# 创建第二坐标轴
ax2 = ax1.twinx()
# 绘制AV和AA的线图
line1, = ax2.plot(scenes, AV, '-o', color='#65AE65', label='AV', markersize=14, linewidth=4)
line2, = ax2.plot(scenes, AA, '-x', color='#FFA657', label='AA', markersize=14, linewidth=4)
ax2.set_ylabel('AV & AA',fontsize=19)
ax2.set_ylim(0, 0.5)
ax2.tick_params(axis='y', labelsize=19)
# 图例
fig.tight_layout()
fig.legend(handles=[bars[0], line1, line2], labels=['AN', 'AV', 'AA'], loc='upper center', bbox_to_anchor=(0.65,0.95), ncol=3,fontsize=19)

plt.show()

