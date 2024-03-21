import matplotlib.pyplot as plt
import numpy as np

# 数据
teams = ['CLE', 'GSW', 'NYK', 'OKC', 'SAS']

groupnet_ade = [0.99, 1.01, 0.99, 0.99, 1.01]
hstte_ade = [0.843, 0.824, 0.837, 0.830, 0.825]
hstte_mvl_ade = [0.704, 0.792, 0.781, 0.802, 0.774]

groupnet_fde = [1.44, 1.43, 1.41, 1.44, 1.46]
hstte_fde = [1.515, 1.471, 1.485, 1.480, 1.481]
hstte_mvl_fde = [1.406, 1.340, 1.355, 1.355, 1.357]

barWidth = 0.20
spacing = 0.05
r1 = np.arange(len(groupnet_ade))
r2 = [x + barWidth + spacing for x in r1]
r3 = [x + barWidth + spacing for x in r2]
# 字体大小
label_fontsize = 28
y_tick_fontsize = 24
legend_fontsize = 24
#x_label_fontsize = 20
x_tick_fontsize = 30
# 使用您提供的图中的颜色
# colors = ['#4E79A7', '#59A14E', '#76B4E3']
#colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F'] # 暖色
colors = ['#65AE65', '#63A0CB', '#FFA657']
# colors = ['#60ACFC', '#5BC49F', '#FEB64D'] 撞色
# colors = ['#60ACFC', '#32D3EB','#5BC49F'] 冷色
# 画ADE图
plt.figure(figsize=(12, 6))
plt.bar(r1, groupnet_ade, color=colors[0], width=barWidth, edgecolor='grey', label='GroupNet ADE')
plt.bar(r2, hstte_ade, color=colors[1], width=barWidth, edgecolor='grey', label='Dual-TT ADE')
plt.bar(r3, hstte_mvl_ade, color=colors[2], width=barWidth, edgecolor='grey', label='Dual-TT+MGTP ADE')
#plt.xlabel('Teams', fontweight='bold', fontsize=x_label_fontsize)
plt.ylabel('ADE Performance (m)', fontsize=label_fontsize)
plt.xticks([r + barWidth + spacing for r in range(len(groupnet_ade))], teams,fontsize=x_tick_fontsize)
plt.yticks(fontsize=y_tick_fontsize)
plt.ylim(0.6, 1.3)
plt.legend(loc="upper left",fontsize=legend_fontsize)
plt.tight_layout()
plt.show()

# 画FDE图
plt.figure(figsize=(12, 6))
plt.bar(r1, groupnet_fde, color=colors[0], width=barWidth, edgecolor='grey', label='GroupNet FDE')
plt.bar(r2, hstte_fde, color=colors[1], width=barWidth, edgecolor='grey', label='Dual-TT FDE')
plt.bar(r3, hstte_mvl_fde, color=colors[2], width=barWidth, edgecolor='grey', label='Dual-TT+MGTP FDE')
#plt.xlabel('Teams', fontweight='bold', fontsize=x_label_fontsize)
plt.ylabel('FDE Performance (m)', fontsize=label_fontsize)
plt.xticks([r + barWidth + spacing for r in range(len(groupnet_fde))], teams,fontsize=x_tick_fontsize)
plt.yticks(fontsize=y_tick_fontsize)
plt.ylim(1.2, 1.8)
plt.legend(loc="upper left",fontsize=legend_fontsize)
plt.tight_layout()
plt.show()



