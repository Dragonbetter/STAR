import matplotlib.pyplot as plt
import numpy as np
import os

# 数据
teams = ["Gates", "Little", "Nexus", "Coupa", "Bookstore", "Quad", "Hyang", "DeathCircle"]
# HSTTE的ADE/FDE数据
groupnet_ade =  [18.50, 21.36, 7.96, 7.49, 9.67, 5.29, 10.41, 20.80]
hstte_ade = [15.50, 18.50, 7.41, 7.21, 9.12, 5.01, 9.91, 18.50]
hstte_mvl_ade = [12.216, 14.581, 5.840, 5.682, 7.188,3.948,7.810,14.581]


groupnet_fde = [32.69, 35.95, 12.97, 15.24, 16.52, 8.42, 17.31, 34.57]
hstte_fde =  [28.69, 32.05, 12.44, 14.11, 14.22, 7.88, 15.72, 30.02]
hstte_mvl_fde = [25.924, 28.960, 11.240, 12.749, 12.849, 7.120, 14.204, 27.126]

barWidth = 0.20
spacing = 0.05
r1 = np.arange(len(groupnet_ade))
r2 = [x + barWidth + spacing for x in r1]
r3 = [x + barWidth + spacing for x in r2]
# 字体大小
label_fontsize = 50
y_tick_fontsize = 50
legend_fontsize = 50
# x_label_fontsize = 20
x_tick_fontsize = 50

# 使用您提供的图中的颜色
# colors = ['#4E79A7', '#59A14E', '#76B4E3']
#colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F'] # 暖色
colors = ['#65AE65', '#63A0CB', '#FFA657']
# colors = ['#60ACFC', '#5BC49F', '#FEB64D'] 撞色
# colors = ['#60ACFC', '#32D3EB','#5BC49F'] 冷色
# 画ADE图
plt.figure(figsize=(30, 14))
plt.bar(r1, groupnet_ade, color=colors[0], width=barWidth, edgecolor='grey', label='PECNet ADE')
plt.bar(r2, hstte_ade, color=colors[1], width=barWidth, edgecolor='grey', label='Dual-TT ADE')
plt.bar(r3, hstte_mvl_ade, color=colors[2], width=barWidth, edgecolor='grey', label='Dual-TT+MGTP ADE')
#plt.xlabel('Teams', fontweight='bold', fontsize=x_label_fontsize)
plt.ylabel('ADE (m)', fontsize=label_fontsize)
plt.xticks([r + barWidth + spacing for r in range(len(groupnet_ade))], teams,fontsize=x_tick_fontsize)
plt.yticks(fontsize=y_tick_fontsize)
plt.ylim(3.5, 23)
plt.legend(loc="upper left",fontsize=legend_fontsize,bbox_to_anchor=(0.5, 1))
plt.tight_layout()
new_file_path1 = os.path.join("/mnt/sda/euf1szh/STAR/result/"+"_SDD_Split_ADE.png")
plt.savefig(new_file_path1, dpi=300)

# 画FDE图
plt.figure(figsize=(30, 14))
plt.bar(r1, groupnet_fde, color=colors[0], width=barWidth, edgecolor='grey', label='PECNet FDE')
plt.bar(r2, hstte_fde, color=colors[1], width=barWidth, edgecolor='grey', label='Dual-TT FDE')
plt.bar(r3, hstte_mvl_fde, color=colors[2], width=barWidth, edgecolor='grey', label='Dual-TT+MGTP FDE')
#plt.xlabel('Teams', fontweight='bold', fontsize=x_label_fontsize)
plt.ylabel('FDE (m)', fontsize=label_fontsize)
plt.xticks([r + barWidth + spacing for r in range(len(groupnet_fde))], teams,fontsize=x_tick_fontsize)
plt.yticks(fontsize=y_tick_fontsize)
plt.ylim(6.5, 38)
plt.legend(loc="upper left",fontsize=legend_fontsize,bbox_to_anchor=(0.5, 1))
plt.tight_layout()
new_file_path2 = os.path.join("/mnt/sda/euf1szh/STAR/result/"+"_SDD_split_FDE.png")
plt.savefig(new_file_path2, dpi=300)



