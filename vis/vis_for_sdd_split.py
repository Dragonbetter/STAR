import matplotlib.pyplot as plt
import numpy as np

# 场景名称
scenes = ["Gates", "Little", "Nexus", "Coupa", "Bookstore", "DeathCircle", "Quad", "Hyang"]
scenes = ["Gates", "Little", "Nexus", "Coupa", "Bookstore", "Quad", "Hyang", "DeathCircle"]
# HSTTE的ADE/FDE数据
hstte_ade = [18.50, 21.36, 7.96, 7.49, 9.67, 5.29, 10.41, 20.80]
hstte_fde = [32.69, 35.95, 12.97, 15.24, 16.52, 8.42, 17.31, 34.57]

# HSTTE+MVL的ADE/FDE数据
hstte_mvl_ade = [15.50, 18.50, 7.41, 7.21, 9.12, 5.01, 9.91, 18.50]
hstte_mvl_fde = [28.69, 32.05, 12.44, 14.11, 14.22, 7.88, 15.72, 30.02]
# 差值
hstte_ade_cha = [hstte_ade[i]-hstte_mvl_ade[i] for i in range(len(hstte_ade))]
hstte_fde_cha = [hstte_fde[i]-hstte_mvl_fde[i] for i in range(len(hstte_fde))]

label_fontsize = 44
y_tick_fontsize = 40
legend_fontsize = 40
# x_label_fontsize = 20
x_tick_fontsize = 44
# 设置柱状图宽度和位置
width = 0.30
x = np.arange(len(scenes))

fig, ax = plt.subplots(figsize=(30, 14))
#colors = ['#32d3eb','#60acfc','#feb64d','#ff7c7c','#5bc49f','#9287e7']
colors = ['#65AE65', '#63A0CB', '#FFA657','#FF927F']
colors = ['#65AE65', '#FFA657', '#63A0CB','#FF927F']
#colors = ['#32d3eb','#feb64d',  '#5bc49f','#ff7c7c']
# 绘制ADE的差值柱状图
rects1 = ax.bar(x - width/2, hstte_mvl_ade, width,color=colors[0], label='Dual-TT+MGTP ADE')
rects2 = ax.bar(x - width/2, hstte_ade_cha, width, color=colors[1],bottom=hstte_mvl_ade, label='Dual-TT ADE drop')

# 绘制FDE的差值柱状图
rects3 = ax.bar(x + width/2, hstte_mvl_fde, width, color=colors[2],label='Dual-TT+MGTP FDE')
rects4 = ax.bar(x + width/2, hstte_fde_cha, width, color=colors[3],bottom=hstte_mvl_fde, label='Dual-TT FDE drop')

# 设置标题、X轴和Y轴的标签
#ax.set_xlabel('Scenes',fontsize=label_fontsize)
ax.set_ylabel('Performance(pixel)',fontsize=label_fontsize)
#ax.set_title('Comparison between HSTTE and HSTTE+MVL',fontsize=16)
ax.set_xticks(x,fontsize=x_tick_fontsize)
ax.tick_params(axis='y',labelsize=y_tick_fontsize)
ax.set_xticklabels(scenes,fontsize=x_tick_fontsize)
ax.legend(loc='upper right', bbox_to_anchor=(0.90, 0.99), ncol=1,fontsize=legend_fontsize)
plt.ylim(0, 37)
# 展示柱状图
plt.show()


