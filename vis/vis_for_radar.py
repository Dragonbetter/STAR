import matplotlib.pyplot as plt
import numpy as np

# 场景名称和数据
labels = ['Bookstore', 'Coupa', 'DeathCircle', 'Gates', 'Hyang', 'Little', 'Nexus', 'Quad']
AN = [22.12, 17.83, 51.64, 19.79, 28.33, 10.83, 19.67, 15.78]
AV = [31.86, 21.41, 54.60, 54.65, 34.25, 68.72, 23.91, 24.17]
AA = [0.200, 0.288, 0.261, 0.438, 0.088, 0.705, 0.034, 0.124]

# 计算角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)


# 在绘图之前，应用对数缩放
# AN_log = np.log10(AN)
# AV_log = np.log10(AV)
# AA_log = [np.log10(value + 0.001) for value in AA]
# AN=AN_log
# AV=AV_log
# AA=AA_log
# 然后在雷达图上使用这些_log变量
# ... [rest of the radar plotting code]

# 添加一个小的函数来动态调整文本位置
def adjust_text_position(angle, value):
    offset = 0.1
    if angle > np.pi / 2 and angle < 3 * np.pi / 2:
        h_align = 'right'
        v_align = 'bottom' if angle < np.pi else 'top'
    else:
        h_align = 'left'
        v_align = 'top' if angle < np.pi / 2 else 'bottom'
    if labels[np.argmin(np.abs(angles - angle))] in ['Hyang', 'Bookstore', 'Coupa', 'DeathCircle', 'Nexus']:
        value += max(AN) * offset
    return h_align, v_align, value


# 为了使雷达图一圈封闭起来，需要再加上第一个值
AN = np.concatenate((AN, [AN[0]]))
AV = np.concatenate((AV, [AV[0]]))
AA = np.concatenate((AA, [AA[0]]))
angles = np.concatenate((angles, [angles[0]]))

legend_size = 20
text_size = 15
label_size = 19
# ...
# 绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, AN, 'o-', linewidth=2, label="Density")
ax.fill(angles, AN, alpha=0.5)
ax.plot(angles, AV, 'o-', linewidth=2, label="Speed")
ax.fill(angles, AV, alpha=0.5)
#ax.plot(angles, AA, 'o-', linewidth=2, label="Acceleration")
#ax.fill(angles, AA, alpha=0.25)

# 添加标签
original_angles = angles[:-1]  # 去除扩展的角度
ax.set_thetagrids(original_angles * 180 / np.pi, labels, fontsize=label_size)
# ax.set_title("Analysis of Density, Speed, and Acceleration across Scenarios")
ax.grid(True)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.13), fontsize=legend_size)
ax.set_yticklabels([])

# 标注数值
for angle, an, av in zip(angles, AN[:-1], AV[:-1]):
    ax.text(angle, an, f"{an:.2f}", ha='center', va='center', fontsize=text_size)
    h_align, v_align, adjusted_av = adjust_text_position(angle, av)
    ax.text(angle, adjusted_av, f"{av:.2f}", ha=h_align, va=v_align, fontsize=text_size)
plt.show()

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, polar=True)
ax2.plot(angles, AA, 'o-', linewidth=2, label="Acceleration", color='#5BC49F')
ax2.fill(angles, AA, alpha=0.4)
original_angles2 = angles[:-1]  # 去除扩展的角度
ax2.set_thetagrids(original_angles2 * 180 / np.pi, labels, fontsize=label_size)
# ax2.set_title("Analysis of Acceleration across Scenarios")
ax2.grid(True)
ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=legend_size)
ax2.set_yticklabels([])

# 标注数值
aa_offset = max(AA) * 0.05  # 设置一个小的偏移量，例如最大值的5%
for angle, aa in zip(angles, AA[:-1]):
    ax2.text(angle, aa + aa_offset, f"{aa:.2f}", ha='center', va='center', fontsize=text_size)
plt.show()
