import matplotlib.pyplot as plt

# 第一幅图片的数据
data_T = {
    1: (14.667, 33.431),
    2: (11.29, 19.369),
    3: (10.82, 18.97),
    4: (10.13, 17.05),
    5: (11.47, 20.184)
}
data_S ={
    1:(16.453,35.25),
    2:(11.68,22.24),
    3:(10.62,18.76),
    4:(10.13,17.05),
    5:(9.858,16.24)
}
label_fontsize = 20
y_tick_fontsize = 18
legend_fontsize = 20
x_tick_fontsize = 20
maker_size = 7
line_width = 2
# 提取ADE和FDE数据
ADE_T = [data_T[key][0] for key in data_T]
FDE_T = [data_T[key][1] for key in data_T]
ADE_S = [data_S[key][0] for key in data_S]
FDE_S = [data_S[key][1] for key in data_S]
# 第二幅图片的数据
lambda_values = [1,2,3,4,5]

# 绘制第1幅图片的数据
plt.figure()
plt.plot(lambda_values, ADE_T, '-o',label='ADE',markersize=maker_size, linewidth=line_width)
plt.plot(lambda_values, FDE_T,'-o', label='FDE',markersize=maker_size, linewidth=line_width)
# plt.xlabel('T',fontsize=15)
plt.xticks(lambda_values,fontsize=x_tick_fontsize)
plt.yticks(fontsize=y_tick_fontsize)
plt.ylabel('Performance',fontsize=label_fontsize)
plt.legend(loc="upper right",fontsize=legend_fontsize)
#plt.title('Sensitivity of T on SDD')
plt.show()

# 绘制第2幅图片的数据
plt.figure()
plt.plot(lambda_values, ADE_S,'-o', label='ADE',markersize=maker_size, linewidth=line_width)
plt.plot(lambda_values, FDE_S,'-o', label='FDE',markersize=maker_size, linewidth=line_width)
# plt.xlabel('T',fontsize=15)
plt.xticks(lambda_values,fontsize=x_tick_fontsize)
plt.yticks(fontsize=y_tick_fontsize)
plt.ylabel('Performance',fontsize=label_fontsize)
plt.legend(loc="upper right",fontsize=legend_fontsize)
#plt.title('Sensitivity of S on SDD')
plt.show()
