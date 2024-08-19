import matplotlib.pyplot as plt
import os
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
if __name__ == '__main__':
    # 0.192/0.364 ==>当为1时 即未取对应的值 
    data_T = {
        1:(0.192,0.364),
        2:(0.189,0.358),
        3:(0.183,0.354),
        4:(0.182,0.351),
        5:(0.196,0.369)
    }
    # 0.192/0.364 ==》 
    data_S= {
        1: (0.201,0.421),
        2: (0.194,0.369),
        3: (0.188,0.362),
        4: (0.182,0.351),
        5: (0.178,0.345)
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
    plt.figure(figsize=(9, 6))
    plt.plot(lambda_values, ADE_T, '-o',label='ADE',markersize=maker_size, linewidth=line_width)
    plt.plot(lambda_values, FDE_T,'-o', label='FDE',markersize=maker_size, linewidth=line_width)
    # plt.xlabel('T',fontsize=15)
    plt.xticks(lambda_values,fontsize=x_tick_fontsize)
    plt.yticks(fontsize=y_tick_fontsize)
    plt.ylabel('Performance',fontsize=label_fontsize)
    plt.legend(loc="upper right",fontsize=legend_fontsize,bbox_to_anchor=(1, 0.8))
    #plt.title('Sensitivity of T on SDD')
    new_file_path1 = os.path.join("/mnt/sda/euf1szh/STAR/result/"+"_Sensitivity_of_T.png")
    plt.savefig(new_file_path1, dpi=300)

    # 绘制第2幅图片的数据
    plt.figure(figsize=(9, 6))
    plt.plot(lambda_values, ADE_S,'-o', label='ADE',markersize=maker_size, linewidth=line_width)
    plt.plot(lambda_values, FDE_S,'-o', label='FDE',markersize=maker_size, linewidth=line_width)
    # plt.xlabel('T',fontsize=15)
    plt.xticks(lambda_values,fontsize=x_tick_fontsize)
    plt.yticks(fontsize=y_tick_fontsize)
    plt.ylabel('Performance',fontsize=label_fontsize)
    plt.legend(loc="upper right",fontsize=legend_fontsize)
    #plt.title('Sensitivity of S on SDD')
    new_file_path2 = os.path.join("/mnt/sda/euf1szh/STAR/result/"+"_Sensitivity_of_S.png")
    plt.savefig(new_file_path2, dpi=300)
