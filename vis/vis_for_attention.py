import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


"""
可视化不同阶段的attention值：
1）输入：
updated_batch_pednum：torch.Size([14])
nei_list：torch.Size([20, 162, 162])
nodes_current ： torch.Size([20, 162, 2])
past_att_dict ： torch.Size([162, 8, 8])
diverse_pred_traj：torch.Size([162, 20, 12, 2])
2）步骤：
a) 依据updated_batch_pednum中的值将上述的另外其他四个值进行分割，从而分成S个组，S个组中的agent序列总和等于B
b) 在每一个组中重复操作：
    依据帧去组织数据 形成dataframe 每一个agent的数据都组织为【frame，x,y,att_temporal,att_spatial】
    其中att——temporal为相对于序列最后一帧的值；
    相应的att-spatial为相对于指定序列上该帧的值。
c) 依据每个agent的值 逐步画出每条轨迹的值
   包括4幅图
"""

def draw_trajectory_for_Temporal(group_list,batch_id,save_loc,image_name,colormap = 'Reds',colorfuture='deepskyblue'):
    obs_length = 8
    for i, data in enumerate(group_list):
        # 先画对应的轨迹可视化数据：
        plt.clf()
        # todo 初步依据观察到的eth-ucy数据设置的 后续再考量如何合适的选取
        ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
        # ax.axis('off')  # 用于关闭坐标轴的显示
        ax.grid(False)
        # 逐步画完所有的对应图于一张图上。==》
        for i in range(data['trajectory'].shape[1]):
            obs_data = data['trajectory'][:obs_length,i,:]
            # 依据传入的name设置对应的obs-data
            if image_name == 'TS_Temporal':
                obs_attn = data['attn_TS_Temporal'][i,:]
            elif image_name == 'ST_Temporal':
                obs_attn = data['attn_ST_Temporal'][i,:]
            # 根据c的值设置颜色映射。这里使用红色为基色，通过调整alpha实现渐变效果
            # 另一种方法是使用亮度或饱和度的变化来实现基于红色的渐变
            cmap = plt.get_cmap(colormap)  # 获取colormap
            norm = Normalize(vmin=obs_attn.min(), vmax=obs_attn.max())  # 标准化c的值到[0, 1]
            sm = ScalarMappable(norm=norm, cmap=cmap)
            # 仅仅为了演示 不用画那么多的未来轨迹点 此处展示4个即可 ！！ 
            future_data = data['trajectory'][obs_length-1:obs_length+4,i,:]
            plt.scatter(obs_data[:, 0], obs_data[:, 1], color=sm.to_rgba(obs_attn),marker='o', s=10) 
            for i in range(obs_length - 1):  # 画线
                plt.plot([obs_data[i, 0],obs_data[i + 1, 0]],[obs_data[i, 1],obs_data[i + 1, 1]], color=cmap(0.5), alpha=0.5, linewidth=1)
            draw_points_and_line(future_data, colorfuture, 'o')
        # 保存绘画 ==>save 在result/vis/eth/
        new_file_path2 = os.path.join(save_loc+"batchid_"+str(batch_id)+"_num_"+str(data['trajectory'].shape[1])+
                       '_'+image_name+".png")
        plt.savefig(new_file_path2, dpi=300)



def draw_points_and_line(data, color,shape):
    # 输入的data为（length，2）
    length = data.shape[0]
    # 绘制散点
    plt.scatter(data[:, 0], data[:, 1], color=color,marker=shape, s=10, alpha=1)  # 散点的大小为 10，不透明度为 1
    for i in range(length - 1):  # 画线
        # 将轨迹中相邻两个点的坐标连接起来，并使用不同的颜色和透明度绘制连线
        # points = [(data[i, 0], data[i, 1]), (data[i + 1, 0], data[i + 1, 1])]
        # (x, y) = zip(*points)
        plt.plot([data[i, 0],data[i + 1, 0]],[data[i, 1],data[i + 1, 1]], color=color, alpha=0.5, linewidth=1)


def draw_attention_map(updated_batch_pednum,nei_list,nodes_current,past_att_dict,diverse_pred_traj,batch_id):
    # 0.预定义一些参数值
    save_loc = "/mnt/sda/euf1szh/STAR/result/vis_atten/eth"
    color_select = ['Reds', 'Green', 'Blues','Greys','Plasma','Viridis','Cividis']
    # 'Plasma'：具有高亮度的彩色系列，适合科技和能量主题 'Viridis'：现代感的绿蓝黄渐变，对于色盲友好 'Cividis'：黄色到深蓝的渐变，也是色盲友好选项
    color_truth_past,color_truth_pred, color_pred = 'tomato','deepskyblue','white'
    
    # 1. tensor to cpu numpy array
    updated_batch_pednum = updated_batch_pednum.cpu().detach().numpy()
    # nei_list = nei_list.cpu().detach().numpy()
    nodes_current = nodes_current.cpu().detach().numpy()
    diverse_pred_traj = diverse_pred_traj.cpu().detach().numpy()
    # 2. 依据updated_batch_pednum中的值将上述的另外其他四个值进行分割，从而分成S个组，S个组中的agent序列总和等于B
    group_list = [{} for _ in range(len(updated_batch_pednum) )]
    # 遍历数组A，并使用A中的数值来分割数组B
    start_idx = 0 
    for i , size in enumerate(updated_batch_pednum):
        end_idx = start_idx + int(size)
        # temp_nei_list = nei_list[:,start_idx:end_idx,:]
        # temp_nei_list = nei_list[:,:,start_idx:end_idx]
        # group_list[i]['nei_list'] = temp_nei_list
        group_list[i]['trajectory'] = nodes_current[:,start_idx:end_idx,:]
        # 注意力矩阵中的每一列代表了作为“键”（key）的特定单词如何被所有“查询”（query）单词关注；
        # 因此，最后一列显示了每个单词（作为查询）对第8个单词（作为键）的注意力分布
        temp_att_TS = past_att_dict['TS_temporal'][start_idx:end_idx,:,:].cpu().detach().numpy()
        group_list[i]['attn_TS_Temporal'] = temp_att_TS[:,:,-1]
        temp_att_ST = past_att_dict['ST_temporal'][start_idx:end_idx,:,:].cpu().detach().numpy()
        group_list[i]['attn_ST_Temporal'] = temp_att_ST[:,:,-1]
        temp_TS_Spatial = past_att_dict['TS_spatial'][:,start_idx:end_idx,:]
        temp_TS_Spatial = temp_TS_Spatial[:,:,start_idx:end_idx]
        group_list[i]['attn_TS_Spatial'] = temp_TS_Spatial
        temp_ST_Spatial = past_att_dict['ST_spatial'][:,start_idx:end_idx,:]
        temp_ST_Spatial = temp_ST_Spatial[:,:,start_idx:end_idx]
        group_list[i]['attn_ST_Spatial'] = temp_ST_Spatial
        group_list[i]['diverse_pred_trajectory'] = diverse_pred_traj[start_idx:end_idx,:,:,:]
        start_idx = end_idx
    # 3. 在每一个组中重复操作：
    # 时间的
    print("开始分析时间注意力可视化图")
    draw_trajectory_for_Temporal(group_list,batch_id,save_loc,image_name='TS_Temporal',colormap=color_select[0],colorfuture=color_truth_pred)
    print("完成TS——Temporal的绘画")
    draw_trajectory_for_Temporal(group_list,batch_id,save_loc,image_name='ST_Temporal',colormap=color_select[0],colorfuture=color_truth_pred)
    print("完成ST——Temporal的绘画")
    # 空间的

