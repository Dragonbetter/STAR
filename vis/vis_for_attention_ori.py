# uncompyle6 version 3.9.1
# Python bytecode version base 3.9.0 (3425)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /mnt/sda/euf1szh/STAR/vis/vis_for_attention.py
# Compiled at: 2024-03-25 23:05:15
# Size of source mod 2**32: 27819 bytes
#!/usr/bin/env python
# visit https://tool.lu/pyc/ for more information
# Version: Python 3.9

import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
import cv2
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
data_dirs = [
    './data/eth/eth',
    './data/eth/hotel',
    './data/ucy/zara/zara01',
    './data/ucy/zara/zara02',
    './data/ucy/univ/students001',
    './data/ucy/univ/students003',
    './data/ucy/univ/uni_examples',
    './data/ucy/zara/zara03']
video_dirs = {
    'eth': './data/eth/eth/seq_eth.avi',
    'hotel': './data/eth/hotel/seq_hotel.avi',
    'zara1': './data/ucy/zara/zara01/crowds_zara01.avi',
    'zara2': './data/ucy/zara/zara02/crowds_zara02.avi',
    'univ03': './data/ucy/univ/students003/students003.avi' }
H = {
    'eth': np.array([
        [0.0281287,0.00200919,-4.66936],
        [0.000806257,0.0251955,-5.06088],
        [0.000345554,9.25122e-05,0.462553]]),
    'hotel': np.array([
        [0.0110482, 0.000669589,-3.32953],
        [-0.0015966,0.0116324,-5.39514],
        [0.000111907,1.36174e-05,0.542766]]),
    'zara1': np.array([
        [-0.0259565,-5.15728e-18,7.83887],
        [-0.00109539,0.0216643,-10.0323],
        [1.95401e-20,4.21714e-19,1]]),
    'zara2': np.array([
        [-0.0259565,-5.15728e-18,7.83887],
        [-0.00109539,0.0216643,-10.0323],
        [1.95401e-20,4.21714e-19,1]
        ]),
    'univ01': np.array([
        [-0.0230028,0.000537419,8.66573],
        [-0.000527538,0.0195652,-6.08892],
        [0,-0,1]]),
    'univ03': np.array([
        [-0.0230028,0.000537419,8.66573],
        [-0.000527538,0.0195652,-6.08892],
        [0,-0,1]
        ]),
    'univ_examples': np.array([
       [-0.0230028,0.000537419,8.66573],
        [-0.000527538,0.0195652,-6.08892],
        [0,-0,1]]),
    'zara03': np.array([
       [-0.0259565,-5.15728e-18,7.83887],
        [-0.00109539,0.0216643,-10.0323],
        [1.95401e-20,4.21714e-19,1]]) }

def draw_trajectory_for_Temporal(group_list, batch_id, save_loc, image_name, colormap= 'Reds', colorfuture='deepskyblue'):
    obs_length = 8
    for i, data in enumerate(group_list):
        plt.clf()
        # todo 初步依据观察到的eth-ucy数据设置的 后续再考量如何合适的选取
        ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
        # ax.axis('off')  # 用于关闭坐标轴的显示
        ax.grid(False)
        for i in range(data['trajectory'].shape[1]):
            obs_data = data['trajectory'][:obs_length,i,:]
            # 依据传入的name设置对应的obs-data
            if image_name == 'TS_Temporal':
                obs_attn = data['attn_TS_Temporal'][i,:]
            elif image_name == 'ST_Temporal':
                obs_attn = data['attn_ST_Temporal'][i,:]
            # 根据c的值设置颜色映射。这里使用红色为基色，通过调整alpha实现渐变效果
            # 另一种方法是使用亮度或饱和度的变化来实现基于红色的渐变
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=obs_attn.min(), vmax=obs_attn.max())  # 标准化c的值到[0, 1]
            sm = ScalarMappable(norm=norm, cmap=cmap)
            # 仅仅为了演示 不用画那么多的未来轨迹点 此处展示4个即可 ！！ 
            future_data = data['trajectory'][obs_length-1:obs_length+4,i,:]
            plt.scatter(obs_data[:, 0], obs_data[:, 1], color=sm.to_rgba(obs_attn),marker='o', s=10) 
            for i in range(obs_length - 1):
                plt.plot([obs_data[(i, 0)],obs_data[(i + 1, 0)]], 
                         [obs_data[(i, 1)],obs_data[(i + 1, 1)]], color=cmap(0.5), alpha=0.5, linewidth=1)
            draw_points_and_line(future_data, colorfuture, 'o')
        new_file_path2 = os.path.join(save_loc+"batchid_"+str(batch_id)+"_num_"+str(data['trajectory'].shape[1])+
                       '_'+image_name+".png")
        plt.savefig(new_file_path2, dpi=300)


def draw_trajectory_on_scene_for_Temporal(group_list, batch_id, save_loc, image_name, colormap, colorfuture = ('Reds', 'deepskyblue')):
    obs_length = 8
    for i, data in enumerate(group_list):
        # 只分析对应的eth场景 hotel作为test
        if data['scene'] in ('zara1', 'zara2', 'univ01', 'univ03', 'univ_examples', 'zara03'):
            continue

        H_scene = H[data['scene']]
        # 获取单应性矩阵的逆矩阵
        H_inv = np.linalg.inv(H_scene)
        # # 扩展B以包含齐次坐标
        B_homogeneous = np.ones((data['trajectory'].shape[0], data['trajectory'].shape[1], 3))
        # 交换x和y的数据 以适配对应的eth的H矩阵设计
        B_homogeneous[:, :, 0] = data['trajectory'][:, :, 1]
        B_homogeneous[:, :, 1] = data['trajectory'][:, :, 0]
        # 重塑B，以便进行矩阵乘法。新形状将是[帧数*人数, 3]
        B_reshaped = B_homogeneous.reshape(-1, 3).T
        # 应用单应性矩阵的逆进行转换
        pixel_positions_homogeneous = np.dot(H_inv, B_reshaped)
        # 归一化坐标（将齐次坐标转换回二维坐标）
        pixel_positions = pixel_positions_homogeneous[:2, :] / pixel_positions_homogeneous[2, :]
        # 重塑回原始帧数和人数的形状
        pixel_positions_reshaped = pixel_positions.T.reshape(data['trajectory'].shape[0], data['trajectory'].shape[1], 2)
        data['pixel_trajectory'] = pixel_positions_reshaped
        data['pixel_trajectory'] = data['pixel_trajectory'][:, :, [1,0]]
         # 3. 获取场景 
        video_path = video_dirs[data['scene']]
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print('Error: Could not open video.')
        else:
            # 设置视频的当前位置
            video.set(cv2.CAP_PROP_POS_FRAMES, data['frame'])
            # 读取当前帧
            ret, frame = video.read()
            if ret:
                 # 如果成功读取帧，显示或处理帧
                img = frame.copy()
                height, width, _ =  img.shape
                # 这里转换颜色空间从 OpenCV读取帧 默认 BGR 到 Matplotlib RGB
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 # 先画对应的轨迹可视化数据：
                plt.clf()
                # todo 初步依据观察到的eth-ucy数据设置的 
                # 问题？ETH-UCY里的像素坐标基于的原点坐标应该是左下角？
                ax = plt.axes(xlim=(0, width), ylim=(0, height))
                ax.grid(False)
                # 逐步画完所有的对应图于一张图上。==》
                for i in range(data['pixel_trajectory'].shape[1]):
                    obs_data = data['pixel_trajectory'][:obs_length,i,:]
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
                    future_data = data['pixel_trajectory'][obs_length-1:obs_length+4,i,:]
                    plt.scatter(obs_data[:, 0], obs_data[:, 1], color=sm.to_rgba(obs_attn),marker='o', s=10) 
                    for i in range(obs_length - 1):  # 画线
                        plt.plot([obs_data[i, 0],obs_data[i + 1, 0]],[obs_data[i, 1],obs_data[i + 1, 1]], color=cmap(0.5), alpha=0.5, linewidth=1)
                    draw_points_and_line(future_data, colorfuture, 'o')
                # flipped_frame = cv2.flip(rgb_frame, 0)  # 0 表示上下翻转
                # 坐标轴的原点似乎有问题 ？？
                plt.imshow(rgb_frame, zorder=0,alpha=1,origin='lower')
                # 保存绘画 ==>save 在result/vis/eth/
                new_file_path2 = os.path.join(save_loc+"scene_"+str(data['scene'])+"_frame_"+str(data['frame'])+"_num_"+str(data['trajectory'].shape[1])+
                       '_'+image_name+".png")
                plt.savefig(new_file_path2, dpi=300)
            else:
                print("Error: Could not read frame.")


def draw_trajectory_on_scene_for_Spatial(group_list, batch_id, save_loc, image_name, colormap, colorfuture = ('Reds', 'deepskyblue')):
    obs_length = 8
    for i, data in enumerate(group_list):
        # 1.先筛选出对应的可以可视化的场景
        #if data['scene'] in ['hotel','zara01','zara02','univ01','univ03','univ_examples','zara03']:
        #if data['scene'] in ['univ01','univ_examples','zara03']:
        # 只可视化eth和hotel的数据 
        if data['scene'] in ['zara01','zara02','univ01','univ03','univ_examples','zara03']:
            continue
        # 该类型的数据不进行绘画
        # 2.获取单应性矩阵的逆矩阵
        H_scene = H[data['scene']]
        H_inv = np.linalg.inv(H_scene)
        # 利用单应性矩阵将轨迹的数据从世界坐标系转化到像素坐标系下
        B_homogeneous = np.ones((data['trajectory'].shape[0], data['trajectory'].shape[1], 3))
        # 需要注意的是此处的单应性矩阵对应的是 (y,x)
        # B_homogeneous[:, :, :2] = data['trajectory']  # 将原始的x,y坐标复制过来
        B_homogeneous[:, :, 0] = data['trajectory'][:,:,1]
        B_homogeneous[:, :, 1] = data['trajectory'][:,:,0]
        # 重塑B，以便进行矩阵乘法。新形状将是[帧数*人数, 3]
        B_reshaped = B_homogeneous.reshape(-1, 3).T  # 转置是为了匹配矩阵乘法的需求

        # 应用单应性矩阵的逆进行转换
        pixel_positions_homogeneous = np.dot(H_inv, B_reshaped)

        # 归一化坐标（将齐次坐标转换回二维坐标）
        pixel_positions = pixel_positions_homogeneous[:2, :] / pixel_positions_homogeneous[2, :]

        # 重塑回原始帧数和人数的形状
        pixel_positions_reshaped = pixel_positions.T.reshape(data['trajectory'].shape[0], data['trajectory'].shape[1], 2)
        data['pixel_trajectory'] = pixel_positions_reshaped
        # 将数据转换回来即有yx-》xy
        data['pixel_trajectory'] = data['pixel_trajectory'][:,:,[1,0]]
        # 3. 获取场景
        video_path = video_dirs[data['scene']]
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Error: Could not open video.")
        else:
            # 设置视频的当前位置
            video.set(cv2.CAP_PROP_POS_FRAMES, data['frame'])
            # 读取当前帧
            ret, frame = video.read()
            if ret:
                # 如果成功读取帧，显示或处理帧
                img = frame.copy()
                height, width, _ =  img.shape
                # # 这里转换颜色空间从 OpenCV读取帧 默认 BGR 到 Matplotlib RGB
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 先画对应的轨迹可视化数据：
                plt.clf()
                ax = plt.axes(xlim=(0, width), ylim=(0, height))
                # ax.axis('off')  # 用于关闭坐标轴的显示
                ax.grid(False)
                frame_index = 7
                for special_index in range(data['pixel_trajectory'].shape[1]):
                    # # 计算箭头的dx和dy（即第二帧与第一帧的坐标差值）
                    dx = data['pixel_trajectory'][frame_index + 2, :, 0] - data['pixel_trajectory'][frame_index, :, 0]
                    dy = data['pixel_trajectory'][frame_index + 2, :, 1] - data['pixel_trajectory'][frame_index, :, 1]
                    plt.figure(figsize=(10, 6))  # 新建图像，大小可以根据需要调整
                    # 仍然取特殊列进行分析
                    if image_name == 'TS_Spatial':
                        spatial_attn = data['attn_TS_Spatial'][7, :, special_index]
                    elif image_name == 'ST_Spatial':
                        spatial_attn = data['attn_ST_Spatial'][7, :, special_index]
                    spatial_attn = spatial_attn.cpu().detach().numpy()
                    circle_sizes = spatial_attn * 1500
                    
                    for i in range(data['pixel_trajectory'].shape[1]):
                        if i != special_index:
                            plt.scatter(data['pixel_trajectory'][frame_index, i, 0], data['pixel_trajectory'][frame_index, i, 1], s=circle_sizes[i], facecolors='none', edgecolors='r')
                             # 绘制箭头  scale的值越大，箭头就越小；反之，值越小，箭头就越大。width 控制箭头的粗细
                            plt.quiver(data['pixel_trajectory'][frame_index, i, 0], data['pixel_trajectory'][frame_index, i, 1], dx[i], dy[i], angles='xy', scale_units='xy', scale=1.2, color='deepskyblue' , width=0.003)
                        else:
                            special_x, special_y = data['pixel_trajectory'][frame_index, special_index, :]                    
                            plt.scatter(special_x, special_y, s=circle_sizes[special_index], facecolors='none',  edgecolors='deepskyblue')
                            plt.quiver(data['pixel_trajectory'][frame_index, special_index, 0], data['pixel_trajectory'][frame_index, special_index, 1], dx[special_index], dy[special_index],angles='xy', scale_units='xy', scale=1.0, color='deepskyblue' , width=0.003)
                    plt.imshow(rgb_frame,zorder=0,alpha=1,origin='lower')
                    new_file_path2 = os.path.join(save_loc+"scene_"+str(data['scene'])+"_frame_"+str(data['frame'])+"_id_"+
                        '_'+str(special_index)+"_" + image_name+".png")
                    plt.savefig(new_file_path2, dpi=300)
            else:
                print('Error: Could not read frame.')



def draw_points_and_line(data, color, shape):
    # 输入的data为（length，2）
    length = data.shape[0]
    # 绘制散点
    plt.scatter(data[:, 0], data[:, 1], color=color,marker=shape, s=10, alpha=1)  # 散点的大小为 10，不透明度为 1
    for i in range(length - 1):  # 画线
        # 将轨迹中相邻两个点的坐标连接起来，并使用不同的颜色和透明度绘制连线
        # points = [(data[i, 0], data[i, 1]), (data[i + 1, 0], data[i + 1, 1])]
        # (x, y) = zip(*points)
        plt.plot([data[i, 0],data[i + 1, 0]],[data[i, 1],data[i + 1, 1]], color=color, alpha=0.5, linewidth=1)


def draw_attention_map(updated_batch_pednum, nei_list, nodes_current, past_att_dict, diverse_pred_traj, batch_id, test_set):
    # 0.预定义一些参数值
    save_loc = "/mnt/sda/euf1szh/STAR/result/vis_atten/hotel/"
    color_select = ['Reds', 'Green', 'Blues','Greys','Plasma','Viridis','Cividis']
    # 'Plasma'：具有高亮度的彩色系列，适合科技和能量主题 'Viridis'：现代感的绿蓝黄渐变，对于色盲友好 'Cividis'：黄色到深蓝的渐变，也是色盲友好选项
    color_truth_past,color_truth_pred, color_pred = 'tomato','deepskyblue','white'
     # 1. tensor to cpu numpy array
    updated_batch_pednum = updated_batch_pednum.cpu().detach().numpy()
    # nei_list = nei_list.cpu().detach().numpy()
    nodes_current = nodes_current.cpu().detach().numpy()
    diverse_pred_traj = diverse_pred_traj.cpu().detach().numpy()
    # 2. 依据updated_batch_pednum中的值将上述的另外其他四个值进行分割，从而分成S个组，S个组中的agent序列总和等于B
    train_set = [i for i in range(len(data_dirs))]
    # 断言 确认相应的test-set在已有数据集中 检查代码中使用的数据集名称是否正确。
    DATASET_NAME_TO_NUM = {'eth': 0, 'hotel': 1, 'zara1': 2, 'zara2': 3, 'univ': 4}
    skip = [6, 10, 10, 10, 10, 10, 10, 10]
    assert test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(test_set)
    # 将其转换为数字形式
    test_set = DATASET_NAME_TO_NUM[test_set]
    if test_set == 4 or test_set == 5:
        test_set_num = [4, 5]
    else:
        test_set_num = [test_set]
    # 分离train和test数据集
    NUM_TO_DATASET_NAME = ['eth','hotel','zara01','zara02','univ01', 'univ03','univ_examples','zara03']
    for x in test_set_num:
        NUM_TO_DATASET_NAME.remove(NUM_TO_DATASET_NAME[x])
        skip.remove(skip[x])
    group_list = [{} for _ in range(len(updated_batch_pednum) )]
    # 遍历数组A，并使用A中的数值来分割数组B
    start_idx = 0 
    for i , size in enumerate(updated_batch_pednum):
        scene = NUM_TO_DATASET_NAME[batch_id[i][0]]
        if scene in  ('zara01', 'zara02', 'univ01', 'univ03', 'univ_examples', 'zara03'):
            continue
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
        group_list[i]['scene'] = NUM_TO_DATASET_NAME[batch_id[i][0]]
        # frame取的应该是观察帧，即cur-frame起始帧加上对应的7*skip
        group_list[i]['frame'] = batch_id[i][1]+skip[batch_id[i][0]]*7
        start_idx = end_idx
    filtered_dict_list = [d for d in group_list if any((value != 0).all() if isinstance(value, np.ndarray) else value != 0 for value in d.values())]
    # 3. 在每一个组中重复操作：
    # 时间的
    print("开始分析时间注意力可视化图")
    draw_trajectory_on_scene_for_Spatial(filtered_dict_list,batch_id,save_loc,image_name='TS_Spatial',colormap=color_select[0],colorfuture=color_truth_pred)
    print("完成TS_spatial的绘画")
    draw_trajectory_on_scene_for_Spatial(filtered_dict_list,batch_id,save_loc,image_name='ST_Spatial',colormap=color_select[0],colorfuture=color_truth_pred)
    print("完成ST_spatial的绘画")
    draw_trajectory_on_scene_for_Temporal(filtered_dict_list,batch_id,save_loc,image_name='TS_Temporal',colormap=color_select[0],colorfuture=color_truth_pred)
    print("完成TS_Temporal的绘画")
    draw_trajectory_on_scene_for_Temporal(filtered_dict_list,batch_id,save_loc,image_name='ST_Temporal',colormap=color_select[0],colorfuture=color_truth_pred)
    print("完成ST_Tempora的绘画")