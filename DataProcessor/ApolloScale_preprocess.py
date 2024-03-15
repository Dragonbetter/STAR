import os
import numpy as np
import pickle
import random
import pandas as pd
import os
from src.Visual import SDD_traj_vis
from DataProcessor.SDD_preprocess import split_fragmented,filter_short_trajectories,sliding_window

from DataProcessor.DataProcessorFactory import DatasetProcessor_BASE

def read_trajectory_origin(folder_path):
    """
    读取原始数据
    :param filename:
    :return:
    """
    # 初始化数据字典
    data = {}
    # 定义列名
    columns = [
        'frame', 'trackId', 'label', 'x', 'y',
        'position_z', 'object_length', 'object_width', 'object_height', 'heading'
    ]
    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为.txt
        if filename.endswith('.txt'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            
            # 读取txt文件到DataFrame，假设数据是用逗号分隔的
            df = pd.read_csv(file_path, names=columns, delimiter=' ')
            
            # 使用文件名（不包括扩展名）作为字典的键
            key_name = os.path.splitext(filename)[0]
            
            # 将DataFrame存储到字典中
            data[key_name] = df
    # 现在，`data`字典包含了所有文件的数据，其中键是文件名，值是对应的DataFrame
    for key in data:
        data[key]['sceneId'] = key  # 'source'是新列的列名，存储每个DataFrame对应的文件名（即字典的键）
    # 第二步：合并所有DataFrame到一个大的DataFrame中
    all_data_df = pd.concat(data.values(), ignore_index=True)
    # 第三步：删除不需要的列
    columns_to_drop = ['position_z', 'object_length', 'object_width', 'object_height', 'heading']
    all_data_df.drop(columns=columns_to_drop, inplace=True)  
    return all_data_df


class DatasetProcessor_ApolloScale(DatasetProcessor_BASE):
    def __init__(self,args):
        """
        """
        super().__init__(args=args)
        assert self.args.dataset == "ApolloScale"
        print("正确完成真实数据集" + self.args.dataset + "的初始化过程")
            # 复用的代码结构
        """
        # 基础结构类
        # def __init__
        # def reset_batch_pointer(self, set, valid=False):
        =====数据打包处理保存类
        # def load_dict(self,data_file):
        # def load_cache(self, cachefile):
        # def pick_cache(self):
        =====数据训练过程中获取类
        # def rotate_shift_batch(self, batch_data, ifrotate=True)
        # def get_train_batch(self, idx)
        # def get_test_batch(self, idx)
        =====通用结构类 
        # def get_data_index(self, data_dict, setname, ifshuffle=True)
        # def get_data_index_single(self,seti,data_dict, setname, ifshuffle=True):
        # def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip,return_len)
        =====顶层设计类
        # def data_preprocess_for_originbatch(self, setname):
        # def data_preprocess_for_MVDGtask(self,setname):
        # def data_preprocess_for_originbatch_split(self):
        =====batch数据形成类
        # def massup_batch(self,batch_data):
        # def get_social_inputs_numpy_HIN(self)
        """
    def data_preprocess_for_origintrajectory(self,args):
        """
        完成从最原始数据（txt）到初步处理的过程
        完成原traject——preprocess 工作
        ===============================
        1.获取数据存取位置
        ./data/ApolloScale/prediction_test
        ./data/ApolloScale/prediction_train
        2.配置基本实验设置
        obs_length,future_length,
        3.数据基本格式分析：
        object_type	small vehicles	big vehicles	pedestrian	motorcyclist and bicyclist	others
        ID	1	2	3	4	5
        1）train:
        53个文件，每个文件1分钟的数据，单独训练；
        数据格式为：
         frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading.
        ==>正常处理即可
        2)test:
         包含prediction——result（预测数据）和prediction——gt(为对应的真值数据)，以及相对应的object-id数据；
        表示该物体需要被考虑。
        从200开始往后：200-205；212-217 如此划分obs；即我需要先合并相应的两个文件的数据，而后按照12进行断开，
        并依据object-id的数据，将
        """
        self.args.seq_length = 12
        self.args.obs_length = 6
        self.args.pred_length = 6
        self.SDD_skip = 1 # 不需要应该 ？
        self.args.relation_num = 1
        window_size, stride = 12,12
        train_data_path = './data/ApolloScale/prediction_train'
        test_data_path = './data/ApolloScale/prediction_test'
        train_data = read_trajectory_origin(train_data_path)
        # 将scene-id和trackid两列拼接起来，并用下划线连接，形成一个新的字符串，最后将所有字符串组成一个新的列表，作为rec&trackId列的值。
        train_data['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
                                   zip(train_data.sceneId, train_data.trackId)]
        # 创建rec-trackID2metaId的字典，用于将每个唯一的rec&trackId映射到一个唯一的metaId（整数编号）
        rec_trackId2metaId = {}
        for i, j in enumerate(train_data['rec&trackId'].unique()):
            rec_trackId2metaId[j] = i
        train_data['metaId'] = [rec_trackId2metaId[i] for i in train_data['rec&trackId']]
        train_data = train_data.drop(columns=['rec&trackId'])
        print('切分断开的轨迹')
        data_continues = split_fragmented(train_data)
        # print('降采样') 不需要降采样 因为此处的数据是
        # data_downsample = downsample_all_frame(df=data_continues, step=SDD_skip)
        print('滤除过短的轨迹')
        data_filter_short = filter_short_trajectories(data_continues, threshold=window_size)
        print('对数据进行分组，划定时间窗口')
        data_sliding = sliding_window(data_filter_short, window_size=window_size, stride=stride)
        # trackID frame label x y sceneid scene metaID
        # drop track_id
        data_sliding = data_sliding.drop(columns=['trackId'])
        # 划分训练和测试数据集 
        self.apollo_train_data = data_sliding
        test_data = data_sliding
        self.apollo_test_data = test_data

    def data_preprocess_for_transformer(self, setname):
        if setname=="train":
            data = self.apollo_train_data
            data_file = self.train_data_file
        elif setname=="test":
            data = self.apollo_test_data
            data_file = self.test_data_file
        else:
            raise ValueError("setname must be 'train' or 'test'")
        print("处理__" + setname + "__数据")
        # 第二步：提取frame-dict与ped-dict
        SDD_origin_data = data.to_numpy().T
        # frame,label,x,y,sceneID,metaID => [frame,metaID,y,x,label,sceneID]
        SDD_origin_data = SDD_origin_data[[0, 5, 3, 2, 1, 4], :]
        # all_frame_data = [] valid_frame_data = []numFrame_data = [] Pedlist_data = []
        frameped_dict = []  # peds id contained in a certain frame
        pedtrajec_dict = []
        scene_list = np.unique(SDD_origin_data[5, :].astype(str)).tolist()  # 场景名称列表
        pedlabel_dict = []  # trajectories of a certain ped
        for seti, scene in enumerate(scene_list):
            print('preprocess  scene ' + str(scene) + ' data')
            data = SDD_origin_data[:, SDD_origin_data[5, :] == scene]
            Pedlist = np.unique(data[1, :]).tolist()
            # numPeds = len(Pedlist)
            # Add the list of frameIDs to the frameList_data
            # Pedlist_data.append(Pedlist)
            # Initialize the list of numpy arrays for the current dataset
            # all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            # valid_frame_data.append([])
            # 整个数据集
            # numFrame_data.append([])
            # 记录了当前数据集的每个帧包含了那些行人
            frameped_dict.append({})
            # 记录了每个行人的轨迹数据 （数据集，行人id，该行人的帧，对应帧下的xy数据）
            pedtrajec_dict.append({})
            pedlabel_dict.append({})
            for ind, pedi in enumerate(Pedlist):
                if ind % 100 == 0:
                    print(ind, len(Pedlist))
                # Extract trajectories of one person 抽取单人的轨迹数据
                FrameContainPed = data[:, data[1, :] == pedi]
                # Extract ped label
                Label = FrameContainPed[4, 0]
                # Extract peds list
                FrameList = FrameContainPed[0, :].tolist()
                if len(FrameList) < 2:
                    continue
                # Add number of frames of this trajectory
                # numFrame_data[seti].append(len(FrameList))
                # Initialize the row of the numpy array
                Trajectories = []
                # For each ped in the current frame
                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                    current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                    # todo 添加label 和 scene
                    # label = FrameContainPed[4,FrameContainPed[0,:]==frame][0]
                    # scene = FrameContainPed[5,FrameContainPed[0,:]==frame][0]
                    # Add their pedID, x, y to the row of the numpy array
                    Trajectories.append([int(frame), current_x, current_y])
                    # Trajectories.append([int(frame), current_x, current_y,label,scene])
                    # 如果当前帧不在frameped_dict中，则相应的添加该帧，并将该帧包含的行人添加；记录了当前数据集的每个帧包含了那些行人
                    if int(frame) not in frameped_dict[seti]:
                        frameped_dict[seti][int(frame)] = []
                    frameped_dict[seti][int(frame)].append(pedi)
                pedtrajec_dict[seti][pedi] = np.array(Trajectories)
                # 保存对应的行人label维度
                pedlabel_dict[seti][pedi] = Label

        f = open(data_file, "wb")
        # 这两个对象序列化到文件中
        pickle.dump((frameped_dict, pedtrajec_dict, scene_list, pedlabel_dict), f, protocol=2)
        f.close()


    def data_preprocess_for_MLDGtask(self, setname):
        # 从字符串"bookstore_0"中提取出"bookstore"
        # 故而此处对应的应该基于apollo的scene-id进行处理分析
        pass

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, pedlabel_dict,data_index,scene_list,setname):
        if self.args.HIN:
            print("MLDG任务中生成的数据是基于HIN的")
            batch = self.get_seq_from_index_balance_HIN(frameped_dict=frameped_dict,pedtraject_dict=pedtraject_dict,
                        pedlabel_dict=pedlabel_dict,scene_list=scene_list,data_index=data_index, setname=setname)
        else :
            print("MLDG任务中生成的数据是基于同质图的")
            batch = self.get_seq_from_index_balance_origin(frameped_dict=frameped_dict,pedtraject_dict=pedtraject_dict,
                        pedlabel_dict=pedlabel_dict,scene_list=scene_list,data_index=data_index, setname=setname)
        return batch
        
    def get_seq_from_index_balance_origin(self, frameped_dict, pedtraject_dict, pedlabel_dict,data_index,scene_list,setname):
        """
        完成get_seq_from_index_balance / get_seq_from_index_balance_meta工作
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
                    batch_data_mass：多个（batch_data, Batch_id）
                    batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
                    nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                    seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                    nei_list_b：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
                    不同的只在于邻接矩阵的不同 单独一个 或则是 【3,20，257,257】
                    nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                    batch_pednum：list 表示该batch下每个时间窗口中的行人数量
        """
        batch_data_mass, batch_data, Batch_id = [], [], []
        ped_cnt, last_frame = 0, 0
        # 注意此处的skip 在不同数据集的差异
        skip = self.SDD_skip
        # 全局处理 混合所有train的帧 形成的windows
        for i in range(data_index.shape[1]):
            '''
            仍然是以对应窗口序列划分 例如test有1443帧，则相应的可以划分处1443个时间窗口，但需要后期依据
            '''
            cur_frame, cur_set, _ = data_index[:, i]
            cur_scene = scene_list[cur_set]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            # 计算并获取对应起始帧（子轨迹）的结束帧，由于当前的子轨迹的结束帧可能会超过数据集的范围，因此使用try-expect语句块处理这种情况
            try:
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + (self.args.seq_length - 1) * skip]) #todo 尽量后续统一skip形式

            except:
                continue
            # todo 合并起始与结束帧中包含的行人
            present_pedi = framestart_pedi | frameend_pedi
            # 如果起始帧与结束帧没有重复的行人id，则抛弃该子轨迹
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            """
            针对由起始帧和结束帧确定的窗口序列以及行人并集，遍历行人，找到该行人在起始帧与结束帧之间存在的片段；若正好全程存在，则iffull为true，
            若有空缺，则iffull为False；ifexistobs标识obs帧是否存在，并删去太短的片段（小于5）；而后去除帧号，只保留这些行人的xy坐标；添加到traject中
            而后将滤除后的行人轨迹数据保留并拼接；batch-pednum为相应的不断累计不同时间窗口轨迹数据的总值，
            """
            for ped in present_pedi:
                # cur-trajec：该行人对应的子轨迹数据（可能是完整的20，也可能小于20） iffull指示其是否满，ifexistobs指示其是否存在我们要求的观测帧
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                               cur_frame, self.args.seq_length,
                                                                               skip)
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                # 此时cur-trajec为固定的（20,3）则[:,1:]保留xy数据，略去时间数据即（20,2）-》reshape为（20,1,2）数据
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            # 按照第二个维度进行拼接，即将同一个windows中行人数据拼接在一起
            traject_batch = np.concatenate(traject, 1)
            # 基于后续叠加各个windows中的行人数据
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]
            # 该windows中的行人数量
            cur_pednum = traject_batch.shape[1]
            print(self.args.dataset + '_' + setname + '_' + str(cur_pednum))
            ped_cnt += cur_pednum
            # todo 后续基于batch-id进行数据提取
            batch_id = (cur_set, cur_frame,)
            """
            如果以当前数据集以及相应的预测帧起始的窗口中包含超过512个行人的轨迹，则将其进行拆分为两个batch，如果处于256和512之间，
            将其打包成为一个batch；如果小于256，则相应的累加其他时间窗口的轨迹数据，直到batch里的行人数大于256,将其打包为一个batch             
            """
            # todo 分数据集提取保存！！[set,batch-data]
            if cur_pednum >= self.args.batch_around_ped * 2:
                # too many people in current scene
                # split the scene into two batches
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_batch_data, cur_Batch_id = [], []
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]],
                              traject_batch[:, ind[cur_pednum // 2:, 0]]]
                for sb in Seq_batchs:
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_batch_data = self.massup_batch(cur_batch_data)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                    cur_batch_data = []
                    cur_Batch_id = []

                last_frame = i
            elif cur_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                cur_batch_data, cur_Batch_id = [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_batch_data = self.massup_batch(cur_batch_data)
                batch_data_mass.append((cur_batch_data, cur_Batch_id,))

                last_frame = i
            else:  # less pedestrian numbers <64
                # accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped:
                    # enough people in the scene
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
                    """
                    输入：多个windows的数据 （windows-num，20，windows-ped，2）
                    batch_data_mass：多个（batch_data, Batch_id）
                    batch_data：(nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

                    nodes_batch_b：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标
                    seq_list_b:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
                    nei_list_b：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
                    nei_num_b：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
                    batch_pednum：list 表示该batch下每个时间窗口中的行人数量
                    """
                    # todo 需要注意的是后续相应的异质网结构的邻接矩阵会不一样 需要特殊处理 但meatID与label一一对应 可以查询的得到
                    batch_data = self.massup_batch(batch_data)
                    if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                        batch_data_mass.append((batch_data, Batch_id,))
                    elif batch_data[4] == []:
                        print('舍弃该数值')
                    last_frame = i
                    batch_data = []
                    Batch_id = []
                else:
                    # todo batch问题 缺失轨迹预测问题 meta-learning task设计问题
                    """
                     一般都要经过累加，相应的过往的batch处理是选择固定的多个20s的场景数据，而此处区别则是每个batch中包含的20s场景数是不同的
                     其以该batch中的人（轨迹）数量为准，直到累加超过阈值；（好处可能是解决了单个轨迹处理耗时慢的问题，同样解决了部分batch数据轨迹太少）
                     需要注意的是 在meta原本思路中，batch的定义与此处不同，batch以task为基础，每个task只需要一条support和一条query，其皆由单个20s场景组成
                     但很可能存在，即相应的20s内无轨迹，或则只有几条行人轨迹可用；故而可以先进行叠加以64-128个行人轨迹数先组成batch，以此batch为support和query                     
                     （问题二 源代码对应的batch的traj中部分行人数据未全程存在，该部分如何预测）   
                    """
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)
        # todo 需要分析针对于不足batch_pednum的最后几个windows的情况，如果是train，则直接舍弃最后的数据，如果是test，而且相应的不是最后一帧，
        #  即没有处理完，而且batch-pednum中行人数大于1，则对其进行相应的batch处理，加到数据集中
        #  if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
        #  需要注意的是 train中的数据也不能直接舍弃，当batch较大的时候，相应的由很多数据形不成512，则会被抛弃，造成数据量的不足 ！！
        if last_frame < data_index.shape[1] - 1 and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            if batch_data[4] != []:  # 即batch_pednum为空 则不添加该数据
                batch_data_mass.append((batch_data, Batch_id,))
            elif batch_data[4] == []:
                print('舍弃该数据')
            # batch_data_mass.append((batch_data, Batch_id,))
        return batch_data_mass


    def get_seq_from_index_balance_HIN(self, frameped_dict, pedtraject_dict, pedlabel_dict,data_index,scene_list, setname):
        # 无用
        pass

    def massup_batch_HIN(self, batch_data,type_data):
        # 无用
        pass

    def get_social_inputs_numpy_HIN(self,  inputnodes, cur_type,relation_num):
        # 无用
        pass