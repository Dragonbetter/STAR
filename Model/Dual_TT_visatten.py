# 本着不改变原来代码结构的情况 重新设计适配于具有atten可视化的代码,相应的权重可以同样的加载；
# 主要修改transformerEncoder 以及TransformerModel 使其能输出atten
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.CVAE_utils import Normal, MLP2, MLP
from Model.star import _get_clones, TransformerEncoderLayer
from Model.star_cvae import Decoder, STAR_CVAE
from Model.DualTT import PositionalAgentEncoding
from vis.vis_for_attention import draw_attention_map


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, need_aligin_loss=False):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        atts = []
        # 时间的Encoder有1层transformer layer
        for i in range(self.num_layers):
            output, attn, qkv_dict = self.layers[i](output, src_mask=mask,
                                                    src_key_padding_mask=src_key_padding_mask,
                                                    need_aligin_loss=need_aligin_loss)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)        
        
        return output, attn, qkv_dict
    
class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):
        # 它通过在原始掩码 mask 上添加一个单位矩阵（identity matrix）来创建。单位矩阵的对角线上的元素为1，其余为0
        # 这个步骤实际上并不阻止序列中的每个位置关注到自身，反而可能会增强这种关注（因为对角线上的值变成了 1 或更大，取决于原始掩码的值）。
        n_mask = mask + torch.eye(mask.shape[0], mask.shape[0]).cuda()
        # 将掩码中的 0 替换为一个非常小的数（接近负无穷）。这在自注意力的 softmax 步骤中有效地屏蔽了这些位置，因为经过 softmax 后这些位置的权重会接近零。
        # 将掩码中的 1 替换为 0。这一步看起来有些多余，因为在 softmax 中，0 会被转换为一个有效的权重值。
        n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        output,attn,qkv_dict = self.transformer_encoder(src, mask=n_mask)

        return output,attn,qkv_dict
    
class Dual_TT_visatten_Encoder(nn.Module):
    # 区别主要体现在 1.时间encoder处，2，输出的结果会包含qkv，需要设计并取出
    def __init__(self, args, stage, dropout_prob=0):
        super(Dual_TT_visatten_Encoder, self).__init__()
        self.embedding_size = 32
        self.dropout_prob = dropout_prob
        self.args = args
        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward network ModelStrategy in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value
        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)
        # Linear layer to map different path TI/IT
        self.input_TI_layer = nn.Linear(32, 32)
        self.input_IT_layer = nn.Linear(32, 32)
        # Linear layer to output and fusion
        self.fusion_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)
        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in_1 = nn.Dropout(self.dropout_prob)
        self.dropout_in_2 = nn.Dropout(self.dropout_prob)
        # 为MLP服务
        self.dropout_in_TI = nn.Dropout(self.dropout_prob)
        self.dropout_in_IT = nn.Dropout(self.dropout_prob)
        # 后续的消融也是基于下文的两个空间两个时间进行分析
        # 空间模块基于transformerModel 其中的mask为对应的nei-lists即邻接矩阵
        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        # todo !! 时间模块 此处的输入数据其实只保留了整个时间序列都存在的序列 基于1层的 新的可输出qkv值的
        self.temporal_encoder_1 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        self.temporal_encoder_2 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        # PE函数 对应于两个时间transformer
        self.pos_encoder1 = PositionalAgentEncoding(emsize, 0.1, concat=False)
        self.pos_encoder2 = PositionalAgentEncoding(emsize, 0.1, concat=False)
        # 涉及到如何将过去8帧的数据或则说是未来的12帧的数据进行合并？？
        # 直接选取最后一帧是常见做法，或则说再过一个linear或max-pooling
        # 注意相对应的ST-HIN采用的是双路的结构 并且用的都是经过transformer后的最后一步的值 故而此处用不到full-layer 需要注释掉 以防其在MLDG的过程中发生错误

    def forward(self, inputs):
        # nodes_current(未有空间或时间归一化)/abs(基于每个场景进行过空间归一化) (length,num_ped,2),nei_list(length, num_ped, num_ped)
        nodes_current, nodes_abs_position, nei_list = inputs
        length = nodes_current.shape[0]  # pred-length;obs-length;
        num_ped = nodes_current.shape[1]
        # todo 先完成一版未基于时间归一化的数据 模型处理
        # TI branch 先时间后空间
        # temporal embedding
        nodes_current_embedded = self.input_embedding_layer_temporal(nodes_current)
        if self.args.PE == 'True':
            nodes_current_pos = self.pos_encoder1(nodes_current_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_current_pos = nodes_current_embedded
        temporal_input_embedded_origin = self.dropout_in_1(self.relu(nodes_current_pos))
        # TM branch 添加对应的att,qkv
        temporal_input_embedded_temporal_TS,att_TS_temporal,qkv_TS = self.temporal_encoder_1(temporal_input_embedded_origin,need_aligin_loss=self.args.need_aligin_loss)
        # X+MLP(TM(X)) todo => 分析此处的X应该是具体何值？高维映射 不然2和32维度无法对应的添加
        temporal_input_embedded_temporal_TS = temporal_input_embedded_origin + self.dropout_in_TI(
            self.relu(self.input_TI_layer(temporal_input_embedded_temporal_TS)))
        # 此处直接用cuda 不知是否需要单独指定device
        # SM branch
        spatial_input_embedded_spatial_TS = torch.zeros(length, num_ped, self.embedding_size).cuda()
        # 因为空间的atten是逐帧的
        att_TS_spatial = torch.zeros(length,num_ped,num_ped).cuda()
        for frame in range(length):
            # [length,num-ped,32]->[num-ped,32]->[num_ped,1,32]
            spatial_input_embedded_spatial_frame_TS, att_TS_spatial_mid, _ = self.spatial_encoder_1(
                temporal_input_embedded_temporal_TS[frame].unsqueeze(1), nei_list[frame])
            # [num_ped,1,32]->[1,num_ped,32]->[num_ped,32]
            spatial_input_embedded_spatial_frame_TS = spatial_input_embedded_spatial_frame_TS.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_TS[frame] = spatial_input_embedded_spatial_frame_TS
            att_TS_spatial[frame] = att_TS_spatial_mid
        TS_embedding = spatial_input_embedded_spatial_TS

        # IT分子 先空间后时间
        # Spatial embedding
        nodes_abs_position_embedded = self.input_embedding_layer_spatial(nodes_abs_position)
        if self.args.PE == 'True':
            # nodes_abs_position_pos = nodes_abs_position_embedded # 为了验证对应的空间Encoder的有效性
            nodes_abs_position_pos = self.pos_encoder2(nodes_abs_position_embedded, num_a=num_ped)
        elif self.args.PE == 'False':
            nodes_abs_position_pos = nodes_abs_position_embedded
        spatial_input_embedded_origin = self.dropout_in_2(self.relu(nodes_abs_position_pos))
        # SM branch
        spatial_input_embedded_spatial_ST = torch.zeros(length, num_ped, self.embedding_size).cuda()
        att_ST_spatial = torch.zeros(length,num_ped,num_ped).cuda()
        for frame in range(length):
            spatial_input_embedded_spatial_frame_ST, att_ST_spatial_mid, _= self.spatial_encoder_2(
                spatial_input_embedded_origin[frame].unsqueeze(1), nei_list[frame])
            spatial_input_embedded_spatial_frame_ST = spatial_input_embedded_spatial_frame_ST.permute(1, 0, 2)[-1]
            spatial_input_embedded_spatial_ST[frame] = spatial_input_embedded_spatial_frame_ST
            att_ST_spatial[frame] = att_ST_spatial_mid
        # X+MLP(SM(X)) todo
        spatial_input_embedded_spatial_ST = spatial_input_embedded_origin + self.dropout_in_IT(
            self.relu(self.input_IT_layer(spatial_input_embedded_spatial_ST)))
        # TM branch  添加对应的att,qkv
        temporal_input_embedded_temporal_ST ,att_ST_temporal, qkv_ST= self.temporal_encoder_2(spatial_input_embedded_spatial_ST,need_aligin_loss=self.args.need_aligin_loss)
        ST_embedding = temporal_input_embedded_temporal_ST
        # 拼接双路结构 todo => MaxPooling?
        # 计算对齐Loss =》 依据余弦相似度
        # 计算A和B之间的余弦相似度
        cosine_sim = F.cosine_similarity(TS_embedding[-1], ST_embedding[-1], dim=1)
        # 定义损失函数为1减去余弦相似度的平均值
        loss = 1 - cosine_sim.mean()
        # 直接拼接最后一维 即【num_ped,dim】
        fusion_feat_origin = torch.cat((TS_embedding[-1], ST_embedding[-1]), dim=1)
        # 拼接双路的frame结构 并fusion !
        fusion_feat = self.fusion_layer(fusion_feat_origin)
        # qkv_dict [TS,ST]
        qkv_dict = {'TS': qkv_TS, 'ST': qkv_ST}
        att_dict = {'TS_spatial': att_TS_spatial,'ST_spatial': att_ST_spatial,'TS_temporal': att_TS_temporal,'ST_temporal': att_ST_temporal}
        return fusion_feat,loss,qkv_dict,att_dict




class Dual_TT_visatten(STAR_CVAE):
    def __init__(self, args):
        super(Dual_TT_visatten, self).__init__(args)
        self.args = args
        self.embedding_size = 32
        # model structure
        self.past_encoder = Dual_TT_visatten_Encoder(args, stage='past')
        self.future_encoder = Dual_TT_visatten_Encoder(args, stage='future')
        self.out_mlp = MLP2(input_dim=64, hidden_dims=[32], activation='relu')
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        self.pz_layer = nn.Linear(self.embedding_size, 2 * self.args.zdim)
        self.decoder = Decoder(args)
    
    def save_atten(self, epoch, updated_batch_pednum,nei_list,nodes_current,past_att_dict,diverse_pred_traj):
        # 保存模型的代码与maml的代码框架合计 origin原始 MLDG 元学习
        model_save_path = os.path.join(self.args.model_dir,f"{self.args.train_model}{str(stage)}_{str(epoch)}.tar")
        torch.save({
            'updated_batch_pednum': updated_batch_pednum,
            'nei_list': nei_list,
            'nodes_current': nodes_current,
            'past_att_dict': past_att_dict,
            'diverse_pred_traj':diverse_pred_traj,
        }, model_save_path)

    def forward(self, inputs, stage,batch_id):
        # 此处的stage是对应的support的
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259)
        # nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 预处理数据只考虑从头到尾的行人，这个只是相当于利用了更多行人的信息；但在此处，我们采用传统思路预处理，因为我们是针对完全形态分析
        # node-idx筛选出从起始到当前帧都存在的ped
        node_index = self.get_node_index(seq_list)
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]
        # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
        if batch_pednum.cpu().detach().numpy().shape[0] - updated_batch_pednum.cpu().detach().numpy().shape[0] > 0:
            print(
                'batch_pednum:' + str(batch_pednum.cpu().detach().numpy().shape) + '/' + 'updated_batch_pednum:' + str(
                    batch_pednum.cpu().detach().numpy().shape))
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_current, st_ed)
        # 输入encoder的前置知识
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:obs_length]
        future_traj = nodes_current[obs_length:], nodes_abs_position[obs_length:], nei_list[obs_length:]
        # (num_ped,hidden_feature32)
        # todo 添加对应的loss输出 ！！基于video表示的余弦相似度
        past_feature,past_loss,past_qkv_dict,past_att_dict = self.past_encoder(past_traj)
        future_feature,future_loss,future_qkv_dict,future_att_dict = self.future_encoder(future_traj)
        loss_TT = past_loss + future_loss
        # todo 输出对应的qkv值
        qkv_dict = {'past':past_qkv_dict,'future':future_qkv_dict}
        # ===>>>CVAE
        # qz_distribution
        # (batch_size,hidden_dim*2) 64
        h = torch.cat((past_feature, future_feature), dim=1)
        # (batch_size,32) 64->32
        h = self.out_mlp(h)
        # 在变分自编码器中，潜变量z的均值和方差是通过编码器网络输出的，并且需要满足一定的分布假设，例如高斯分布。
        # 因此，该线性层的作用是将 MLP 的输出映射到满足分布假设的潜变量均值和方差，从而使得潜变量 z 可以被正确地解码和重构。
        qz_param = self.qz_layer(h)
        qz_distribution = Normal(params=qz_param)
        qz_sampled = qz_distribution.rsample()
        # pz_distribution =》可学习的先验分析！！
        # 使用线性层对象self.pz_layer生成一个包含均值和标准差参数的向量pz_param，并以此创建一个Normal分布pz_distribution。
        pz_param = self.pz_layer(past_feature)
        pz_distribution = Normal(params=pz_param)
        # 真值
        node_past = nodes_current[:obs_length].transpose(0, 1)  # (num-ped,8,2)
        node_future = nodes_current[obs_length:].transpose(0, 1)  # (num-ped,12,2)
        # decoder
        pred_traj, recover_traj = self.decoder(past_feature, qz_sampled, node_past, sample_num=1)
        assert pred_traj.shape[0] == node_future.shape[0] == recover_traj.shape[0] == node_past.shape[0]
        batch_size = pred_traj.shape[0]
        # loss-recover,loss-pred,loss-kl
        loss_recover = self.calculate_loss_recover(recover_traj, node_past, batch_size)
        loss_pred = self.calculate_loss_pred(pred_traj, node_future, batch_size)
        batch_pednum = past_feature.shape[0]
        # todo KL-Loss分析！！min-clap 2->0
        loss_kl = self.calculate_loss_kl(qz_distribution, pz_distribution, batch_pednum, min_clip=0)
        # for loss-diversity ==》p dist for best 20 loss
        """
        主要目的在于计算loss-divers -- 由Social-GAN提出来的
        根据均值和标准差参数p_z_params（如果self.args.learn_prior为True，则使用线性层对象self.pz_layer生成），创建先验分布p(z)，用于计算ELBO损失。
        区别在于，这里对每个样本采样sample_num次，以便在计算ELBO损失时可以更精确地估计期望值。
        将past_feature张量重复sample_num次，以便将batch_size和agent_num两个维度扩展为(batch_size * agent_num * sample_num)。
        """
        # 默认20 =》 loss_diversity:基于过去的轨迹 不是结合未来和过去
        sample_num = self.args.sample_num
        # (batch_size * agent_num * sample_num,embeddings)。
        past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
        pz_params_repeat = self.pz_layer(past_feature_repeat)
        pz_distribution = Normal(params=pz_params_repeat)
        pz_sampled = pz_distribution.rsample()
        # loss-diverse
        diverse_pred_traj, _ = self.decoder(past_feature_repeat, pz_sampled, node_past, sample_num=self.args.sample_num,
                                            mode='inference')
        loss_diverse = self.calculate_loss_diverse(diverse_pred_traj, node_future)
        # total-loss!!
        total_loss = loss_pred + loss_recover + loss_kl + loss_diverse + loss_TT
        # 不同的参数组合设计 todo
        # total_loss = 1.5*loss_pred + loss_recover + loss_kl + 2.0*loss_diverse
        # 这个可以等到后期相应的完全训练完后 再基于该数据去跑一遍 但是不更新数据！！即用好的参数在模型在跑一遍数据 
        draw_attention_map(updated_batch_pednum,nei_list,nodes_current,past_att_dict,diverse_pred_traj,batch_id)
        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item(),loss_TT.item(),qkv_dict

    def inference(self, inputs):
        # 这是一个模型推理方法的实现。该方法接收包含过去轨迹的数据，并根据学习到的先验和给定的过去轨迹输出多样化的预测轨迹。
        # 该方法使用过去编码器计算过去特征，使用pz_layer从先验分布中采样，并使用解码器生成多样化的预测轨迹。最终输出的结果是多样性预测轨迹。
        # 注意此处inputs前期输入的是19s，此处更改为正确的20s，因为方法不一样了
        # nodes_abs 为原始的轨迹 后续也用这个 seq
        obs_length = self.args.obs_length
        pred_length = self.args.pred_length
        # nodes_abs未归一化 /nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        # 更新考虑从头到尾存在的行人=》数据预处理
        node_index = self.get_node_index(seq_list)
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]
        updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
        st_ed = self.get_st_ed(updated_batch_pednum)
        nodes_current = nodes_abs[:, node_index]
        nodes_abs_position = self.mean_normalize_abs_input(nodes_current, st_ed)
        past_traj = nodes_current[:obs_length], nodes_abs_position[:obs_length], nei_list[:obs_length]
        # encoder
        past_feature,past_qkv_dict,_ = self.past_encoder(past_traj)
        target_pred_traj = nodes_current[obs_length:]
        sample_num = self.args.sample_num
        # cvae
        past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
        pz_params_repeat = self.pz_layer(past_feature_repeat)
        pz_distribution = Normal(params=pz_params_repeat)
        pz_sampled = pz_distribution.rsample()
        node_past = nodes_current[:obs_length].transpose(0, 1)
        diverse_pred_traj, _ = self.decoder(past_feature_repeat, pz_sampled, node_past, sample_num=self.args.sample_num,
                                            mode='inference')
        #  (agent_num, sample_num, self.pred_length, 2) -> (sample_num,agent_num,self.pred_length, 2)
        diverse_pred_traj = diverse_pred_traj.permute(1, 0, 2, 3)
        if self.args.phase == 'test' and self.args.vis == 'sne':
            return diverse_pred_traj, target_pred_traj, past_feature
        else:
            return diverse_pred_traj, target_pred_traj


    



