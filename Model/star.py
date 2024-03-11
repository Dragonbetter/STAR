import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.multi_attention_forward import multi_head_attention_forward
from Model.CVAE_utils import Normal,MLP2,MLP
from torch.distributions.normal import Normal as Normal_official
torch.manual_seed(0)
def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiheadAttention(nn.Module):
    r"""Allows the ModelStrategy to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the ModelStrategy.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']
    # 真实情况下 self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # 此处决定的是否用单独的，如果qkv的嵌入维度相同，则使用单独的
        # 假设原始的Q、K、V都是32维的，in_proj_weight实际上包含了三组转换矩阵，每组的大小都是32，分别用于Q、K、V的线性变换。
        # 在通过这个合并的权重矩阵进行变换后，我们通常会将结果切分成三个部分，每个部分都是32维的，分别对应于转换后的Q、K、V
        # 有效地利用了一个大矩阵同时对Q、K、V进行线性变换，而通过后续的操作（如切分）
        # todo 直接修改此处的参数 ！！将默认的True改成False =》保证参数不变的话 应该只需要在整个的梯度字典中寻找该对应的参数，而后将其变为None即可
        self._qkv_same_embed_dim = False
        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        # 在实践中，每个头的维度通常是原始嵌入维度除以头数。例如，如果模型的嵌入维度是512，使用8个头，那么每个头的维度将会是64
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # 投影权重: 根据_qkv_same_embed_dim标志，这些权重要么作为一个合并的权重矩阵in_proj_weight初始化，要么作为分开的q_proj_weight、k_proj_weight和v_proj_weight初始化。
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            # 原文用的是这个
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        # true
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # false
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        # false
        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        # 区别在于： use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
        # v_proj_weight=self.v_proj_weight
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward ModelStrategy
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Multi Multihead Attention
        # todo 需要在此处添加对应的qk输出权重的结果
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        # Add and norm
        src = src + self.dropout1(src2)
        '''
        nn.LayerNorm () layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        此处normalized_shape：表示输入张量的形状，可以是一个整数或一个元组。如果输入张量的形状是 (batch_size, num_features)，
        则 normalized_shape 应为 num_features,此处为32，（seq，batch-size，num-features），seq和batch-size都是可以变化的，
        此处的归一化应该值得是对于batch中的每一个样本，seq中的每一个时刻，对num-feature做归一化；而且此处针对的是时序，不考虑空间关系
        此处batch-size只为充分利用GPU并行计算各个ped行人序列
        batch-norm此处也不应该针对seq，应该是seq的每个时刻，每个feature，以batch-size做归一化；
        那此处这就相当于基于空间位置（2-32）归一化？但前述已经有过了，或许特征变换后不该怎么理解。
        temporal:(seq,batch-size,num-feature)  spatial:(batch-size,1,num-feature)
        '''
        src = self.norm1(src)
        # LayerNorm -- BatchNorm 改进
        # Feed Forward
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # Add and norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
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
        # Encoder有1层transformer layer
        for i in range(self.num_layers):
            output, attn = self.layers[i](output, src_mask=mask,
                                          src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output


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
        output = self.transformer_encoder(src, mask=n_mask)

        return output


class STAR(torch.nn.Module):

    def __init__(self, args, dropout_prob=0):
        super(STAR, self).__init__()

        # set parameters for network architecture
        self.embedding_size = [32]
        self.output_size = 2
        self.dropout_prob = dropout_prob
        self.args = args
        self.mean = []
        self.var = []

        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward network ModelStrategy in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value

        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        """
        舍弃无效的self.temporal_encoder_layer 后续直接添加 避免梯度为NONE
        self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)
        self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
        self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer, 1)
        """
        self.temporal_encoder_1 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        self.temporal_encoder_2 = TransformerEncoder(TransformerEncoderLayer(d_model=32, nhead=8), 1)
        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)

        # Linear layer to output and fusion
        # inplace-operation 问题点 最终显示该变量version变化，即需要version-7，但后续变量已经被修改了 成version——9
        self.output_layer = nn.Linear(48, 2)
        self.fusion_layer = nn.Linear(64, 32)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(self.dropout_prob)
        self.dropout_in2 = nn.Dropout(self.dropout_prob)
        """
        inplace
        ReLU uses its output for the gradient computation as defined here and as shown in this code snippet:
        故而relu在使用时需要注意其 inplace为true或false true节省开销 但容易造成inplace operation
        """

    def get_st_ed(self, batch_num):
        """

        :param batch_num: contains number of pedestrians in different scenes for a batch
        :type batch_num: list
        :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
        :rtype: list
        """
        cumsum = torch.cumsum(batch_num, dim=0) # 沿着第一个维度进行累加求和操作
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

        st_ed.insert(0, (0, int(cumsum[0])))

        return st_ed

    def get_node_index(self, seq_list):
        """

        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):

            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices

    def update_batch_pednum(self, batch_pednum, ped_list):
        """

        :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
        :type list
        :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
        :type tensor
        :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
        :rtype: list
        """
        updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
        updated_batch_pednum = copy.deepcopy(batch_pednum)

        cumsum = np.cumsum(updated_batch_pednum_)
        new_ped = copy.deepcopy(ped_list).cpu().numpy()

        for idx, num in enumerate(cumsum):
            num = int(num)
            if idx == 0:
                updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
            else:
                updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

        return updated_batch_pednum

    def mean_normalize_abs_input(self, node_abs, st_ed):
        """

        :param node_abs: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: node_abs: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        node_abs = node_abs.permute(1, 0, 2)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)
    def forward(self,inputs,stage,iftest=False):

        # ifmeta-test用来表征相应的是否需要进行注入操作，而相应的mean-list以及var——list则表示的是对应的不同域的特征均值和方差
        # nodes_abs/nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

        noise = get_noise((1, 16), 'gaussian')

        for framenum in range(self.args.seq_length - 1):

            if framenum >= self.args.obs_length and iftest:
                # iftest 标志测试阶段，测试数据只有过往的8帧，后续生成的应该全部都是 基于这8帧预测得到的
                # 提取从起始到观测帧都存在的行人，以便后续预测
                node_index = self.get_node_index(seq_list[:self.args.obs_length])
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                st_ed = self.get_st_ed(updated_batch_pednum)
                # 回传outputs的上一帧的结果用于新一帧的预测；
                nodes_current = outputs[self.args.obs_length - 1:framenum, node_index]
                nodes_current = torch.cat((nodes_norm[:self.args.obs_length, node_index], nodes_current))
                node_abs_base = nodes_abs[:self.args.obs_length, node_index]
                # 将经过归一化的outputs数据返回至未归一化效果
                node_abs_pred = shift_value[self.args.obs_length:framenum + 1, node_index] + outputs[
                                                                                             self.args.obs_length - 1:framenum,
                                                                                             node_index]
                node_abs = torch.cat((node_abs_base, node_abs_pred), dim=0)
                # We normalize the absolute coordinates using the mean value in the same scene
                node_abs = self.mean_normalize_abs_input(node_abs, st_ed)

            else:
                # node-idx筛选出从起始到当前帧都存在的ped
                node_index = self.get_node_index(seq_list[:framenum + 1])
                nei_list = nei_lists[framenum, node_index, :]
                nei_list = nei_list[:, node_index]
                # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；仍然会随着frame的变换而变换
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
                st_ed = self.get_st_ed(updated_batch_pednum)
                # 只取到framenum的数据
                nodes_current = nodes_norm[:framenum + 1, node_index]
                # We normalize the absolute coordinates using the mean value in the same scene
                # 基于同一场景中的行人数据进行标准化，即运用该windows所有行人的平均xy坐标进行分析
                node_abs = self.mean_normalize_abs_input(nodes_abs[:framenum + 1, node_index], st_ed)

            # Input Embedding
            # 此处作用将输入的xy 2维坐标变量转换为32维向量，相应的先linear，后relu，而后dropout；此处无inplace=True问题，也无+=问题；
            if framenum == 0:
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
            else:
                # 当framenum=18时，相应的nodes——current为【19,140,2】
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
                # GM对应论文中的Graph Memory
                # todo 错误的是 【19,140,32】 即最后一轮的temporal——input-embedded，此处传完后version由0变成1，
                #  故而此处改,由索引传递变量也是一个原位操作，会引起_version的变化,故而相应的使用.clone()完成修复
                #  .clone()函数返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
                #  故而相应的inplace-operation是在复制后的张量上进行，不会改变其值，但其后续产生的梯度仍然可以正常回传，故而避免了backward时找不到对应当时值可行计算梯度
                temporal_input_embedded = temporal_input_embedded.clone()
                temporal_input_embedded[:framenum] = GM[:framenum, node_index]
            # 需要注意 temporal（nodes-current：经过基于obs观测帧的归一化）和spatial（node-abs基于全局的norm）输入的node序列数据经过不同的处理，spatial_input_embedded_(seq,num_ped,hidden_vim)
            spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs)))
            # 数据流式处理，空间输入基于最新的一帧进行分析，过往的空间关系不考虑
            # spatial_input_embedded_[-1].unsqueeze(1) (num_ped,1,hi_vim) nei_list(num_ped,num_ped) spatial_input_embedded(num_ped,1,hi_vim)
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)
            #  spatial_input_embedded （num_ped,hid_vim）
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
            # 时间输入基于完整的截止当前帧的数据（但是其中只有最新帧的数据是输入的，最新帧往前的数据是基于过往的预测结果的，而不是原始的结果）输出会重构所有帧数据，但只取最后一帧
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)[-1]
            # 取temporal-input-embedded初始到倒数第二个数据，此处倒数第一个数据经过encoder后被重构！！
            temporal_input_embedded = temporal_input_embedded[:-1]

            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)
            # 经过第一次encoder后的最新帧数据都变了，对其拼接，输入spatial-encoder
            spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)
            # 将经过spatial_encoder_2的数据与原先的temporal_input_embedded（初始到倒数第二个）进行拼接：正好又拼接成完整的序列
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)[-1]  # 此处后从seq.228.32变为228.32
            #  在此处添加对应的BN或则对应的attention机制 temporal-input-embedded为(batch，feature）=》均值/方差(1，feature）
            # 根据beta分布生成权重 将来当执行MixUp时,就可以通过线性组合sample1和输入features,实现特征空间中的混合mixup
            temporal_input_embedded = temporal_input_embedded
            # 添加噪声
            noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)
            # print("self.output_layer.weight_before", str(self.output_layer.weight._version))
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            # outputs_current = nn.functional.linear(temporal_input_embedded_wnoise,self.output_layer.weight.clone(),self.output_layer.bias)
            # 回传 outputs：相应的将预测结果传回，在预测新的一帧时运用，GM则回传中间特征层的数据，每次都只回传最新帧
            # print("self.output_layer.weight_after",str(self.output_layer.weight._version))
            outputs[framenum, node_index] = outputs_current
            # todo是否是因为此处还未backwardtemporal_input_embedded，而GM的值就已经改变，而使得对应的值也被变了
            GM[framenum, node_index] = temporal_input_embedded
        # self.mean 00-18 共19个
        return outputs

    def forward_mixup(self, inputs, stage, mean_list, var_list, iftest=False,ifmixup = False):
        # ifmeta-test用来表征相应的是否需要进行注入操作，而相应的mean-list以及var——list则表示的是对应的不同域的特征均值和方差
        # nodes_abs/nodes_norm(归一化后)(19,259,2)  shift_value(19 259 2) seq_list (19,259) nei_lists(19 259 259) nei_num(19,259) batch_pednum(50)
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

        noise = get_noise((1, 16), 'gaussian')

        for framenum in range(self.args.seq_length - 1):

            if framenum >= self.args.obs_length and iftest:
                # iftest 标志测试阶段，测试数据只有过往的8帧，后续生成的应该全部都是 基于这8帧预测得到的
                # 提取从起始到观测帧都存在的行人，以便后续预测
                node_index = self.get_node_index(seq_list[:self.args.obs_length])
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                st_ed = self.get_st_ed(updated_batch_pednum)
                # 回传outputs的上一帧的结果用于新一帧的预测；
                nodes_current = outputs[self.args.obs_length - 1:framenum, node_index]
                nodes_current = torch.cat((nodes_norm[:self.args.obs_length, node_index], nodes_current))
                node_abs_base = nodes_abs[:self.args.obs_length, node_index]
                # 将经过归一化的outputs数据返回至未归一化效果
                node_abs_pred = shift_value[self.args.obs_length:framenum + 1, node_index] + outputs[
                                                                                             self.args.obs_length - 1:framenum,
                                                                                             node_index]
                node_abs = torch.cat((node_abs_base, node_abs_pred), dim=0)
                # We normalize the absolute coordinates using the mean value in the same scene
                node_abs = self.mean_normalize_abs_input(node_abs, st_ed)

            else:
                # node-idx筛选出从起始到当前帧都存在的ped
                node_index = self.get_node_index(seq_list[:framenum + 1])
                nei_list = nei_lists[framenum, node_index, :]
                nei_list = nei_list[:, node_index]
                # 更新batch-pednum：去除消失的行人后batch中每个windows下的新的人数；仍然会随着frame的变换而变换
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                # 依据updated-batch-pednum得出相应的每个windows中开始和结束的行人序列号，便于分开处理
                st_ed = self.get_st_ed(updated_batch_pednum)
                # 只取到framenum的数据
                nodes_current = nodes_norm[:framenum + 1, node_index]
                # We normalize the absolute coordinates using the mean value in the same scene
                # 基于同一场景中的行人数据进行标准化，即运用该windows所有行人的平均xy坐标进行分析
                node_abs = self.mean_normalize_abs_input(nodes_abs[:framenum + 1, node_index], st_ed)

            # Input Embedding
            # 此处作用将输入的xy 2维坐标变量转换为32维向量，相应的先linear，后relu，而后dropout；此处无inplace=True问题，也无+=问题；
            if framenum == 0:
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
            else:
                # 当framenum=18时，相应的nodes——current为【19,140,2】
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
                # GM对应论文中的Graph Memory
                # todo 错误的是 【19,140,32】 即最后一轮的temporal——input-embedded，此处传完后version由0变成1，
                #  故而此处改,由索引传递变量也是一个原位操作，会引起_version的变化,故而相应的使用.clone()完成修复
                #  .clone()函数返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
                #  故而相应的inplace-operation是在复制后的张量上进行，不会改变其值，但其后续产生的梯度仍然可以正常回传，故而避免了backward时找不到对应当时值可行计算梯度
                temporal_input_embedded = temporal_input_embedded.clone()
                temporal_input_embedded[:framenum] = GM[:framenum, node_index]
            # 需要注意 temporal（nodes-current：经过基于obs观测帧的归一化）和spatial（node-abs基于全局的norm）输入的node序列数据经过不同的处理，spatial_input_embedded_(seq,num_ped,hidden_vim)
            spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs)))
            # 数据流式处理，空间输入基于最新的一帧进行分析，过往的空间关系不考虑
            # spatial_input_embedded_[-1].unsqueeze(1) (num_ped,1,hi_vim) nei_list(num_ped,num_ped) spatial_input_embedded(num_ped,1,hi_vim)
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)
            #  spatial_input_embedded （num_ped,hid_vim）
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
            # 时间输入基于完整的截止当前帧的数据（但是其中只有最新帧的数据是输入的，最新帧往前的数据是基于过往的预测结果的，而不是原始的结果）输出会重构所有帧数据，但只取最后一帧
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)[-1]
            # 取temporal-input-embedded初始到倒数第二个数据，此处倒数第一个数据经过encoder后被重构！！
            temporal_input_embedded = temporal_input_embedded[:-1]

            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)
            # 经过第一次encoder后的最新帧数据都变了，对其拼接，输入spatial-encoder
            spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)
            # 将经过spatial_encoder_2的数据与原先的temporal_input_embedded（初始到倒数第二个）进行拼接：正好又拼接成完整的序列
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)[-1]  # 此处后从seq.228.32变为228.32
            #  在此处添加对应的BN或则对应的attention机制 temporal-input-embedded为(batch，feature）=》均值/方差(1，feature）
            # 根据beta分布生成权重 将来当执行MixUp时,就可以通过线性组合sample1和输入features,实现特征空间中的混合mixup
            if ifmixup:
                if stage == 'query' and iftest == False:
                    # meta-test阶段 均匀分布
                    lam = np.random.beta(1., 1.)
                    from torch.distributions.normal import Normal
                    # todo 先单域注入后续再改成多域注入
                    # mean_list[framenum] 为32 Distri1为正太分布对象 sample1抽样对象为 （228.32） 而后混合特征
                    Distri1 = Normal(mean_list[framenum], var_list[framenum])
                    sample1 = Distri1.sample([temporal_input_embedded.size(0), ])
                    # lam产生的值会不会太大 或则说特征的混合需要换一下 0.96较大
                    # todo 问题 特征混合应该不在这个地方 或则说特征混合的时间跨度应该只在 前8s ？ 此处特征已经完全
                    temporal_input_embedded = lam * temporal_input_embedded + (1 - lam) * sample1
                elif stage == 'support' and iftest == False:
                    # meta-train阶段
                    mean = temporal_input_embedded.mean(dim=0)
                    var = temporal_input_embedded.var(dim=0)
                    self.mean.append(mean)
                    self.var.append(var)
                    temporal_input_embedded = temporal_input_embedded
                elif iftest == True:
                    temporal_input_embedded = temporal_input_embedded
            else:
                temporal_input_embedded = temporal_input_embedded
            # 添加噪声
            noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)
            # print("self.output_layer.weight_before", str(self.output_layer.weight._version))
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            # outputs_current = nn.functional.linear(temporal_input_embedded_wnoise,self.output_layer.weight.clone(),self.output_layer.bias)
            # 回传 outputs：相应的将预测结果传回，在预测新的一帧时运用，GM则回传中间特征层的数据，每次都只回传最新帧
            # print("self.output_layer.weight_after",str(self.output_layer.weight._version))
            outputs[framenum, node_index] = outputs_current
            # todo是否是因为此处还未backwardtemporal_input_embedded，而GM的值就已经改变，而使得对应的值也被变了
            GM[framenum, node_index] = temporal_input_embedded
        # self.mean 00-18 共19个
        return outputs, self.mean, self.var


# 相应的添加其对应的训代码 为了扩展的考虑

"""
舍弃无用参数赋值类
class ModifiableModule(nn.Module):
    def params(self):
        return [p for _,p in self.named_params()]

    def named_leaves(self):
        #获取叶子节点的参数
        return [self.get_inner_loop_parameter_dict(self.net.named_parameters())]

    def named_submodules(self):
        #  获取子模块  此处子模块的获取应该是一层一层的  name_children ：只获取一层，name_modules:获取所有的子模块
        # 但相比self.net。state——dict（）似乎少了一些 ？'spatial_encoder_1.transformer_encoder.layers.0.self_attn.in_proj_weight'
        # return [(name, module) for name, module in self.named_modules() if len(list(module.modules())) == 1]
        return self.named_children()

    def named_params(self):
        subparams = []
        for name,mod in self.named_children():
            for subname, param in mod.named_params():
                subparams.append((name+'.'+subname,param))
        return self.named_leaves()+subparams

    def set_param(self,curr_mod,name,param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name,mod in curr_mod.named_children():
                if module_name == name:
                    mod.set_param(mod,rest,param)
                    break
        else:
            setattr(curr_mod,name,param)

    def copy(self,other,same_var=False):
        for name, param in other.named_parameters():
            if not same_var:
                param = V(param.data.clone(),requires_grad=True)
            self.set_param(other,name,param)
"""
