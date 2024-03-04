import re
import copy
import logging
from tqdm import tqdm
from torch import nn

from Model.star import STAR
from Model.star_cvae import STAR_CVAE
from Model.star_cvae_hin import STAR_CVAE_HIN
from Model.DualTT import Dual_TT

from src.Visual import draw_result_NBA,VISUAL_TSNE,draw_result_SDD,draw_result_SDD_comparison
from src.visual_loss import draw_loss
from src.utils import *
from src.Loss import getLossMask,L2forTestS,L2forTest_RMSE_MAE,L2forTestS_NEWSTAR,L2forTest_RMSE_MAE_NEWSTAR

from DataProcessor.SDD_Dataloader import *
from DataProcessor.ETH_UCY_preprocess import DatasetProcessor_ETH_UCY
from DataProcessor.SDD_preprocess import DatasetProcessor_SDD
from DataProcessor.NBA_preprocess import DatasetProcessor_NBA
from DataProcessor.Soccer_preprocess import DatasetProcessor_Soccer
from DataProcessor.Ship_preprocess import DatasetProcessor_ship
from DataProcessor.Ship_FVessel_preprocess import DatasetProcessor_ship_FVessel


class processor(object):
    def __init__(self, args):
        self.args = args
        self.logger = self.set_logging(log_file=os.path.join(self.args.model_dir, 'recoder.log'))
        # load data
        if self.args.dataset == "ETH_UCY":
            self.dataloader = DatasetProcessor_ETH_UCY(args)
        elif self.args.dataset == "SDD":
            self.dataloader = DatasetProcessor_SDD(args)
        elif self.args.dataset == "NBA":
            self.dataloader = DatasetProcessor_NBA(args)
        elif self.args.dataset == "Soccer":
            self.dataloader = DatasetProcessor_Soccer(args)
        elif self.args.dataset == "Ship":
            self.dataloader = DatasetProcessor_ship(args)
        elif self.args.dataset == "Ship_FVessel":
            self.dataloader = DatasetProcessor_ship_FVessel(args)
        else:
            raise ValueError("Unknown dataset type")

        # load model
        if self.args.train_model == 'star':
            self.net = STAR(args)
        elif self.args.train_model == 'new_star':
            self.net = STAR_CVAE(args)
        elif self.args.train_model == 'new_star_hin':
            self.net = STAR_CVAE_HIN(args)
        elif self.args.train_model == 'Dual_TT':
            self.net = Dual_TT(args)
        # todo 对于学习率优化方面 需要添加对应的内外循环各自的代码

        self.set_optimizer()
        self.task_learning_rate = self.args.task_learning_rate
        # 配置环境
        torch.cuda.set_device(args.device)
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        # 分析模型文件夹是否存在，若不存在，则创立一个
        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        # 将模型的结构写入文本文件 net.txt中，
        # 初始化变量等 'a+' 表示在文件末尾追加新内容，并允许读取文件内容。如果文件不存在，则会创建一个新文件。
        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        # log
        # 设置日志文件映射=>todo 后续如果改了对应的参数，此处需要同步修改
        self.log_files = {
            'origin': 'log_curve.txt',
            'MLDG': 'MLDG_log_curve.txt',
            'MVDG': 'MVDG_log_curve.txt',
            'MVDGMLDG': 'MVDGMLDG_log_curve.txt',
        }
        # metric
        self.best_ade = 100
        self.best_fde = 100
        self.mae = 100
        self.best_epoch = -1

    def to_device(self, inputs, device):
        """将输入数据转移到指定的设备上"""
        if isinstance(inputs, (list, tuple)):
            return [self.to_device(x, device) for x in inputs]
        return inputs.to(device)

    def save_model(self, epoch, stage):
        # 保存模型的代码与maml的代码框架合计 origin原始 MLDG 元学习
        model_save_path = os.path.join(self.args.model_dir,f"{self.args.train_model}{str(stage)}_{str(epoch)}.tar")
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_save_path)

    def load_model(self, stage):
        if self.args.load_model is not None:
            self.args.model_save_path = os.path.join(self.args.model_dir,f"{self.args.train_model}{str(stage)}_{str(self.args.load_model)}.tar")
            self.logger.info(f'{self.args.model_save_path}')
            if os.path.isfile(self.args.model_save_path):
                self.logger.info('Loading checkpoint')
                try:
                    checkpoint = torch.load(self.args.model_save_path)
                    model_epoch = checkpoint['epoch']
                    self.net.eval()
                    # todo eval 模式 以及参数匹配的严格性
                    # self.net.load_state_dict(checkpoint['state_dict'])
                    self.net.load_state_dict(checkpoint['state_dict'], strict=True)
                    self.logger.info(f'Loaded checkpoint at epoch {model_epoch}')
                except Exception as e:
                    self.logger.error(f'Failed to load the model from {self.args.model_save_path}: {e}')
            else:
                self.logger.warning(f'Checkpoint file does not exist: {self.args.model_save_path}')

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def set_logging(self,log_file='recoder.log', log_level=logging.DEBUG, console_log_level=logging.INFO):
        # 配置 logger
        """
        日志信息级别：DEBUG,INFO,WARNING,ERROR,CRITICAL
        日志信息格式：
        """
        logger = logging.getLogger('my_application')
        logger.setLevel(log_level)  # 设置最低的日志级别
        # 创建一个输出到控制台的Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)  # 设置控制台日志级别INFO
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 创建一个写入日志文件的Handler
        file_handler = logging.FileHandler(log_file, mode='a')  # 'a'模式表示追加模式
        file_handler.setLevel(log_level)  # 设置文件日志级别Debug
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                           datefmt='%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def close_and_reopen_log_file(self):
        """关闭并重新打开当前日志文件，用于轮转日志内容"""
        self.current_log_file.close()
        self.current_log_file = open(self.current_log_file_path, 'a+')

    def load_model_with_selective_strictness(self,model, state_dict, expected_missing_keys=None):
        if expected_missing_keys is None:
            expected_missing_keys = {
                'past_encoder.pos_encoder1.pe',
                'future_encoder.pos_encoder1.pe',
                'past_encoder.pos_encoder2.pe',
                'future_encoder.pos_encoder2.pe'
            }
            # 这是一个集合，列出了在加载模型时可以安全忽略的缺失键。这个列表应该根据模型的具体情况进行调整。

        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing_keys = model_keys - loaded_keys
        extra_keys = loaded_keys - model_keys

        # 检查是否所有缺失的键都在预期缺失的键中
        # unexpected_missing_keys：这是实际缺失的键与预期缺失的键之间的差集。
        # 如果这个集合为空，意味着所有缺失的键都是预期内的，可以安全地忽略；否则，会记录错误并抛出异常。
        unexpected_missing_keys = missing_keys - expected_missing_keys
        if len(unexpected_missing_keys) == 0 and len(extra_keys) == 0:
            # 如果没有意外缺失的键或额外的键，可以安全地使用strict=False加载
            model.load_state_dict(state_dict, strict=False)
            # self.logger.info("Model loaded with specific keys ignored.")
        else:
            # 如果有意外缺失的键或额外的键，记录错误并抛出异常
            error_message = f"ERROR - Unexpected missing keys in loaded dict: {unexpected_missing_keys}\n" \
                            f"Extra keys in loaded dict: {extra_keys}"
            self.logger.error(error_message)  # 或者使用 logger.error(error_message) 如果你有设置日志
            raise RuntimeError(error_message)

    def test(self):
        self.logger.info('Testing begin')
        self.load_model(stage=self.args.stage)
        self.net.eval()
        if self.args.train_model == 'star':
            test_error, test_final_error = self.test_epoch()
        elif self.args.train_model in ['new_star', 'new_star_hin','Dual_TT']:
            test_error, test_final_error = self.test_new_epoch()
        self.logger.info(f'Set{self.args.test_set},epoch{self.args.load_model},test_error{test_error},test_final_error{test_final_error}')

    def test_vis(self):
        self.logger.info('Testing begin')
        self.load_model(stage=self.args.stage)
        self.net.eval()
        if self.args.train_model == 'star':
            test_error, test_final_error = self.test_epoch()
        elif self.args.train_model in ['new_star', 'new_star_hin','Dual_TT']:
            if self.args.vis == 'traj' or self.args.vis == 'traj_comparison':
                if self.args.dataset == 'NBA':
                    test_error, test_final_error = self.test_new_visualNBA_epoch()
                elif self.args.dataset == 'SDD':
                    test_error,test_final_error = self.test_new_visualSDD_epoch()
            elif self.args.vis == 'sne':
                test_error, test_final_error = self.test_new_sne_epoch()
        self.logger.info(f'Set{self.args.test_set},epoch{self.args.load_model},test_error{test_error},test_final_error{test_final_error}')

    def train(self):
        self.logger.info('Training begin')
        if self.args.load_model is not None:
            # 需要提取出self。args。load——model中的轮数即epoch
            # 使用正则表达式提取数字
            self.load_model(stage=self.args.stage)
            epoch_start = int(re.search(r'\d+', self.args.load_model).group())
        else:
            epoch_start = 0
        test_error, test_final_error = 0, 0

        file_star_cvae_curve = open(os.path.join(self.args.model_dir, 'star_cvae_log_curve.txt'), 'a+')
        support_query_log_curve = open(os.path.join(self.args.model_dir,'support_query_log_curve.txt'),'a+')
        # 根据当前stage打开相应的日志文件
        self.current_log_file_path = os.path.join(self.args.model_dir, self.log_files[self.args.stage])
        self.current_log_file = open(self.current_log_file_path, 'a+')

        for epoch in range(epoch_start, self.args.num_epochs):
            self.net.train()
            if self.args.stage == 'origin':
                train_loss,loss_pred_epoch,loss_recover_epoch,loss_kl_epoch,loss_diverse_epoch,loss_TT_epoch = self.train_epoch(epoch)
                support_loss, query_loss = 0,0
                file_star_cvae_curve.write(f"{epoch},{train_loss},{loss_pred_epoch},{loss_recover_epoch},{loss_kl_epoch},{loss_diverse_epoch},{loss_TT_epoch}\n")
            elif self.args.stage == 'MLDG':
                # train_loss = self.train_MLDG_epoch(epoch)
                if self.args.meta_way == 'sequential1':
                    train_loss,support_loss,query_loss = self.train_MLDG_mixup_new_epoch_sequential(epoch)
                elif self.args.meta_way == 'parallel2':
                    train_loss, support_loss, query_loss = self.train_MLDG_mixup_new_epoch_parallel(epoch)
                support_query_log_curve.write(f"{epoch},{support_loss},{query_loss}\n")
            elif self.args.stage == 'MVDG':
                train_loss = self.train_MVDG_epoch_new(epoch)
                support_loss, query_loss = 0,0
            elif self.args.stage == 'MVDGMLDG':
                if self.args.meta_way == 'sequential1':
                    train_loss = self.train_MVDGMLDG_epoch_sequential(epoch)
                elif self.args.meta_way == 'parallel2':
                    train_loss = self.train_MVDGMLDG_epoch_parallel(epoch)
                support_loss,query_loss = 0,0

            if epoch >= self.args.start_test:
                # 如果当前轮数大于或等于 args.start_test，则将模型设置为评估模式（self.net.eval()），后续每轮都会跑
                self.net.eval()
                # todo 这里的test-epoch的数据输入与最终的test的时候是一致的，那么其相应的不就是在训练的使用了测试数据吗，此处的数据应该是train分出来的val才对
                if self.args.train_model == 'star':
                    test_error, test_final_error = self.test_epoch()
                elif self.args.train_model in ['new_star', 'new_star_hin','Dual_TT']:
                    test_error, test_final_error = self.test_new_epoch()
                # 调用 test_epoch 函数计算模型在测试集上的 ADE 和 FDE，并将其存储在 test_error 和 test_final_error 变量中。
                # 然后，判断当前的 FDE 是否优于历史最佳 FDE，如果是，则更新 best_ade、best_fde 和 best_epoch 变量的值，并调用 save_model 函数保存模型。
                if test_final_error <= self.best_fde:
                    self.best_ade = test_error
                    self.best_fde = test_final_error
                    self.best_epoch = epoch
                    self.save_model(epoch,stage=self.args.stage)
                # 如果 start_test 大于等于当前轮数，打印当前轮数的训练损失、ADE 和 FDE，否则则只打印训练损失。
                self.logger.info(
                    f'epoch {epoch}, train_loss={train_loss:.9f}, ADE={test_error:.9f}, FDE={test_final_error:.9f}, Best_ADE={self.best_ade:.9f}, Best_FDE={self.best_fde:.9f} at Epoch {self.best_epoch}')
            else:
                self.logger.info(f'epoch {epoch}, train_loss={train_loss:.9f}')
            # 将训练损失、ADE、FDE 和学习率等信息写入日志文件 log_file_curve 中
            # f-string会自动将变量的值转换为字符串，并根据您在大括号{}中放置的表达式来格式化这些值。
            self.current_log_file.write(f"{epoch},{train_loss},{support_loss},{query_loss},{test_error},{test_final_error},{self.args.learning_rate}\n")
            # 并判断当前轮数是否为4 的倍数，如果是，则关闭日志文件并重新打开以防止文件过大
            if epoch % 4 == 0:
                self.close_and_reopen_log_file()
                file_star_cvae_curve.close()
                file_star_cvae_curve = open(os.path.join(self.args.model_dir, 'star_cvae_log_curve.txt'), 'a+')
                support_query_log_curve.close()
                support_query_log_curve = open(os.path.join(self.args.model_dir,'support_query_log_curve.txt'),'a+')
        # 完成所有训练之后的loss绘画分析
        draw_loss(os.path.join(self.args.model_dir, 'star_cvae_log_curve.txt'),self.args.num_epochs)
        draw_loss(os.path.join(self.args.model_dir, 'star_cvae_log_curve.txt'), int(self.args.num_epochs/5))

    def train_epoch(self, epoch):
        # 重置训练数据集的指针，使得数据训练从第一帧开始；
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        loss_pred_list, loss_recover_list, loss_kl_list, loss_diversity_list,loss_TT_list = 0,0,0,0,0
        for batch in range(self.dataloader.train_batchnums):

            start = time.time()
            # todo 获取对应训练batch的数据 相应的有旋转操作以及基于观测点位置坐标的归一化操作
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            # todo 每个batch单独的转移?数据传输很慢？
            # 将数据转成pytorch的tensor格式 转移到GPU上
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = self.to_device(inputs, self.device)
            loss_pred_recover = torch.zeros(1,device=self.device)
            loss = torch.zeros(1,device=self.device)
            """
            分析相应的各个数据项的含义，并寻找是否可以基于此进行task的拆分，而不是从最原始的数据处理开始拆分task
            batch_abs：(seq_length, num_Peds，2) 每帧，每个行人 xy坐标，未归一化
            batch_norm ：(seq_length, num_Peds，2) 数据经过坐标归一化后的结果
            shift_value：(seq_length, num_Peds，2) 观测序列最后一帧数据的复制
            seq_list:(seq_length, num_Peds)（20，257）值为01,1表示该行人在该帧有数据
            {{注意力机制中的掩码详解：允许我们发送不同长度的批次数据一次性的发送到transformer中，在代码中是通过将所有序列填充到相同的长度，
            然后使用“atten-mask”--seq-list张量来识别那些令牌是填充的；在显存允许的情况下，使用批处理输入的速度更快，但不同的输入序列具有不同长度，
            他们无法直接组合成一个张量，此时可以用虚拟标记填充较短的序列，以使每个序列具有相同的长度；
            在注意力掩码中，输入是0和1，但在最终的计算时，会将在无效位置的注意力权重设为一个很小的值，通常为负无穷，以便在计算注意力分数时将其抑制为接近0的概率
            相应的在后续进过softmax函数时，softmax会对输入值进行指数运算，然后进行归一化，当输入值非常小或负无穷时，经过指数运算后会接近0.
            排除无效位置的影响，通过将无效位置的注意力权重设置为负无穷，可以有效的将这些位置的权重压低，}}
            nei_list：(seq_length, num_Peds，num_Peds) （20,257，257） 值为01 以空间距离为基准 分析邻接关系
            nei_num：(seq_length, num_Peds）（20,257）表示每帧下每个行人的邻居数量
            batch_pednum：list 表示该batch下每个时间窗口中的行人数量
            """
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            # 梯度清零
            self.net.zero_grad()
            # outputs (seq_length,batch-pednum,2)(eg:0-18,0-264,2) 和inputs-forward[0]的形式一致
            if self.args.train_model in ['new_star', 'new_star_hin']:
                # 不需要删除最后一秒
                # lossmask 覆盖的是从第1帧开始持续到当前帧都存在的损失，但此处我们的损失应该只是保留从头开始到尾都存在的，故而数据最好进行预处理，在输入进去
                total_loss, loss_pred, loss_recover, loss_kl, loss_diverse,_,_ = self.net.forward(inputs,stage='support',mean_list=[],var_list=[],ifmixup=False)
                loss,loss_TT = total_loss,0

            elif self.args.train_model=='Dual_TT':
                total_loss, loss_pred, loss_recover, loss_kl, loss_diverse,loss_TT = self.net.forward(inputs,stage='support')
                loss = total_loss

            elif self.args.train_model == 'star':
                # 整体将序列中的最后一帧的数据删除 20帧-》19帧
                inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], \
                                 nei_list[:-1], nei_num[:-1], batch_pednum
                outputs, _, _ = self.net.forward(inputs_forward, stage='support', mean_list=[], var_list=[],
                                                 iftest=False)
                # lossmask 表示当前帧和上一帧中是否都存在数据。该掩码用于计算损失函数时去除缺失数据的贡献，避免缺失数据对损失函数的计算造成影响。
                lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
                # 此处的损失（MSE）计算似乎同时计算观测的值和预测的值，是整个20s序列的；不单单是未来12s。
                loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
                loss_pred_recover += (torch.sum(loss_o * lossmask / num))
                loss = loss_pred_recover
                loss_pred,loss_recover,loss_kl,loss_diverse,loss_TT = 0,0,0,0,0

            loss_pred_list += loss_pred
            loss_recover_list += loss_recover
            loss_kl_list +=  loss_kl
            loss_diversity_list += loss_diverse
            loss_TT_list += loss_TT
            loss_epoch += loss.item()
            # 损失反向传播 梯度裁剪 优化器的step函数
            loss.backward()
            """
            对神经网络模型的梯度进行裁剪;self.net.parameters() 表示获取神经网络模型中所有可训练参数的迭代器。
            然后，nn.utils.clip_grad_norm_() 函数会计算出所有参数的梯度的范数并进行裁剪，以避免梯度爆炸的问题。
            self.args.clip 是梯度裁剪的阈值，表示梯度的范数超过这个值时，会将梯度进行缩放，以使梯度的范数不超过这个阈值。
            clip_grad_norm_() 函数的返回值是裁剪后的梯度范数值;这个函数会修改模型参数的梯度值，
            因此在调用 backward() 函数计算梯度之后，但在执行 step() 函数更新参数之前，通常会使用这个函数对梯度进行裁剪。
            """
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)

            self.optimizer.step()

            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                self.logger.info('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}'.format(
                    batch,self.dataloader.train_batchnums,epoch, loss.item(),end - start)
)
        # 计算所有训练batch的平均损失
        train_loss_epoch = loss_epoch / self.dataloader.train_batchnums
        loss_pred_epoch = loss_pred_list / self.dataloader.train_batchnums
        loss_recover_epoch = loss_recover_list / self.dataloader.train_batchnums
        loss_kl_epoch = loss_kl_list /  self.dataloader.train_batchnums
        loss_diverse_epoch = loss_diversity_list / self.dataloader.train_batchnums
        loss_TT_epoch = loss_TT_list / self.dataloader.train_batchnums

        return train_loss_epoch,loss_pred_epoch,loss_recover_epoch,loss_kl_epoch,loss_diverse_epoch,loss_TT_epoch

    def star_mixup_forward(self, model, data, stage):
        """
        loss 不在这里面求解
        """
        model.train()
        data_set = self.dataloader.rotate_shift_batch(data[0])
        # 将数据转成pytorch的tensor格式 将其转移到GPU上
        data_set = tuple([torch.Tensor(i) for i in data_set])
        data_set = tuple([i.cuda() for i in data_set])
        loss = torch.zeros(1).cuda()
        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = data_set
        set_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                       :-1], batch_pednum
        # todo forward需要与对应的参数结合起来  内外参数
        # print('begin ' + stage)
        # todo iftest ?
        outputs, mean_list, var_list = model.forward(set_forward, stage, iftest=False)
        lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
        loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
        # 测试时此处应该batch——norm只有[1:8,:,:2] [7,258,2] outputs[:7,:,:]  lossmask[:7,:] 相应的num也需要变sum(sum[lossmask])
        # lossmask/loss_o size [19(time),258(batch_pednum)] ==> [8,258] (seq_length, num_Peds)
        loss = loss + torch.sum(loss_o * lossmask / num)
        return loss

    def new_star_forward(self,model,data,stage):
        # 基于最新的new-star 包含初步的数据处理与后续的forward分析
        model.train()
        data_set = self.dataloader.rotate_shift_batch(data[0])
        # 将数据转成pytorch的tensor格式 将其转移到GPU上
        data_set = tuple([torch.Tensor(i) for i in data_set])
        data_set = self.to_device(data_set, self.device)
        model.zero_grad()
        total_loss, loss_pred, loss_recover, loss_kl, loss_diverse,mean,var = model.forward(data_set,stage=stage)
        return total_loss, loss_pred, loss_recover, loss_kl, loss_diverse

    def Dual_TT_forward(self,model,data,stage):
        # 基于最新的new-star 包含初步的数据处理与后续的forward分析
        model.train()
        data_set = self.dataloader.rotate_shift_batch(data[0])
        # 将数据转成pytorch的tensor格式 将其转移到GPU上
        data_set = tuple([torch.Tensor(i) for i in data_set])
        data_set = self.to_device(data_set, self.device)
        model.zero_grad()
        total_loss, loss_pred, loss_recover, loss_kl, loss_diverse,loss_TT = model.forward(data_set, stage=stage)
        return total_loss, loss_pred, loss_recover, loss_kl, loss_diverse,loss_TT

    def process_set(self,model,data,stage='support'):
        # 依据self.args.train_model的选项，选择合适的model，基于输入的数据计算并获得对应的loss
        # 此处输入的data 包含batch-data，batch-id
        # 返回的值 固定 如果有model无法返回 则相应的设为默认值0 =》 total_loss,loss_pred,loss_recover,loss_kl,loss_diverse,loss_TT
        total_loss, loss_pred, loss_recover, loss_kl, loss_diverse, loss_TT = 0,0,0,0,0,0
        if self.args.train_model in ['new_star','new_star_hin']:
            total_loss, loss_pred, loss_recover, loss_kl, loss_diverse= self.new_star_forward(model, data,stage=stage)
        elif self.args.train_model == 'star':
            total_loss= self.star_mixup_forward(model, data,stage=stage)
        elif self.args.train_model == 'Dual_TT':
            total_loss,loss_pred,loss_recover,loss_kl,loss_diverse,loss_TT = self.Dual_TT_forward(model,data,stage=stage)
        return total_loss,loss_pred,loss_recover,loss_kl,loss_diverse,loss_TT

    # 合并meta写法1 并行 常用 和new-star框架的代码，基于不同的forwrad函数
    def train_MLDG_mixup_new_epoch_sequential(self, epoch):
        """
        集合MLDG框架重写该代码--注意此处要同时加上meta-train和meta-test的loss，new-model在meta-test时加入
        """
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch,query_loss_epoch,support_loss_epoch = 0,0,0
        for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
            # 数据组织形式【support，query】-》task-》4个task组成一个batch
            self.logger.info(f'begin{epoch}{batch_task_id}')
            start = time.time()
            self.net.zero_grad()
            task_support_loss = []
            """
            注意此处有两种写法--此处为写法1 -- 参考M3L
            写法1：串行：首先依次计算4个meta-train的loss，而后取平均作为train-loss；
            继而在单一meta-test上进行计算的meta-test的loss；后运用test-loss以及train-loss共同更新初始参数
            写法2：并行：按循环，首先第一个域，计算meta-train-loss，而后建立新模型在meta-test上计算test-loss；
            依次循环进行，得到4个train-loss和test-loss；而后加和取平均，用以更新总损失。
            """
            # 1.计算support loss
            for task_id, task_batch_data in enumerate(batch_task_data):
                # 每次循环添加2000M
                support_set_inital = task_batch_data[1]
                support_total_loss, support_loss_pred, support_loss_recover, support_loss_kl, support_loss_diverse,support_loss_TT = self.process_set(self.net,support_set_inital,'support')
                task_support_loss.append(support_total_loss)
            # 对于源域中的四个meta-train，利用原始模型计算出对应的四个loss，而后平均作为meta-train-loss
            task_support_loss = torch.mean(torch.stack(task_support_loss))
            # 2.更新参数
            # --9061M
            names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            # grads caulcate 添加5696M
            grads = torch.autograd.grad(task_support_loss, names_weights_copy.values(), create_graph=True,
                                        retain_graph=True, allow_unused=True)
            new_model = copy.deepcopy(self.net).train().cuda()
            inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
            # 能否复用相应的adam 或则说此处的dict更新需要有选择性的更新
            new_inner_dict = {key: names_weights_copy[key] - self.args.inner_learning_rate * inner_dict_grads[key] for
                              key in names_weights_copy.keys()}
            # todo 问题仍然出现 即对于位置编码层past_encoder.pos_encoder1.pe", "past_encoder.pos_encoder2.pe的参数 元学习无法load？ ==>忽略这些缺失的键 strict参数来控制是否严格匹配键名。将strict参数设置为False可以忽略缺失的键
            new_model.load_state_dict(new_inner_dict,strict=False)
            new_model.zero_grad()
            self.net.zero_grad()
            # del grads 只会减少 grads 变量的引用计数，如果其他地方仍然存在对 grads 的引用，那么内存可能不会立即释放。确保 grads 变量没有其他引用，并且在删除之后没有进一步使用。
            del grads, inner_dict_grads, new_inner_dict
            # 3.计算query loss
            query_set_inital = batch_task_data[0][0] # 此处如此些的原因在于此处4个task中的query值是一样的
            # 2000M
            query_total_loss, query_loss_pred, query_loss_recover, query_loss_kl, query_loss_diverse,query_loss_TT = self.process_set(new_model,query_set_inital,'query')
            task_query_loss = query_total_loss
            # 计算总的task loss
            task_loss = task_support_loss + task_query_loss
            # 转换为Python标量并缓存，避免重复调用`.item()`
            task_query_loss_item = task_query_loss.item()
            task_support_loss_item = task_support_loss.item()
            self.logger.info(f'task_query_loss: {task_query_loss_item}, task_support_loss: {task_support_loss_item}')
            # 累加到对应的epoch级别的损失变量
            query_loss_epoch += task_query_loss_item
            support_loss_epoch += task_support_loss_item
            # 直接使用之前转换的标量值
            loss_epoch += task_query_loss_item + task_support_loss_item
            # 如果query和support的对齐loss的话 相应的此处一方面需要对应的Q和K值的输出，可以形成一个dict，另一方面需要定义一个完整的loss计算方式
            # 4.损失反向传播 更新
            self.optimizer.zero_grad()
            task_loss.backward()
            # 分析task-loss 计算的梯度时self.net（support-loss）和new_model（query-loss）都更新了grads
            # 此处需要将new——model计算得到的梯度叠加给self.net
            for old, new in zip(self.net.named_parameters(), new_model.named_parameters()):
                # 返回一个tuple 【名称，参数tensor】
                old[1].grad += new[1].grad
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                self.logger.info(f'train-{batch_task_id}/{len(self.dataloader.train_batch_MLDG_task)} (epoch {epoch}), train_loss = {task_loss.item():.5f}, time/batch = {end - start:.5f}')
            del task_support_loss, task_query_loss, task_loss, query_total_loss
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        train_support_loss_epoch = support_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        train_query_loss_epoch = query_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        self.logger.info(f'epoch {epoch}, loss = {train_loss_epoch:.5f}, support_loss = {train_support_loss_epoch:.5f}, query_loss = {train_query_loss_epoch:.5f}')
        return train_loss_epoch,train_support_loss_epoch,train_query_loss_epoch
    # 合并meta写法2 串行 常用 和new-star框架的代码，基于不同的forward函数
    def train_MLDG_mixup_new_epoch_parallel(self, epoch):
        """
        结合MLDG框架重写该部分代码 -- new-ModelStrategy 在测试时加入，对应的并行方法
        """
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch, support_loss_epoch ,query_loss_epoch = 0,0,0
        start = time.time()
        for batch_task_id, batch_task_data in tqdm(enumerate(self.dataloader.train_batch_MLDG_task),
                                                   total=len(self.dataloader.train_batch_MLDG_task),
                                                   desc="Training Progress"):
        #for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
            #self.logger.info(f'begin{epoch}{batch_task_id}')
            self.net.zero_grad()
            task_support_loss,task_query_loss = [],[]
            for task_id, task_batch_data in enumerate(batch_task_data):
                support_set_inital,query_set_inital  = task_batch_data[1], task_batch_data[0]
                support_total_loss, support_loss_pred, support_loss_recover, support_loss_kl, support_loss_diverse,\
                support_loss_TT = self.process_set(self.net, support_set_inital, 'support')
                task_support_loss.append(support_total_loss)
                # 浅拷贝 names_weights_copy的值随着self.net改变,todo 此处删除了部分参数 或许出问题
                names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
                grads = torch.autograd.grad(support_total_loss, names_weights_copy.values(), create_graph=True,
                                            retain_graph=True, allow_unused=True)
                new_model = copy.deepcopy(self.net).train().cuda()
                # 在直接操作.grad属性之前，确保它不是None。如果是None，你需要先初始化它为0。可能错误在于此时的self.net的grad与grads的梯度不一样
                """
                for param in self.net.parameters():
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                        self.logger.error(f'param{param},grad is None')
                """
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.args.inner_learning_rate * inner_dict_grads[key]
                                  for key in names_weights_copy.keys()}
                # 示例：在加载前检查键的匹配
                self.load_model_with_selective_strictness(new_model, new_inner_dict)
                new_model.zero_grad()
                self.net.zero_grad()
                del grads, inner_dict_grads, new_inner_dict
                query_total_loss, query_loss_pred, query_loss_recover, query_loss_kl, query_loss_diverse, \
                query_loss_TT = self.process_set(new_model, query_set_inital, 'query')
                task_query_loss.append(query_total_loss)
            task_support_loss = torch.mean(torch.stack(task_support_loss))
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            task_loss = task_support_loss + task_query_loss
            # 调试分析，相应的task-support-loss更新的是self.net的梯度，而task-query-loss更新的是new——model的梯度，
            # 故而此处每轮参数更新是按原来写法，不传递累加梯度的话其实只是用support的loss
            # 此处需要进行debug分析，分析相应的query-loss或support-loss是否更新了模型的梯度，即需要证明正确性。
            support_loss_epoch += task_support_loss.item()
            query_loss_epoch += task_query_loss.item()
            loss_epoch += task_loss.item()
            self.optimizer.zero_grad()
            task_loss.backward()
            # 分析task-loss 计算的梯度时self.net（support-loss）和new_model（query-loss）都更新了grads
            # 此处需要将new——model计算得到的梯度叠加给self.net
            for old, new in zip(self.net.named_parameters(), new_model.named_parameters()):
                # 返回一个tuple 【名称，参数tensor】
                old[1].grad += new[1].grad
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            #self.logger.info(f'train-{batch_task_id}/{len(self.dataloader.train_batch_MLDG_task)} (epoch {epoch}),task_support_loss = {task_support_loss.item():.5f},task_query_loss = {task_query_loss.item():.5f}')
        train_support_loss_epoch = support_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        train_query_loss_epoch = query_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        end = time.time()
        self.logger.info(
            f'epoch {epoch}, loss = {train_loss_epoch:.5f}, support_loss = {train_support_loss_epoch:.5f}, query_loss = {train_query_loss_epoch:.5f},time/batch = {end - start:.5f}')
        return train_loss_epoch,train_support_loss_epoch,train_query_loss_epoch

    def train_MVDG_epoch_new(self, epoch):
        """
        结合MVDG框架重写该部分代码；
        每次有三个轨迹，每个轨迹内部有4个task；相应的需要注意此处3个轨迹是添加训练时间还是原始的batch数分3块；两种都试一下
        内外循环 reptile
        此处的写法是将原有的batch在分成3个部分
        ==》分析数据集本身 发现eth确实与其他四个相差较大，行人更少，速度更快；
        ==》实验思路：包括调节学习率，更改数据生成。
        todo 现有的结果是hotel0.25-0.22-0.18已经有显著提升，zara1,zara2,univ仍然在下降中，eth效果混乱，没有学到泛化性，对于源域过拟合了
              后续需要针对eth多次实验，观测器参数是否进入极小值点了，以及相应的以eth为测试域，或则说用另外其他四个做训练会有什么区别
        """
        # 第一步依据完整数据拆分出tra，batch，task
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        MVDG_optimizers = torch.optim.Adam(self.net.parameters(), lr=self.args.outer_learning_rate)
        self.net.zero_grad()
        fast_models = []
        for batch_id, batch_data in tqdm(enumerate(self.dataloader.train_batch_MVDG_task),
                                         total=len(self.dataloader.train_batch_MVDG_task),
                                         desc="Processing MVDG Batches"):
        #for batch_id, batch_data in enumerate(self.dataloader.train_batch_MVDG_task):
            # 每个batch数据包含3个traj
            # self.logger.info(f'begin{epoch},batch_traj{batch_id}')
            start = time.time()
            # 此处每个traj——data有4个task
            task_query_loss = []
            for traj_id, traj_data in enumerate(batch_data):
                # 一个轨迹里出一个系数
                # self.logger.info(f'begin{epoch},batch_traj{batch_id},optim_traj{traj_id}')
                fast_model = copy.deepcopy(self.net).train().cuda()
                fast_opts = torch.optim.Adam(fast_model.parameters(), lr=self.args.inner_learning_rate,
                                             betas=(0.9, 0.999),
                                             weight_decay=5e-4)
                # 每个task内包含一个support和query
                traj_query_loss = []
                for task_id, task_data in enumerate(traj_data):
                    # todo 如果有多个任务的话 则会反复更新 级联形式 ==》 后续其实就是基于此处进行更改论文的模型,添加对应的对齐loss
                    support_set_inital,query_set_inital = task_data[1],task_data[0]
                    # 1.计算support-loss
                    support_total_loss, support_loss_pred, support_loss_recover, support_loss_kl, \
                    support_loss_diverse, support_loss_TT = self.process_set(fast_model,support_set_inital,'support')
                    # 2.更新fast-model
                    fast_opts.zero_grad()
                    support_total_loss.backward()
                    fast_opts.step()
                    # 3.计算query-loss
                    query_total_loss, query_loss_pred, query_loss_recover, query_loss_kl, \
                    query_loss_diverse, query_loss_TT = self.process_set(fast_model,query_set_inital,'query')
                    # todo 第二种写法，在此处更改 MAML写法 加和两种loss
                    traj_query_loss.append(query_total_loss)
                    # 4.更新fast-model
                    fast_opts.zero_grad()
                    query_total_loss.backward()
                    fast_opts.step()
                task_query_loss.append(torch.mean(torch.stack(traj_query_loss)))
                parameters = dict(fast_model.named_parameters())
                fast_models.append(parameters)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            # self.logger.info(f'task_query_loss:{task_query_loss.item()}')
            loss_epoch += task_query_loss.item()
            # parameters字典中的值是模型参数tensor的直接引用,不是copy。
            # 所以修改字典值实际上就是在修改模型参数内存中的tensor值。
            MVDG_params = dict(self.net.named_parameters())
            MVDG_optimizers.zero_grad()
            # update_grad
            for k in MVDG_params.keys():
                new_v, old_v = 0, MVDG_params[k]
                for m in fast_models:
                    new_v += m[k]
                new_v = new_v / len(fast_models)
                MVDG_lr = 1
                MVDG_params[k].grad = ((old_v - new_v) / MVDG_lr).data
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            MVDG_optimizers.step()
            end = time.time()
            if batch_id % self.args.show_step == 0 and self.args.ifshow_detail:
                self.logger.info(f'train-{batch_id}/{len(self.dataloader.train_batch_MVDG_task)} (epoch {epoch}),task_query_loss = {task_query_loss.item():.5f},time/batch = {end - start:.5f}')
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MVDG_task)
        return train_loss_epoch

    def train_MVDGMLDG_epoch_sequential(self, epoch):
        """
        结合MVDG框架重写该部分代码；
        每次有三个轨迹，每个轨迹内部有4个task；相应的需要注意此处3个轨迹是添加训练时间还是原始的batch数分3块；两种都试一下
        内外循环 reptile
        此处的写法是将原有的batch在分成3个部分
        ==》分析数据集本身 发现eth确实与其他四个相差较大，行人更少，速度更快；
        ==》实验思路：包括调节学习率，更改数据生成。
        todo 现有的结果是hotel0.25-0.22-0.18已经有显著提升，zara1,zara2,univ仍然在下降中，eth效果混乱，没有学到泛化性，对于源域过拟合了
              后续需要针对eth多次实验，观测器参数是否进入极小值点了，以及相应的以eth为测试域，或则说用另外其他四个做训练会有什么区别
        """
        # 第一步依据完整数据拆分出tra，batch，task
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        MVDG_optimizers = torch.optim.Adam(self.net.parameters(), lr=self.args.outer_learning_rate)
        self.net.zero_grad()
        fast_models = []
        for batch_id, batch_data in enumerate(self.dataloader.train_batch_MVDG_task):
            # 每个batch数据包含3个traj
            print('begin' + str(epoch) + 'batch_traj' + str(batch_id))
            start = time.time()
            # 此处每个traj——data有4个task
            task_query_loss = []
            for traj_id, traj_data in enumerate(batch_data):
                # print('begin' + str(epoch) + 'batch_traj' + str(batch_id) + 'optim_traj' + str(traj_id))
                fast_model = copy.deepcopy(self.net).train().cuda()
                fast_opts = torch.optim.Adam(fast_model.parameters(), lr=self.args.inner_learning_rate,
                                             betas=(0.9, 0.999),
                                             weight_decay=5e-4)
                # 每个task内包含一个support和query
                traj_query_loss,traj_support_loss = [],[]
                mean_list,var_list =[],[]
                for task_id,task_data in enumerate(traj_data):
                    support_set_inital = task_data[1]
                    if self.args.train_model == 'new_star' or self.args.train_model =='new_star_hin':
                        support_total_loss, support_loss_pred, support_loss_recover, support_loss_kl, \
                        support_loss_diverse, mean_support, var_support = self.new_star_forward(fast_model,
                                                                                                support_set_inital,
                                                                                                stage='support',
                                                                                                mean_list=[],
                                                                                                var_list=[],
                                                                                                ifmixup=self.args.ifmixup)
                    elif self.args.train_model == 'star':
                        support_total_loss, mean_support, var_support = self.star_mixup_forward(fast_model, support_set_inital,
                                                                                          stage='support', mean_list=[],
                                                                                          var_list=[], ifmixup=self.args.ifmixup)
                    mean_list.append(mean_support)
                    var_list.append(var_support)
                    traj_support_loss.append(support_total_loss)
                traj_support_loss = torch.mean(torch.stack(traj_support_loss))
                names_weights_copy = self.get_inner_loop_parameter_dict(fast_model.named_parameters())
                grads = torch.autograd.grad(traj_support_loss, names_weights_copy.values(), create_graph=True,
                                        retain_graph=True, allow_unused=True)
                fast_new_model = copy.deepcopy(fast_model).train().cuda()
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.args.inner_learning_rate * inner_dict_grads[key]
                                  for key in names_weights_copy.keys()}
                fast_new_model.load_state_dict(new_inner_dict)
                fast_new_model.zero_grad()
                fast_model.zero_grad()
                del grads, inner_dict_grads, new_inner_dict
                query_set_inital = traj_data[0][0]
                if self.args.train_model == 'new_star' or self.args.train_model =='new_star_hin':
                    query_total_loss, query_loss_pred, query_loss_recover, query_loss_kl, \
                    query_loss_diverse, _, _ = self.new_star_forward(fast_new_model, query_set_inital, stage='query',
                                                                     mean_list=mean_list, var_list=var_list,
                                                                     ifmixup=self.args.ifmixup)
                elif self.args.train_model == 'star':
                    query_total_loss, _, _ = self.star_mixup_forward(fast_new_model, query_set_inital, stage='query',
                                                                     mean_list=mean_support, var_list=var_support,
                                                                     ifmixup=self.args.ifmixup)
                traj_query_loss = query_total_loss
                traj_loss = traj_support_loss + traj_query_loss
                fast_opts.zero_grad()
                traj_loss.backward()
                for old, new in zip(fast_model.named_parameters(), fast_new_model.named_parameters()):
                    # 返回一个tuple 【名称，参数tensor】
                    old[1].grad += new[1].grad
                torch.nn.utils.clip_grad_norm_(fast_model.parameters(), self.args.clip)
                fast_opts.step()

                task_query_loss.append(traj_query_loss)
                parameters = dict(fast_model.named_parameters())
                fast_models.append(parameters)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            # print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
            loss_epoch = loss_epoch + task_query_loss.item()
            # parameters字典中的值是模型参数tensor的直接引用,不是copy。
            # 所以修改字典值实际上就是在修改模型参数内存中的tensor值。
            MVDG_params = dict(self.net.named_parameters())
            MVDG_optimizers.zero_grad()
            # update_grad
            for k in MVDG_params.keys():
                new_v, old_v = 0, MVDG_params[k]
                for m in fast_models:
                    new_v += m[k]
                new_v = new_v / len(fast_models)
                MVDG_lr = 1
                MVDG_params[k].grad = ((old_v - new_v) / MVDG_lr).data
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            MVDG_optimizers.step()
            end = time.time()
            if batch_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_id, len(self.dataloader.train_batch_MVDG_task), epoch, task_query_loss.item(),
                    end - start))
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MVDG_task)
        return train_loss_epoch

    def train_MVDGMLDG_epoch_parallel(self,epoch):
        """
        结合MVDG框架重写该部分代码；
        每次有三个轨迹，每个轨迹内部有4个task；相应的需要注意此处3个轨迹是添加训练时间还是原始的batch数分3块；两种都试一下
        内外循环 + reptile
        此处的写法是将原有的batch在分成3个部分
        multi-task 中的梯度更新方式改为损失加和

        """
        # 第一步依据完整数据拆分出tra，batch，task
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        MVDG_optimizers = torch.optim.Adam(self.net.parameters(), lr=self.args.outer_learning_rate)
        self.net.zero_grad()
        fast_models = []
        for batch_id, batch_data in enumerate(self.dataloader.train_batch_MVDG_task):
            print('begin' + str(epoch) + 'batch_traj' + str(batch_id))
            start = time.time()
            # 此处每个traj——data有4个task
            traj_query_loss = []
            for traj_id, traj_data in enumerate(batch_data):
                fast_model = copy.deepcopy(self.net).train().cuda()
                fast_opts = torch.optim.Adam(fast_model.parameters(), lr=self.args.inner_learning_rate,
                                             betas=(0.9, 0.999),weight_decay=5e-4)
                task_query_loss= []
                for task_id, task_data in enumerate(traj_data):
                    support_set_inital = task_data[1]
                    query_set_inital = task_data[0]
                    if self.args.train_model == 'new_star' or self.args.train_model =='new_star_hin':
                        support_total_loss, support_loss_pred, support_loss_recover, support_loss_kl, \
                        support_loss_diverse, mean_support, var_support = self.new_star_forward(fast_model,
                                                                                                support_set_inital,
                                                                                                stage='support',
                                                                                                mean_list=[],
                                                                                                var_list=[],
                                                                                                ifmixup=self.args.ifmixup)
                    elif self.args.train_model == 'star':
                        support_total_loss, mean_support, var_support = self.star_mixup_forward(fast_model,
                                                                                                support_set_inital,
                                                                                                stage='support',
                                                                                                mean_list=[],
                                                                                                var_list=[],
                                                                                                ifmixup=self.args.ifmixup)
                    names_weights_copy = self.get_inner_loop_parameter_dict(fast_model.named_parameters())
                    grads = torch.autograd.grad(support_total_loss, names_weights_copy.values(), create_graph=True,
                                                retain_graph=True, allow_unused=True)
                    fast_new_model = copy.deepcopy(fast_model).train().cuda()
                    inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                    new_inner_dict = {
                        key: names_weights_copy[key] - self.args.inner_learning_rate * inner_dict_grads[key]
                        for key in names_weights_copy.keys()}
                    fast_new_model.load_state_dict(new_inner_dict)
                    fast_new_model.zero_grad()
                    fast_model.zero_grad()
                    del grads, inner_dict_grads, new_inner_dict
                    if self.args.train_model == 'new_star' or self.args.train_model =='new_star_hin':
                        query_total_loss, query_loss_pred, query_loss_recover, query_loss_kl, \
                        query_loss_diverse, _, _ = self.new_star_forward(fast_new_model, query_set_inital,
                                                                         stage='support',
                                                                         mean_list=[],
                                                                         var_list=[],
                                                                         ifmixup=self.args.ifmixup)
                    elif self.args.train_model == 'star':
                        query_total_loss, _, _ = self.star_mixup_forward(fast_new_model, query_set_inital,
                                                                         stage='support',
                                                                         mean_list=[],
                                                                         var_list=[],
                                                                         ifmixup=self.args.ifmixup)
                    task_loss = support_total_loss + query_total_loss
                    fast_opts.zero_grad()
                    task_loss.backward()
                    for old, new in zip(fast_model.named_parameters(), fast_new_model.named_parameters()):
                        # 返回一个tuple 【名称，参数tensor】
                        old[1].grad += new[1].grad
                    torch.nn.utils.clip_grad_norm_(fast_model.parameters(), self.args.clip)
                    fast_opts.step()
                    task_query_loss.append(query_total_loss)
                traj_query_loss.append(torch.mean(torch.stack(task_query_loss)))
                parameters = dict(fast_model.named_parameters())
                fast_models.append(parameters)
            batch_query_loss =  torch.mean(torch.stack(traj_query_loss))
            loss_epoch += batch_query_loss.item()
            # parameters字典中的值是模型参数tensor的直接引用,不是copy。
            # 所以修改字典值实际上就是在修改模型参数内存中的tensor值。
            MVDG_params = dict(self.net.named_parameters())
            MVDG_optimizers.zero_grad()
            # update_grad
            for k in MVDG_params.keys():
                new_v, old_v = 0, MVDG_params[k]
                for m in fast_models:
                    new_v += m[k]
                new_v = new_v / len(fast_models)
                MVDG_lr = 1
                MVDG_params[k].grad = ((old_v - new_v) / MVDG_lr).data
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            MVDG_optimizers.step()
            end = time.time()
            if batch_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_id, len(self.dataloader.train_batch_MVDG_task), epoch, batch_query_loss.item(),
                    end - start))
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MVDG_task)
        return train_loss_epoch

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        将相应的参数转移到指定设备上，并排除部分不需要优化的参数，或特殊要求下的参数；最终获得参数的字典，需要关注后续如何将参数字典给到模型
        """
        return {
            name: param.to(device=self.device)
            for name, param in params
            if param.requires_grad
        }

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
        error_rmse_epoch,error_mae_epoch = 0,0,
        error_rmse_count_epoch,error_mae_count_epoch = 1e-5, 1e-5
        for batch in tqdm(range(self.dataloader.test_batchnums)):
            # 与train相同的batch处理步骤
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            all_output = []
            # 同一组数据 预测20次
            for i in range(self.args.sample_num):
                # 相应的设置iftest标记为true 测试模式
                outputs_infer, _, _ = self.net.forward(inputs_forward, stage='support', mean_list=[], var_list=[],
                                                       iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)
            # 此处all——output为[20,19,257,2]
            # lossmask [19,257]
            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # todo 相较于train的MSE 此处只计算在整个时间窗口都存在的行人的损失
            # 需要添加 mse=》rmse，mae的
            error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_norm[1:, :, :2],
                                                                        self.args.obs_length, lossmask,self.args.sample_num)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            # 添加对应的 rmse mae
            error_rmse, error_rmse_count, error_mae, error_mae_count = L2forTest_RMSE_MAE(all_output,
                                      batch_norm[1:, :, :2],self.args.obs_length,lossmask,self.args.sample_num)
            error_rmse_epoch += error_rmse
            error_rmse_count_epoch += error_rmse_count
            error_mae_epoch += error_mae
            error_mae_count_epoch += error_mae_count
        
        print("RMSE"+str(error_rmse_epoch/error_rmse_count_epoch))
        print("MAE"+str(error_mae_epoch/error_mae_count_epoch))
        
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    @torch.no_grad()
    def test_new_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
        error_rmse_epoch, error_mae_epoch = 0, 0,
        error_rmse_count_epoch, error_mae_count_epoch = 1e-5, 1e-5
        for batch in tqdm(range(self.dataloader.test_batchnums)):
            # 与train相同的batch处理步骤
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])
            with torch.no_grad():
                prediction, target_pred_traj = self.net.inference(inputs)
                # prediction (sample_num, agent_num, self.pred_length, 2) -> (sample_num, self.pred_length,agent_num, 2)
                # target_pred_traj  (self.pred_length,agent_num, 2) 进去的时候为512人，出来的时候只有115人 那么需要对应这115人的信息
                prediction = prediction.permute(0, 2, 1, 3)
            # error
            error, error_cnt, final_error, final_error_cnt = L2forTestS_NEWSTAR(prediction=prediction,target_pred_traj=target_pred_traj,num_samples=self.args.sample_num)
            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            # rmse-error
            error_rmse, error_rmse_count, error_mae, error_mae_count = L2forTest_RMSE_MAE_NEWSTAR(outputs=prediction,targets=target_pred_traj,num_samples=self.args.sample_num)
            error_rmse_epoch += error_rmse
            error_rmse_count_epoch += error_rmse_count
            error_mae_epoch += error_mae
            error_mae_count_epoch += error_mae_count
            """
            #  L2 范数  error (num_samples, pred_length, num_Peds) inputs (261人 6-43)epoch 0 batch 0 (20 2 138 2)
            error_full = torch.norm(prediction - target_pred_traj, p=2, dim=3)
            # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
            error_full_sum = torch.sum(error_full, dim=1)
            # error_full_sum (20,pde-full) error_full_sum (1,pde-full)
            error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)
            best_error = []
            # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
            for index, value in enumerate(min_index):
                best_error.append(error_full[value, :, index])
            best_error = torch.stack(best_error)
            best_error = best_error.permute(1, 0)
            # error为总的误差 所有行人不同采样中的最佳值累加
            error = torch.sum(error_full_sum_min)
            # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
            error_cnt = error_full.numel() / self.args.sample_num
            # 只取终点位置 其为FDE值
            final_error = torch.sum(best_error[-1])
            final_error_cnt = error_full.shape[-1]
            error_epoch += error.item()
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error.item()
            final_error_cnt_epoch += final_error_cnt
            """
        print("RMSE" + str(error_rmse_epoch / error_rmse_count_epoch))
        print("MAE" + str(error_mae_epoch / error_mae_count_epoch))
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    @torch.no_grad()
    def test_new_visualSDD_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
        sdd_vis = []
        for batch in tqdm(range(self.dataloader.test_batchnums)):
            # 与train相同的batch处理步骤
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            scene_frame = pd.DataFrame(columns=['scene', 'frame'])
            for tpl in batch_id:
                repeated_tpl = np.tile(tpl[:2], (tpl[2], 1))  # 重复前两个元素，并复制第三个元素的次数
                repeated_df = pd.DataFrame(repeated_tpl, columns=['scene', 'frame'])
                scene_frame = pd.concat([scene_frame, repeated_df])
            scene_frame.reset_index(drop=True, inplace=True)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])
            with torch.no_grad():
                prediction, nodes_current,new_scene_frame = self.net.inference_visual(inputs,scene_frame)
                # prediction (sample_num, agent_num, self.pred_length, 2) -> (sample_num, self.pred_length,agent_num, 2)
                # target_pred_traj  (self.pred_length,agent_num, 2) 进去的时候为512人，出来的时候只有115人 那么需要对应这115人的信息
                prediction = prediction.permute(0, 2, 1, 3)
            #  L2 范数  error_full (num_samples, pred_length, num_Peds) inputs (261人 6-43)epoch 0 batch 0 (20 2 138 2)
            target_pred_traj = nodes_current[self.args.obs_length:]
            error_full = torch.norm(prediction - target_pred_traj, p=2, dim=3)
            # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
            error_full_sum = torch.sum(error_full, dim=1)
            # error_full_sum (20,pde-full) error_full_sum_min (1,pde-full)
            error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)
            best_error = []
            # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
            for index, value in enumerate(min_index):
                best_error.append(error_full[value, :, index])
            best_error = torch.stack(best_error)
            best_error = best_error.permute(1, 0)
            # error为总的误差 所有行人不同采样中的最佳值累加
            error = torch.sum(error_full_sum_min)
            # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
            error_cnt = error_full.numel() / self.args.sample_num
            # 只取终点位置 其为FDE值
            final_error = torch.sum(best_error[-1])
            final_error_cnt = error_full.shape[-1]
            error_epoch += error.item()
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error.item()
            final_error_cnt_epoch += final_error_cnt

            # 一个场景只需要一副图像即可 不需要去管如coupa里的0-1-2；初步设置 每个batch里找一张图片
            # top_indices = torch.topk(error_full_sum_min,k=1,largest=True).indices
            sorted_indices = torch.argsort(error_full_sum_min,descending=True)  # 降序排列
            k = self.args.k_best
            top_indices = sorted_indices[k - 1]  # 减1是因为Python的索引是从0开始的
            scene_nodes_current = nodes_current[:,top_indices] # 单独的过去

            scene_nodes_prediction = prediction[:,:,top_indices] # 未来轨迹
            scene_name = new_scene_frame.iloc[top_indices.cpu().numpy()]['scene']
            scene_data_dict={'scene':scene_name,'truth_traj':scene_nodes_current,'predict_traj':scene_nodes_prediction}
            sdd_vis.append(scene_data_dict)
        vis_num = len(sdd_vis)
        if self.args.vis == 'traj':
            draw_result_SDD(sdd_vis,self.args)
        elif self.args.vis == 'traj_comparison':
            draw_result_SDD_comparison(sdd_vis,self.args)
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    @torch.no_grad()
    def test_new_visualNBA_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
        visual_batch_inputs = []
        visual_batch_outputs = []
        visual_batch_error = []
        for batch in tqdm(range(self.dataloader.test_batchnums)):
            # 第一步：获取对应的预测结果 预测结果取的是20次预测中每个独立个体最好的一次 然后进行保存
            # 与train相同的batch处理步骤
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])
            with torch.no_grad():
                prediction, target_pred_traj = self.net.inference(inputs)
                # prediction (sample_num, agent_num, self.pred_length, 2) -> (sample_num, self.pred_length,agent_num, 2)
                # target_pred_traj  (self.pred_length,agent_num, 2)
                prediction = prediction.permute(0, 2, 1, 3)
            # 过去轨迹inputs[0] nodes_abs [15,520,2] 真实的轨迹[10,520,2] 完全对应的 预测的轨迹 prediction [20,10,520,2] 此处的520 是因为设置512个行人数据
            # 所有的行人被统一拼接到对应的20帧内 则后续需要绘画结果时 无法依照时间和地点去取出对应的值
            # 设计格式 [meta-id,start-frame,end-frame,sceneid]得添加一个数组 标注每个inputs相应的name和id；从而方便后期依照对应的id和name从统一的保存的预测结果中索引对应的值
            # 针对于NBA数据集 可以进行相应的简化 因为其每次都是10个人 分析清楚如何拆分的 即可将原始数据进行拆分
            error_full = torch.norm(prediction - target_pred_traj, p=2,
                                    dim=3)  # 计算xy对应的误差 由【20,10,520,2】转变成error_full【20,10,520】
            error_full_sum = torch.sum(error_full, dim=1)  # 汇总这10s的总误差 【20，10,520】-》error_full_sum 【20.520】
            error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)  # 从20个选出最好的一个 【20,520】 -> 【520】
            best_error,prediction_mindex = [],[]
            for index, value in enumerate(min_index):
                best_error.append(error_full[value, :, index])
                prediction_mindex.append(prediction[value, :, index, :])

            best_error = torch.stack(best_error)
            best_error = best_error.permute(1, 0)
            prediction_mindex = torch.stack(prediction_mindex)
            prediction_mindex = prediction_mindex.permute(1,0,2)
            # error为总的误差 所有行人不同采样中的最佳值累加
            error = torch.sum(error_full_sum_min)
            # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
            error_cnt = error_full.numel() / self.args.sample_num
            # 只取终点位置 其为FDE值
            final_error = torch.sum(best_error[-1])
            final_error_cnt = error_full.shape[-1]
            error_epoch += error.item()
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error.item()
            final_error_cnt_epoch += final_error_cnt

            # 依照10人一组 将这个520拆分开来并汇总误差 选取最好的前3个和最差的1个 两轮选取 第一轮在每个batch内部 第二轮在整个数据集选取
            error_full_summin_reshaped = error_full_sum_min.reshape(-1, 10)
            group_sums = torch.sum(error_full_summin_reshaped, dim=1)
            top_2_indices = torch.topk(group_sums, k=1, largest=False).indices
            worst_index = torch.argmax(group_sums)
            # 从对应的原始预测数据中取出 历史值和未来预测值
            for index, value in enumerate(top_2_indices):
                visual_batch_inputs.append(inputs[0][:, value * 10:(value + 1) * 10, :])
                # 拼接历史的origin与预测的prediction 成为完整数据
                full_prediction_withinputs = torch.cat((inputs[0][0:5, value * 10:(value + 1) * 10, :],
                                                        prediction_mindex[:, value * 10:(value + 1) * 10, :]), dim=0)
                visual_batch_outputs.append(full_prediction_withinputs)
                visual_batch_error.append(group_sums[value])
        #  依据所有的实验batch数据 在进行一次排序 选取相应的最小的5个error数据
        min_indices_all = [index for index, value in sorted(enumerate(visual_batch_error), key=lambda x: x[1])[:5]]
        gt_data, pre_data, scene = [], [], []
        for index, value in enumerate(min_indices_all):
            gt_data.append(visual_batch_inputs[value].cpu().numpy())
            pre_data.append(visual_batch_outputs[value].cpu().numpy())
            scene.append('nba')
        print('ADE: '+str(error_epoch / error_cnt_epoch)+'FDE: '+str(final_error_epoch / final_error_cnt_epoch))
        draw_result_NBA(gt_data,scene,self.args,mode='gt')
        print('finish all gt png')
        draw_result_NBA(pre_data, scene, self.args, mode='pre')
        print('finish all pre png')
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    @torch.no_grad()
    def test_new_sne_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        ADE_dict,FDE_dict,Feature_dict = {},{},{}
        for scene,scene_data in self.dataloader.test_snedata.items():
            print('evluate and save data in '+str(scene))
            error_epoch, final_error_epoch = 0, 0,
            error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5
            scene_feature = []
            for batch in tqdm(range(len(scene_data))):
                # 分别处理 此处与train相同的处理步骤 但相应的需要多返回一个past-feature的值
                inputs = self.dataloader.rotate_shift_batch(scene_data[batch][0],ifrotate=False)
                inputs = tuple([torch.Tensor(i) for i in inputs])
                if self.args.using_cuda:
                    inputs = tuple([i.cuda() for i in inputs])
                with torch.no_grad():
                    #features(num_peds,dims)
                    prediction,target_pred_traj,features = self.net.inference(inputs)
                    prediction = prediction.permute(0, 2, 1, 3)
                # 转为numpy保存
                scene_feature.append(features)
                # ===== 也可以尝试一下都跑跑看看效果如何
                #  L2 范数  error (num_samples, pred_length, num_Peds) inputs (261人 6-43)epoch 0 batch 0 (20 2 138 2)
                error_full = torch.norm(prediction - target_pred_traj, p=2, dim=3)
                # 选择预测误差最小的一组 并保存 ; ，每个行人在其20次采样中挑选最好的
                error_full_sum = torch.sum(error_full, dim=1)
                # error_full_sum (20,pde-full) error_full_sum (1,pde-full)
                error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)
                best_error = []
                # 添加每个行人最好采样下的pred-seq的error数据 （pred-seq，pedi-full）
                for index, value in enumerate(min_index):
                    best_error.append(error_full[value, :, index])
                best_error = torch.stack(best_error)
                best_error = best_error.permute(1, 0)
                # error为总的误差 所有行人不同采样中的最佳值累加
                error = torch.sum(error_full_sum_min)
                # error_cnt:相应的为损失计算中的总行人数 (obs_length * num_samples * num_Peds) / num_samples = obs_length  * num_Peds
                error_cnt = error_full.numel() / self.args.sample_num
                # 只取终点位置 其为FDE值
                final_error = torch.sum(best_error[-1])
                final_error_cnt = error_full.shape[-1]
                error_epoch += error.item()
                error_cnt_epoch += error_cnt
                final_error_epoch += final_error.item()
                final_error_cnt_epoch += final_error_cnt
            scene_feature_single = torch.cat(scene_feature,dim=0)
            scene_feature_single = scene_feature_single.cpu().numpy()
            scene_ADE = error_epoch/error_cnt_epoch
            scene_FDE = final_error_epoch/final_error_cnt_epoch
            ADE_dict[scene] = scene_ADE
            FDE_dict[scene] = scene_FDE
            Feature_dict[scene] = scene_feature_single
        print('beigin draw pictures !!')
        # 依据相应特征数据 进行后续的降维度分析
        # 299 :hotel:1079 zara01:2376 zara02:5958 univ:27880 eth:2614
        # 52:
        VISUAL_TSNE(data_dict=Feature_dict,args=self.args)
        return ADE_dict,FDE_dict
    # 原始meta的代码
    def train_meta_epoch(self, epoch):
        """
        结合Meta框架重写该部分代码；
        内外循环
        """
        # 第一步依据完整数据拆分出batch list
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
            # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
            #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
            # 针对batch-task-data中的4个task进行处理 list
            print('begin' + str(epoch) + str(batch_task_id))
            start = time.time()
            self.net.zero_grad()
            # !!!!(1)注意的是 state——dict是浅拷贝，即net-initial-dict改变的话，那么当你修改param，相应地也会修改model的参数。
            # model这个对象实际上是指向各个参数矩阵的，而浅拷贝只会拷贝最外层的这些“指针；
            # from copy import deepcopy  best_state = copy.deepcopy(ModelStrategy.state_dict()) 深拷贝 互不影响
            net_initial_dict = copy.deepcopy(self.net.state_dict())
            task_query_loss = []
            for task_id, task_batch_data in enumerate(batch_task_data):
                # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
                print('begin' + str(epoch) + '--' + str(batch_task_id) + '---' + str(task_id))
                # 1 !!!! (2)每次都从1开始 清零 初始化一个self.new——model的话会导致反复更新累加 导致原位操作 7-9
                new_model = STAR(self.args).cuda()
                # 2
                new_model.load_state_dict(net_initial_dict)
                new_model.zero_grad()
                # 准备数据
                support_set_inital = task_batch_data[1]
                query_set_inital = task_batch_data[0]
                # todo forward需要与对应的参数结合起来  内外参数
                support_loss = self.meta_forward(new_model, support_set_inital, stage='support')
                # 计算grad
                names_weights_copy = self.get_inner_loop_parameter_dict(new_model.named_parameters())
                # create_graph,retain_graph的取值
                grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=False,
                                            retain_graph=False, allow_unused=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                                  in names_weights_copy.keys()}
                # 3加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
                new_model.load_state_dict(new_inner_dict)
                # 按理此处没有grad？
                new_model.zero_grad()
                del grads
                query_loss = self.meta_forward(new_model, query_set_inital, stage='query')
                task_query_loss.append(query_loss)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
            loss_epoch = loss_epoch + task_query_loss.item()
            """
            # ！！！（3）
            todo task_query_loss是由内部的new_model计算得到的，loss backward只会计算new_model网络的梯度，此时其初始值是不同于self.net的
            我们后期只需要他的梯度，不需要他的值，故而设计函数将梯度对应传回来即可
            torch1.5以下，不会监查原位操作的问题，但相应的其实其梯度计算错误。
            """
            task_query_loss.backward()
            for old, new in zip(self.net.named_parameters(), new_model.named_parameters()):
                # 返回一个tuple 【名称，参数tensor】
                old[1].grad = new[1].grad
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            end = time.time()
            if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                        batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch, task_query_loss.item(),
                        end - start))
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        return train_loss_epoch

    def meta_forward(self, model, data, stage):
        """
        loss 不在这里面求解
        """
        model.train()
        data_set = self.dataloader.rotate_shift_batch(data[0])
        # 将数据转成pytorch的tensor格式 将其转移到GPU上
        data_set = tuple([torch.Tensor(i) for i in data_set])
        data_set = tuple([i.cuda() for i in data_set])
        loss = torch.zeros(1).cuda()
        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = data_set
        set_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                       :-1], batch_pednum
        # todo forward需要与对应的参数结合起来  内外参数
        print('begin ' + stage)
        # todo iftest ?
        outputs = model.forward(set_forward, iftest=False)
        lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
        loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
        # 测试时此处应该batch——norm只有[1:8,:,:2] [7,258,2] outputs[:7,:,:]  lossmask[:7,:] 相应的num也需要变sum(sum[lossmask])
        # lossmask/loss_o size [19(time),258(batch_pednum)] ==> [8,258] (seq_length, num_Peds)
        loss = loss + torch.sum(loss_o * lossmask / num)
        return loss
