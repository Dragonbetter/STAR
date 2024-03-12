# 多种不同的尝试
# MAML+mixup
# MVDG代码 只适用于原始的star
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





    # 添加注入的代码 MAML + mixup
    def train_meta_mixup_epoch(self, epoch):
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
                # support loss 此处输入的mean/var-list应该为空 【】
                # 加3000M
                support_loss, mean_support, var_support = self.meta_mixup_forward(new_model, support_set_inital,
                                                                                  stage='support', mean_list=[],
                                                                                  var_list=[])
                # 计算grad
                names_weights_copy = self.get_inner_loop_parameter_dict(new_model.named_parameters())
                # create_graph,retain_graph的取值 加1000M retain-graph true14469 False也是一样
                grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True,
                                            retain_graph=False, allow_unused=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                                  in names_weights_copy.keys()}
                # 3加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
                new_model.load_state_dict(new_inner_dict)
                # 按理此处没有grad？
                new_model.zero_grad()
                del grads
                # 加2000M
                query_loss, _, _ = self.meta_mixup_forward(new_model, query_set_inital, stage='query',
                                                           mean_list=mean_support, var_list=var_support)
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
            # task_query_loss.backward() 需要注意此处task_loss的backward是否是基于new_model 需要注意 ！！
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

    def train_meta_mixup_epoch_withloss(self, epoch):
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
            # todo 需要注意之前MLDG框架中忘记添加support的loss了；最小化元训练和元测试领域的损失；传统的优化器会很高兴地进行非对称调整，
            #  专注于哪个领域更容易最小化。Eq. 7中第三项提供的正则化倾向于更新权重，其中两个优化曲面在梯度上一致。
            #  它通过寻找一条最小化路径来减少对单个域的过拟合，其中两个子问题在路径上所有点的方向一致。
            task_support_loss = []
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
                # support loss 此处输入的mean/var-list应该为空 【】
                support_loss, mean_support, var_support = self.meta_mixup_forward(new_model, support_set_inital,
                                                                                  stage='support', mean_list=[],
                                                                                  var_list=[])
                # todo 同一个support-loss需要二次损失反向传播，需要注意该损失是直接.clone取值还是需要真正的反向回传
                task_support_loss.append(support_loss.clone())
                # 计算grad
                names_weights_copy = self.get_inner_loop_parameter_dict(new_model.named_parameters())
                # create_graph,retain_graph的取值
                grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True,
                                            retain_graph=True, allow_unused=True)
                inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
                new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                                  in names_weights_copy.keys()}
                # 3加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
                new_model.load_state_dict(new_inner_dict)
                # 按理此处没有grad？
                new_model.zero_grad()
                del grads
                query_loss, _, _ = self.meta_mixup_forward(new_model, query_set_inital, stage='query',
                                                           mean_list=mean_support, var_list=var_support)
                task_query_loss.append(query_loss)
            task_query_loss = torch.mean(torch.stack(task_query_loss))
            task_support_loss = torch.mean(torch.stack(task_support_loss))
            print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()) + 'task_support_loss' +
                  str(task_support_loss.cpu().detach().numpy()))
            task_loss = task_support_loss + task_query_loss
            loss_epoch = loss_epoch + task_query_loss.item() + task_support_loss.item()
            """
            # ！！！（3）
            todo task_query_loss是由内部的new_model计算得到的，loss backward只会计算new_model网络的梯度，此时其初始值是不同于self.net的
            我们后期只需要他的梯度，不需要他的值，故而设计函数将梯度对应传回来即可
            torch1.5以下，不会监查原位操作的问题，但相应的其实其梯度计算错误。
            """
            # task_query_loss.backward() 需要注意此处task_loss的backward是否是基于new_model 需要注意 ！！
            task_loss.backward()
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
                        batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch, task_loss.item(),
                        end - start))
        train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
        return train_loss_epoch

# MVDG代码 只适用于原始的star
"""
 def train_MVDG_epoch_star(self, epoch):

    结合MVDG框架重写该部分代码；
    每次有三个轨迹，每个轨迹内部有4个task；相应的需要注意此处3个轨迹是添加训练时间还是原始的batch数分3块；两种都试一下
    内外循环 reptile
    此处的写法是将原有的batch在分成3个部分
    ==》分析数据集本身 发现eth确实与其他四个相差较大，行人更少，速度更快；
    ==》实验思路：包括调节学习率，更改数据生成。
    todo 现有的结果是hotel0.25-0.22-0.18已经有显著提升，zara1,zara2,univ仍然在下降中，eth效果混乱，没有学到泛化性，对于源域过拟合了
          后续需要针对eth多次实验，观测器参数是否进入极小值点了，以及相应的以eth为测试域，或则说用另外其他四个做训练会有什么区别

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
            print('begin' + str(epoch) + 'batch_traj' + str(batch_id) + 'optim_traj' + str(traj_id))
            fast_model = copy.deepcopy(self.net).train().cuda()
            fast_opts = torch.optim.Adam(fast_model.parameters(), lr=self.args.inner_learning_rate,
                                         betas=(0.9, 0.999),
                                         weight_decay=5e-4)
            # 每个task内包含一个support和query
            traj_query_loss = []
            for task_id, task_data in enumerate(traj_data):
                support_set_inital = task_data[1]
                query_set_inital = task_data[0]
                support_loss, mean_support, var_support = self.meta_mixup_forward(fast_model, support_set_inital,
                                                                                  stage='support', mean_list=[],
                                                                                  var_list=[], ifmixup=False)
                fast_opts.zero_grad()
                support_loss.backward()
                fast_opts.step()
                query_loss, _, _ = self.meta_mixup_forward(fast_model, query_set_inital, stage='query',
                                                           mean_list=mean_support, var_list=var_support,
                                                           ifmixup=False)

                traj_query_loss.append(query_loss)
                fast_opts.zero_grad()
                query_loss.backward()
                fast_opts.step()
            task_query_loss.append(torch.mean(torch.stack(traj_query_loss)))
            parameters = dict(fast_model.named_parameters())
            fast_models.append(parameters)
        task_query_loss = torch.mean(torch.stack(task_query_loss))
        print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
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
"""
# train_MLDG_mixup_epoch 有新的结合不同模型选项的更替
"""
def train_MLDG_mixup_epoch_1(self, epoch):

    集合MLDG框架重写该代码--注意此处要同时加上meta-train和meta-test的loss，new-model在meta-test时加入

    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch,query_loss_epoch,support_loss_epoch = 0,0,0
    for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        print('begin' + str(epoch) + str(batch_task_id))
        start = time.time()
        self.net.zero_grad()
        task_support_loss = []

        注意此处有两种写法--此处为写法1 -- 参考M3L
        写法1：串行：首先依次计算4个meta-train的loss，而后取平均作为train-loss；
        继而在单一meta-test上进行计算的meta-test的loss；后运用test-loss以及train-loss共同更新初始参数
        写法2：并行：按循环，首先第一个域，计算meta-train-loss，而后建立新模型在meta-test上计算test-loss；
        依次循环进行，得到4个train-loss和test-loss；而后加和取平均，用以更新总损失。

        for task_id, task_batch_data in enumerate(batch_task_data):
            # 每次循环添加2000M
            support_set_inital = task_batch_data[1]
            # support loss 此处输入的mean/var-list应该为空 【】
            support_loss, mean_support, var_support = self.meta_mixup_forward(self.net, support_set_inital,
                                                                              stage='support', mean_list=[],
                                                                              var_list=[])
            task_support_loss.append(support_loss)
        # 对于源域中的四个meta-train，利用原始模型计算出对应的四个loss，而后平均作为meta-train-loss
        task_support_loss = torch.mean(torch.stack(task_support_loss))
        # --9061M
        names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
        # grads caulcate 添加5696M
        grads = torch.autograd.grad(task_support_loss, names_weights_copy.values(), create_graph=True,
                                    retain_graph=True, allow_unused=False)
        new_model = copy.deepcopy(self.net).train().cuda()
        inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
        new_inner_dict = {key: names_weights_copy[key] - self.args.inner_learning_rate * inner_dict_grads[key] for
                          key in names_weights_copy.keys()}
        new_model.load_state_dict(new_inner_dict)
        new_model.zero_grad()
        self.net.zero_grad()
        # del grads 只会减少 grads 变量的引用计数，如果其他地方仍然存在对 grads 的引用，那么内存可能不会立即释放。确保 grads 变量没有其他引用，并且在删除之后没有进一步使用。
        del grads, inner_dict_grads, new_inner_dict
        query_set_inital = batch_task_data[0][0]
        # 2000M
        query_loss, _, _ = self.meta_mixup_forward(new_model, query_set_inital, stage='query',
                                                   mean_list=mean_support, var_list=var_support)
        task_query_loss = query_loss
        task_loss = task_support_loss + task_query_loss
        print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()) + 'task_support_loss' +
              str(task_support_loss.cpu().detach().numpy()))
        query_loss_epoch += task_query_loss.item()
        support_loss_epoch += task_support_loss.item()
        loss_epoch += task_query_loss.item() + task_support_loss.item()
        self.optimizer.zero_grad()
        task_loss.backward()
        # 分析task-loss 计算的梯度时self.net（support-loss）和new_model（query-loss）都更新了grads
        # 此处需要将new——model计算得到的梯度叠加给self.net
        for old, new in zip(self.net.named_parameters(), new_model.named_parameters()):
            # 返回一个tuple 【名称，参数tensor】
            old[1].grad += new[1].grad
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        task_loss_info = task_loss.detach().clone()
        del task_support_loss, task_query_loss, task_loss
        end = time.time()
        if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
            print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch, task_loss_info.item(),
                    end - start))
    train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    train_support_loss_epoch = support_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    train_query_loss_epoch = query_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    print('epoch {} ,loss = {:.5f},support_loss = {:.5f},query_loss = {:.5f}'.format(epoch,train_loss_epoch,
                                                                                     support_loss_epoch,query_loss_epoch))
    return train_loss_epoch,train_support_loss_epoch,train_query_loss_epoch
def train_MLDG_mixup_epoch_2(self, epoch):

    结合MLDG框架重写该部分代码 -- new-ModelStrategy 在测试时加入，对应的并行方法

    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch, support_loss_epoch ,query_loss_epoch = 0,0,0
    start = time.time()
    for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        print('begin' + str(epoch) + str(batch_task_id))
        self.net.zero_grad()
        task_support_loss = []
        task_query_loss = []
        for task_id, task_batch_data in enumerate(batch_task_data):
            support_set_inital = task_batch_data[1]
            query_set_inital = task_batch_data[0]
            support_loss, mean_support, var_support = self.meta_mixup_forward(self.net, support_set_inital,
                                                                              stage='support', mean_list=[],
                                                                              var_list=[])
            task_support_loss.append(support_loss)
            # 浅拷贝 names_weights_copy的值随着self.net改变
            names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True,
                                        retain_graph=True, allow_unused=False)
            new_model = copy.deepcopy(self.net).train().cuda()
            inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
            new_inner_dict = {key: names_weights_copy[key] - self.args.inner_learning_rate * inner_dict_grads[key]
                              for key
                              in names_weights_copy.keys()}
            new_model.load_state_dict(new_inner_dict)
            new_model.zero_grad()
            self.net.zero_grad()
            new_model_weights_copy = self.get_inner_loop_parameter_dict(new_model.named_parameters())
            del grads, inner_dict_grads, new_inner_dict
            query_loss, _, _ = self.meta_mixup_forward(new_model, query_set_inital, stage='query',
                                                       mean_list=mean_support, var_list=var_support)
            task_query_loss.append(query_loss)
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
        print('train-{}/{},epoch{},task_support_loss = {:.5f},task_query_loss = {:.5f}'.format(batch_task_id,
                                                                                               len(self.dataloader.train_batch_MLDG_task),
                                                                                               epoch,
                                                                                               task_support_loss,
                                                                                               task_query_loss))
    train_support_loss_epoch = support_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    train_query_loss_epoch = query_loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    end = time.time()
    print('epoch{},loss = {:.5f} support_loss = {:.5f},query_loss = {:.5f},time/epoch = {:.5f}'.format(epoch,train_loss_epoch,
                                                                                                     train_support_loss_epoch,
                                                                                                     train_query_loss_epoch,
                                                                                                     (end - start)))
    return train_loss_epoch,train_support_loss_epoch,train_query_loss_epoch
"""
# meta false
"""
    def train_meta_epoch_false(self, epoch):

    结合Meta框架重写该部分代码；
    内外循环

    # 第一步依据完整数据拆分出batch list
    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch = 0
    for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
        #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
        # 针对batch-task-data中的4个task进行处理 list
        print('begin' + str(batch_task_id))
        start = time.time()
        self.net.zero_grad()
        # net_initial_dict 包含值 device
        net_initial_dict = self.net.state_dict()
        task_query_loss = []
        for task_id, task_batch_data in enumerate(batch_task_data):
            # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
            print('begin' + str(batch_task_id) + '---' + str(task_id))
            self.trajectory_pred.load_state_dict(net_initial_dict)
            self.trajectory_pred.zero_grad()
            # 准备数据
            support_set_inital = task_batch_data[0]
            query_set_inital = task_batch_data[1]
            support_loss = self.meta_forward(self.trajectory_pred, support_set_inital, stage='support')
            # 依据support loss更新参数
            # inner_dict = self.get_inner_loop_parameter_dict(inner_dict)
            names_weights_copy = self.get_inner_loop_parameter_dict(self.trajectory_pred.named_parameters())
            # names_weights_copy 包含值 device 还有对应的requires_grad
            grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=False,
                                        allow_unused=True)
            inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
            new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                              in names_weights_copy.keys()}
            # 加载内循环更新完的参数 以新参数计算query的loss
            self.trajectory_pred.load_state_dict(new_inner_dict)
            self.trajectory_pred.zero_grad()
            query_loss = self.meta_forward(self.trajectory_pred, query_set_inital, stage='query')
            task_query_loss.append(query_loss)
        task_query_loss = torch.mean(torch.stack(task_query_loss))
        print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
        loss_epoch = loss_epoch + task_query_loss.item()

        # todo task_query_loss是由内部的trajectory计算得到的，但是我们此处需要用其去更新外围的参数，
        此处直接的loss backward会使得其同时计算对于trajectory和net的网络的梯度，故而相应的使用已经更新过的参数去更新参数，
        会导致原位操作的问题；而相应的分析，此处我们只需要更新self.net的参数，故而可以进行指定（torch1.8以上可以）
        torch1.5以下，不会监查原位操作的问题，但相应的其实其梯度计算错误。
        ==>> 事实是一直没更新，backward与step未正确的更新参数

        task_query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()

        end = time.time()
        if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
            print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch, task_query_loss.item(),
                    end - start))
    train_loss_epoch = loss_epoch / len(self.dataloader.train_batch_MLDG_task)
    return train_loss_epoch
"""
# multi gpus
"""
def get_params(self,params,device):
    "将模型的参数进行复制"

    new_params = {name:param.to(device=device) for name,param in params.items()}
    for name,param in new_params.items():
        param.requires_grad_()
    return new_params

def all_reduce(self,data):
    for i in range(1,len(data)):
        data[0][:] +=data[i].to(data[0].device)
    for i in range(1,len(data)):
        data[i][:] = data[0].to(data[i].device)

def train_meta_epoch_multi_gpus(self,epoch):
    # 第一步依据完整数据拆分出batch list
    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch = 0
    gpus_nums = torch.cuda.device_count()
    gpus = [i for i in range(gpus_nums)]
    for batch_task_id,batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
        #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
        # 针对batch-task-data中的4个task进行处理 list
        print('begin' +str(epoch)+str(batch_task_id))
        start = time.time()
        self.net.zero_grad()
        # !!!!(1)注意的是 state——dict是浅拷贝，即net-initial-dict改变的话，那么当你修改param，相应地也会修改model的参数。
        # model这个对象实际上是指向各个参数矩阵的，而浅拷贝只会拷贝最外层的这些“指针；
        # from copy import deepcopy  best_state = copy.deepcopy(ModelStrategy.state_dict()) 深拷贝 互不影响
        self.task_weight = [copy.deepcopy(self.net.state_dict()) for i in range(gpus_nums)]
        self.task_query_loss =[0,0,0,0]
        # 此处可以加速 串行改成四卡并行  8s--2s
        self.train_batch(batch_task_data, self.task_weight, gpus)
        task_query_loss = torch.mean(torch.stack(self.task_query_loss))
        print('task_query_loss:'+str(task_query_loss.cpu().detach().numpy()))
        loss_epoch =loss_epoch + task_query_loss.item()
        for old,new in zip(self.net.named_parameters(),task_weight[0]):
            # 返回一个tuple 【名称，参数tensor】 此处注意是否要除以4
            old[1].grad = task_weight[0][new].to(self.device) + task_weight[1][new].to(self.device) +task_weight[2][new].to(self.device)+task_weight[3][new].to(self.device)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        end = time.time()
        if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
            print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch,task_query_loss.item(),end - start))
    train_loss_epoch = loss_epoch/len(self.dataloader.train_batch_MLDG_task)
    return train_loss_epoch

def meta_forward_multi_gpus(self,ModelStrategy,data,stage,device,optim=None):
    ModelStrategy.train()
    if optim is not None:
        optim.zero_grad()
    data_set = self.dataloader.rotate_shift_batch(data[0])
    # 将数据转成pytorch的tensor格式 将其转移到GPU上
    data_set = tuple([torch.Tensor(i) for i in data_set])
    data_set = tuple([i.cuda(device) for i in data_set])
    loss = torch.zeros(1).cuda(device)
    batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = data_set
    set_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum
    # todo forward需要与对应的参数结合起来  内外参数
    # print('begin '+stage)
    # todo iftest ?
    outputs = ModelStrategy.forward(set_forward, iftest=False,device=device)
    lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:],using_cuda=self.args.using_cuda,device=device)
    loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
    loss = loss + torch.sum(loss_o * lossmask / num)
    # loss.backward(create_graph=create_graph,retain_graph=True)
    if optim is not None:
        optim.step()

    if stage == 'support':
        names_weights_copy = self.get_inner_loop_parameter_dict(ModelStrategy.named_parameters(),device=device)
        # create_graph,retain_graph的取值
        grads = torch.autograd.grad(loss, names_weights_copy.values(), create_graph=False,
                                    retain_graph=False, allow_unused=True)
        inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
        new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key in
                          names_weights_copy.keys()}
        del grads
    elif stage == 'query':
        loss.backward()
        inner_dict_grads = {name:param.grad for name,param in ModelStrategy.named_parameters()}
        new_inner_dict = ModelStrategy.state_dict()
    return new_inner_dict,inner_dict_grads ,loss

def train_batch(self,batch_data,params,device):
    "输入数据 四个GPU上并行跑 返回参数以及loss"
    # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算

    # 1 !!!! (2)每次都从1开始 清零 初始化一个self.new——model的话会导致反复更新累加 导致原位操作 7-9
    new_model = STAR(self.args).cuda(device)
    # 2
    params = self.get_params(params,device)
    new_model.load_state_dict(params)
    new_model.zero_grad()
    # 准备数据
    support_set_inital = batch_data[1]
    query_set_inital = batch_data[0]
    # todo forward需要与对应的参数结合起来  内外参数
    support_dict,support_grads,support_loss = self.meta_forward_multi_gpus(new_model, support_set_inital, stage='support',device=device)
    # 3加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
    new_model.load_state_dict(support_dict)
    # 按理此处没有grad？
    new_model.zero_grad()
    query_dict,query_grads,query_loss = self.meta_forward_multi_gpus(new_model, query_set_inital, stage='query',device=device)
    query_grads = self.get_params(query_grads,device=device)
    return query_grads,query_loss

"""
# ANOTHER MISTAKE
"""
舍弃错误代码
def train_meta_epoch_v4(self,epoch,first_order=False):
    # 第一步依据完整数据拆分出batch list
    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch = 0
    for batch_task_id,batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
        #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
        # 针对batch-task-data中的4个task进行处理 list
        print('begin' + str(batch_task_id))
        start = time.time()
        self.net.zero_grad()
        task_query_loss = []
        for task_id,task_batch_data in enumerate(batch_task_data):
            # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
            print('begin'+str(batch_task_id)+'---'+str(task_id))
            new_model = STAR(self.args).cuda()
            # todo debug
            new_model.copy(self.net, same_var=True)
            # 准备数据
            query_set_inital = task_batch_data[0]
            support_set_inital = task_batch_data[1]
            support_loss = self.meta_forward_v2(new_model, support_set_inital, stage='support')
            # 计算第一次的梯度
            new_model.zero_grad()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            grads = torch.autograd.grad(support_loss,names_weights_copy,create_graph=True,retain_graph=True)
            inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
            new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                              in names_weights_copy.keys()}
            # todo 加载内循环更新完的参数 此处更新参数 从而更改version 以新参数计算query的loss
            for name,param in new_model.named_parameters():
                if name in new_inner_dict:
                    new_model.set_param(new_model,name,new_inner_dict[name])
                else:
                    new_model.set_param(new_model,name,param)
            del grads
            query_loss = self.meta_forward_v2(new_model,query_set_inital,stage='query')
            query_loss.backward(create_graph=False,retain_graph=True)
            task_query_loss.append(query_loss)
        task_query_loss = torch.mean(torch.stack(task_query_loss))
        print('task_query_loss:'+str(task_query_loss.cpu().detach().numpy()))
        loss_epoch =loss_epoch + task_query_loss.item()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # new_model.zero_grad()
        end = time.time()
        if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
            print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch,task_query_loss.item(),end - start))
    train_loss_epoch = loss_epoch/len(self.dataloader.train_batch_MLDG_task)
    return train_loss_epoch

def train_meta_epoch_v3(self,epoch,first_order=False):
    # 第一步依据完整数据拆分出batch list 结合Meta框架重写该部分代码；
    #         内外循环
    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch = 0
    for batch_task_id,batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        # todo 明晰参数复制的过程，以及初步更新和二次更新的不同点，
        #  相应的support loss计算 query loss计算以及二次更新的结果 以及对应的后续将其函数化
        # 针对batch-task-data中的4个task进行处理 list
        print('begin' + str(batch_task_id))
        start = time.time()
        self.net.zero_grad()
        task_query_loss = []
        for task_id,task_batch_data in enumerate(batch_task_data):
            # 复制原始net的参数，并加载到对应的模型中，后续的net用这个去计算
            print('begin'+str(batch_task_id)+'---'+str(task_id))
            new_model = STAR(self.args).cuda()
            new_model.copy(self.net,same_var=True)
            # 准备数据
            query_set_inital = task_batch_data[0]
            support_set_inital = task_batch_data[1]
            support_loss = self.meta_forward(new_model, support_set_inital, stage='support', create_graph=not first_order)
            for name,param in new_model.named_params():
                grad = param.grad
                if first_order:
                    grad = V(grad.detach().data)
                new_model.set_params(name,param-self.task_learning_rate*grad)

                # new_model.zero_grad()
            query_loss = self.meta_forward(new_model,query_set_inital,stage='query',create_graph=False)
            task_query_loss.append(query_loss)
        task_query_loss = torch.mean(torch.stack(task_query_loss))
        print('task_query_loss:'+str(task_query_loss.cpu().detach().numpy()))
        loss_epoch =loss_epoch + task_query_loss.item()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # new_model.zero_grad()
        end = time.time()
        if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
            print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                    batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch,task_query_loss.item(),end - start))
    train_loss_epoch = loss_epoch/len(self.dataloader.train_batch_MLDG_task)
    return train_loss_epoch

def meta_forward_v1(self,ModelStrategy,data,stage,create_graph=False,optim=None):
    ModelStrategy.train()
    if optim is not None:
        optim.zero_grad()
    data_set = self.dataloader.rotate_shift_batch(data[0])
    # 将数据转成pytorch的tensor格式 将其转移到GPU上
    data_set = tuple([torch.Tensor(i) for i in data_set])
    data_set = tuple([i.cuda() for i in data_set])
    loss = torch.zeros(1).cuda()
    batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = data_set
    set_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum
    # todo forward需要与对应的参数结合起来  内外参数
    print('begin '+stage)
    # todo iftest ?
    outputs = ModelStrategy.forward(set_forward, iftest=False)
    lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:],using_cuda=self.args.using_cuda)
    loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)
    loss = loss + torch.sum(loss_o * lossmask / num)
    loss.backward(create_graph=create_graph,retain_graph=True)
    if optim is not None:
        optim.step()
    return loss

def train_meta_epoch_v2(self,epoch):
    self.dataloader.reset_batch_pointer(set='train', valid=False)
    loss_epoch =[]
    for batch_task_id, batch_task_data in enumerate(self.dataloader.train_batch_MLDG_task):
        print('begin' + str(batch_task_id))
        start = time.time()
        self.net.zero_grad()
        net_initial_dict = copy.deepcopy(self.net.state_dict())
        task_query_loss =[]
        for task_id, task_batch_data in enumerate(batch_task_data):
            print('begin' + str(batch_task_id) + '---' + str(task_id))
            self.net.load_state_dict(net_initial_dict)
            # 准备数据 support 不全
            # support_set_inital = task_batch_data[0]
            support_set_inital = task_batch_data[1]
            # 数据旋转 基于观测帧归一化 只取batch-data 忽略 batch-id
            support_set = self.dataloader.rotate_shift_batch(support_set_inital[0])
            # 将数据转成pytorch的tensor格式 将其转移到GPU上
            support_set = tuple([torch.Tensor(i) for i in support_set])
            support_set = tuple([i.cuda() for i in support_set])
            # 准备数据  query 有完整的格式
            query_set_inital = task_batch_data[0]
            query_set = self.dataloader.rotate_shift_batch(query_set_inital[0])
            # 将数据转成pytorch的tensor格式 将其转移到GPU上
            query_set = tuple([torch.Tensor(i) for i in query_set])
            query_set = tuple([i.cuda() for i in query_set])
            support_loss = torch.zeros(1).cuda()
            support_batch_abs, support_batch_norm, support_shift_value, support_seq_list, support_nei_list, \
            support_nei_num, support_batch_pednum = support_set
            support_set_forward = support_batch_abs[:-1], support_batch_norm[:-1], support_shift_value[
                                                                                   :-1], support_seq_list[:-1], \
                                  support_nei_list[:-1], support_nei_num[:-1], support_batch_pednum
            # todo forward需要与对应的参数结合起来  内外参数
            print('begin support')
            # todo 原来的模型参数是直接赋值params的，那么相应的赋值不会改变其对应的net本身的参数结构 内循环情况下version不变 故而无inplace operation
            support_outputs = self.net.forward(support_set_forward, iftest=False)
            support_lossmask, num = getLossMask(support_outputs, support_seq_list[0], support_seq_list[1:],
                                                using_cuda=self.args.using_cuda)
            support_loss_o = torch.sum(self.criterion(support_outputs, support_batch_norm[1:, :, :2]), dim=2)
            support_loss = support_loss + torch.sum(support_loss_o * support_lossmask / num)
            # 依据support loss更新参数
            # inner_dict = self.get_inner_loop_parameter_dict(inner_dict)
            names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            # names_weights_copy 包含值 device 还有对应的requires_grad
            grads = torch.autograd.grad(support_loss, names_weights_copy.values(),
                                        create_graph=self.args.second_order and
                                                     epoch > self.args.first_order_to_second_order_epoch,
                                        allow_unused=True)
            inner_dict_grads = dict(zip(names_weights_copy.keys(), grads))
            new_inner_dict = {key: names_weights_copy[key] - self.task_learning_rate * inner_dict_grads[key] for key
                              in names_weights_copy.keys()}
            # 加载内循环更新完的参数 以新参数计算query的loss
            self.net.load_state_dict(new_inner_dict)
            print('begin query')
            query_loss = torch.zeros(1).cuda()
            query_batch_abs, query_batch_norm, query_shift_value, query_seq_list, query_nei_list, \
            query_nei_num, query_batch_pednum = query_set
            query_set_forward = query_batch_abs[:-1], query_batch_norm[:-1], query_shift_value[:-1], query_seq_list[
                                                                                                     :-1], \
                                query_nei_list[:-1], query_nei_num[:-1], query_batch_pednum
            query_outputs = self.net.forward(query_set_forward, iftest=False)
            query_lossmask, num = getLossMask(query_outputs, query_seq_list[0], query_seq_list[1:],
                                              using_cuda=self.args.using_cuda)
            query_loss_o = torch.sum(self.criterion(query_outputs, query_batch_norm[1:, :, :2]), dim=2)
            query_loss = query_loss + torch.sum(query_loss_o * query_lossmask / num)
            task_query_loss.append(query_loss)
            # 四次的初始参数一致，需要的是各自更新计算后的loss，参数可以摒弃
            self.net.zero_grad()
            #
            self.net.load_state_dict(net_initial_dict)
        task_query_loss = torch.mean(torch.stack(task_query_loss))
        print('task_query_loss:' + str(task_query_loss.cpu().detach().numpy()))
        loss_epoch.append(task_query_loss)
        self.net.zero_grad()
        origin_name_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
        # 原位操作问题 ！！
        task_query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        test_name_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
        end = time.time()
        if batch_task_id % self.args.show_step == 0 and self.args.ifshow_detail:
            print(
            'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(
                batch_task_id, len(self.dataloader.train_batch_MLDG_task), epoch, task_query_loss.item(), end - start))
    train_loss_epoch = torch.mean(torch.stack(loss_epoch))
    return train_loss_epoch

"""
# TTA 测试时进行适应 不需要了
"""
def test_meta(self):
    print('Testing begin')
    self.load_model(stage=self.args.stage)
    print('Testing fine_tuning')
    self.best_fde, self.best_ade, self.best_epoch = 100, 100, -1
    for epoch in range(0, self.args.fine_tuning_nums_epoch):
        fine_tuning_loss = self.test_meta_once().cpu().detach().numpy()
        error, final_error = self.test_epoch()
        if final_error < self.best_fde:
            self.best_ade = error
            self.best_epoch = epoch
            self.best_fde = final_error
            self.save_meta_model(epoch)
        else:
            self.best_ade = self.best_ade
            self.best_epoch = self.best_epoch
            self.best_fde = self.best_fde
        print('epoch' + str(epoch) + 'loss: ' + str(fine_tuning_loss) + 'ADE:' + str(error) + 'FDE:' + str(
            final_error))
        print(
            'best epoch' + str(self.best_epoch) + 'Best_ADE' + str(self.best_ade) + 'Best_FDE' + str(self.best_fde))
    self.load_meta_model(self.best_epoch)
    self.net.eval()
    print('Testing eval')
    test_error, test_final_error = self.test_epoch()
    print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set,
                                                                          self.args.load_model,
                                                                          test_error, test_final_error))

def test_meta_once(self):
    self.dataloader.reset_batch_pointer(set='test')
    print('begin test fine-tuning')
    loss_epoch = 0
    for batch in tqdm(range(self.dataloader.test_batchnums)):
        # 与train相同的batch处理步骤
        inputs, batch_id = self.dataloader.get_test_batch(batch)
        inputs = tuple([torch.Tensor(i) for i in inputs])
        loss = torch.zeros(1).cuda()
        if self.args.using_cuda:
            inputs = tuple([i.cuda() for i in inputs])
        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
        inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], \
                         nei_list[:-1], nei_num[:-1], batch_pednum
        # 利用test的数据
        self.net.zero_grad()
        outputs = self.net.forward(inputs_forward, iftest=True)
        lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
        new_lossmask = lossmask[3:7, :]
        new_num = sum(sum(new_lossmask))
        loss_o = torch.sum(self.criterion(outputs[3:7, :, :], batch_norm[4:8, :, :2]), dim=2)
        loss = loss + torch.sum(loss_o * new_lossmask / new_num)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
        self.optimizer.step()
        loss_epoch = loss_epoch + loss
    return loss_epoch

def save_meta_model(self, epoch):
# 保存模型的代码与maml的代码框架合计
model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_fine_tuning_' + \
             str(epoch) + '.tar'
torch.save({
    'epoch': epoch,
    'state_dict': self.net.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict()
}, model_path)

def load_meta_model(self, best_epoch):
if self.args.load_model is not None:
    self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_fine_tuning_' + \
                                str(best_epoch) + '.tar'
    print(self.args.model_save_path)
    if os.path.isfile(self.args.model_save_path):
        print('Loading fine-tuning checkpoint')
        checkpoint = torch.load(self.args.model_save_path)
        model_epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at fine-tuning epoch', model_epoch)

"""