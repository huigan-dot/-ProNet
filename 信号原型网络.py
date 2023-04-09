#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# python 信号原型网络.py --save-dir Proto_CNN/
import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import Logger, AverageMeter
from torch import nn
from typing import Iterable, List
from few_shot.core import prepare_nshot_task
from few_shot.proto import proto_net_episode
from pytorchtools import EarlyStopping
from few_shot.metrics import categorical_accuracy
import matplotlib.pyplot as plt
import sys
import math
import argparse
import os.path as osp
import data_read

parser = argparse.ArgumentParser("proto_nets_Example")
parser.add_argument('--lr-model', type=float, default=0.0005, help="learning rate for model")  # 对于wifi数据集开始的学习率应该设置大点
parser.add_argument('--max-epoch', type=int, default=100)  # 100
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--model', type=str, default='ConvNet_2D')
parser.add_argument('--save-dir', type=str, default='mylog1')
args = parser.parse_args()
SNR = 60
file = "小样本/原型网络/"
file_data = "data/"


sys.stdout = Logger(file + osp.join(args.save_dir, 'WiFi2_ProtoNet_log.txt'))
if SNR < 50:
    dataset_name = None  # None 仅仅进行模型的测试
    dataset_eval_name = 'Eval_20way300shotV1_' + str(SNR) + 'dB.mat'
    dataset_test_name = "Test_20way300shotV1_" + str(SNR) + "dB.mat"
else:
    dataset_name = "Train_60way60shot.mat"
    dataset_eval_name = 'Eval_20way300shotV1.mat'
    dataset_test_name = "Test_20way300shotV1.mat"
path1 = osp.join(file + args.save_dir + "ProtoNet.pt")
patience = 35
n_train = 5  # shot 与测试集的任务要一致
k_train = 30  # 类别数 这里要设置大一点  way
q_train = 30  # query 也要设置大点，因为模型的反向传播主要靠这个
distance = 'l2'
episodes_per_epoch = 32  # 每次迭代beach 的大小
num_tasks = 1  # 一个beach里面选取的任务数
episodes_per_epoch_test = 300
n_test = n_train
k_test = 10
q_test = q_train


print('进行-' + str(n_train) + '-shot ' + str(k_train) + '-way 训练   ' + str(n_test) + '-shot ' + str(
    k_test) + '-way 验证' + "distance = " + distance)
print("超参数为：batch_size={}，lr_model={} earying_stop={}".format(episodes_per_epoch, args.lr_model, patience))


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.
        # Arguments
        #数据集：torch.utils.data.Dataset 的实例，从中抽取样本
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch  在一个时期内生成的任意数量的 n-shot 任务批次
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset  # 可迭代对象
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch  # 返回数据集的大小

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1
                '''           subset           alphabet  ...     id class_id
                   100    background  images_background  ...    100        5
                '''
                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
                # 创建字典：支持集对应的类别和样本 样本处先设置为空之后填充
                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    # 在对应类别随机选择样本
                    support = df[df['class_id'] == k].sample(self.n)  # 从一列/行数据里返回指定数量的随机样本
                    support_k[k] = support

                    for i, s in support.iterrows():  # support.iterrows() 遍历support中的每一行 返回索引和数据
                        batch.append(s['id'])  # 保存选取样本的表格ID

                for k in episode_classes:  # query数据集与support set不重合 选择q个
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])
            yield np.stack(batch)  # 将support set 和qury set的表格行数在垂直方向堆叠起来


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def get_few_shot_encoder(num_input_channels=2) -> nn.Module:
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(num_input_channels, 2 * 25, kernel_size=(1, 13), padding=(0, 6), bias=False),
            nn.BatchNorm2d(2 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),

            nn.Conv2d(2 * 25, 3 * 25, kernel_size=(1, 11), padding=(0, 5), bias=False),
            nn.BatchNorm2d(3 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 3)),

            nn.Conv2d(3 * 25, 4 * 25, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(4 * 25, 6 * 25, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(6 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(6 * 25, 8 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(8 * 25),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * 25, 8 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(8 * 25, 12 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv2d(12 * 25, 12 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(12 * 25, 12 * 25, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv2d(12 * 25, 12 * 25, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv2d(12 * 25, 12 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
    )


def main():
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if dataset_name is not None:
        # 训练集
        dataset = data_read.SignalDataset2D_fewshot(file_data + dataset_name)
        background_taskloader = DataLoader(
            dataset,
            batch_sampler=NShotTaskSampler(dataset, episodes_per_epoch, n_train, k_train, q_train, num_tasks=num_tasks),
            num_workers=0
        )
        # 验证集
        validate_set = data_read.SignalDataset2D_fewshot(file_data + dataset_eval_name)
        evaluation_taskloader = DataLoader(
            validate_set,
            batch_sampler=NShotTaskSampler(validate_set, episodes_per_epoch_test, n_test, k_test, q_test,
                                           num_tasks=num_tasks),
            num_workers=0
        )
        if args.model == "ConvNet_2D":
            model = get_few_shot_encoder()
        else:
            raise RuntimeError("模型类型错误")
        model = model.cuda()
        print(f'Training Prototypical network on {dataset_name}...')
        # optimiser = Adam(model.parameters(), lr=args.lr_model, weight_decay=5e-04)
        optimiser = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        loss_fn = torch.nn.NLLLoss().cuda()
        prepare_batch = prepare_nshot_task(n_train, k_train, q_train)
        fit_function_kwargs = {'n_shot': n_train, 'k_way': k_train, 'q_queries': q_train, 'train': True,
                               'distance': distance}

        fit_function_kwargs1 = {'n_shot': n_test, 'k_way': k_test, 'q_queries': q_test, 'train': False,
                                'distance': distance}

        prepare_batch1 = prepare_nshot_task(n_test, k_test, q_test)
        loss1 = []
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=20, eta_min=5e-7, last_epoch=-1)
        # scheduler = lr_scheduler.ExponentialLR(optimiser, 0.9, last_epoch=-1)
        acc2 = []
        loss1_evl = []
        acc2_evl = []
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=path1)
        for epoch in range(args.max_epoch):
            acc = []
            acc_evl = []
            loss2 = AverageMeter()
            loss2_evl = AverageMeter()
            for batch_index, batch in enumerate(background_taskloader):  # 一个beach 循环一次一共循环64个
                model.train()
                x, y = prepare_batch(batch)
                loss, y_pred = proto_net_episode(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
                loss2.update(loss.item(), y.size(0))
                acc1 = categorical_accuracy(y, y_pred)
                acc.append(acc1)
            with torch.no_grad():
                model.eval()
                for batch_index, batch in enumerate(evaluation_taskloader):
                    x, y = prepare_batch1(batch)
                    loss_evl, y_pred = proto_net_episode(model, optimiser, loss_fn, x, y, **fit_function_kwargs1)
                    loss2_evl.update(loss_evl.item(), y.size(0))
                    acc1_evl = categorical_accuracy(y, y_pred)
                    acc_evl.append(acc1_evl)
            loss1.append(loss2.avg)
            acc2.append(np.mean(acc))
            loss1_evl.append(loss2_evl.avg)
            acc2_evl.append(np.mean(acc_evl))
            # scheduler.step(np.mean(loss2_evl))
            scheduler.step()
            print("=====>Batch {}/{}\nLoss mean: {:.4f}\tAcc mean: {:.4f}\nEval loss: {:.4f}\tAcc-mean: {:.4f}".format(
                epoch,
                args.max_epoch, loss2.avg, np.mean(acc), loss2_evl.avg, np.mean(acc_evl)))
            early_stopping(loss2_evl.avg, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        np.save(osp.join(file + args.save_dir, "acc_train.npy"), acc2)
        np.save(osp.join(file + args.save_dir, "acc_eval.npy"), acc2_evl)
        np.save(osp.join(file + args.save_dir, "loss_train.npy"), loss1)
        np.save(osp.join(file + args.save_dir, "loss_eval.npy"), loss1_evl)
        # 绘制图像
        fig = plt.figure(figsize=(8, 6.5))
        # fig = plt.figure(1, dpi=120)
        plt.plot(loss1, linewidth=1.5, label="train_loss")
        plt.plot(loss1_evl, linewidth=1.5, label="eval_loss")
        plt.ylabel('Loss', fontsize=20, labelpad=4.5)
        plt.xlabel('Number of iterations', fontsize=20, labelpad=4.5)
        # plt.rcParams.update({'font.size': 15})
        plt.legend(frameon=False)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.gcf().subplots_adjust(bottom=0.16, left=0.15)
        plt.show()

        fig = plt.figure(figsize=(8, 6.5))
        plt.plot(acc2, linewidth=1.5, label="train_acc")
        plt.plot(acc2_evl, linewidth=1.5, label="eval_acc")
        plt.ylabel('Accuracy', fontsize=20, labelpad=4.5)
        plt.xlabel('Number of iterations', fontsize=20, labelpad=4.5)
        plt.legend(frameon=False)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.gcf().subplots_adjust(bottom=0.16, left=0.15)
        plt.show()
    else:
        print("##############仅仅进行测试%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        if args.model == "ConvNet_2D":
            model = get_few_shot_encoder()
        else:
            raise RuntimeError("模型类型错误")
        model = model.cuda()
        fit_function_kwargs1 = {'n_shot': n_test, 'k_way': k_test, 'q_queries': q_test, 'train': False,
                                'distance': distance}
        prepare_batch1 = prepare_nshot_task(n_test, k_test, q_test)
        optimiser = Adam(model.parameters(), lr=args.lr_model, weight_decay=5e-04)
        loss_fn = torch.nn.NLLLoss().cuda()

    model.load_state_dict(torch.load(path1))
    print("导入模型成功！！")
    loss2_evl = []
    acc_evl = []
    print(file_data + dataset_test_name)
    dataset_test = data_read.SignalDataset2D_fewshot(file_data + dataset_test_name)
    test_taskloader = DataLoader(
        dataset_test,
        batch_sampler=NShotTaskSampler(dataset_test, episodes_per_epoch_test, n_test, k_test, q_test,
                                       num_tasks=num_tasks),
        num_workers=0
    )

    with torch.no_grad():
        model.eval()
        for batch_index, batch in enumerate(test_taskloader):
            x, y = prepare_batch1(batch)  # 注意仅仅qury数据集存在标签
            loss_evl, y_pred = proto_net_episode(model, optimiser, loss_fn, x, y, **fit_function_kwargs1)
            loss2_evl.append(loss_evl.item())
            acc1_evl = categorical_accuracy(y, y_pred)
            acc_evl.append(acc1_evl)
            # if batch_index == 0:
    print("平均准确率为:{:0.4f} +/- {:0.4f}".format(np.mean(acc_evl), 1.96 * np.std(acc_evl, ddof=1) / math.sqrt(
        episodes_per_epoch_test)))
    print("方差为" + str(np.var(acc_evl)))
    print("标准差为" + str(1.96 * np.std(acc_evl, ddof=1) / math.sqrt(episodes_per_epoch_test)))
    acc_evl[:] = [abs(x - np.mean(acc_evl)) for x in acc_evl]
    print("最大差值为" + str(np.max(acc_evl)))
    xdata = x.cpu().numpy()
    y_pred = np.argmax(y_pred.cpu().numpy(), axis=1)
    y = y.cpu().numpy()
    np.save(osp.join(file + args.save_dir, str(SNR) + "xdata.npy"), xdata)
    np.save(osp.join(file + args.save_dir, str(SNR) + "pred_label.npy"), y_pred)
    np.save(osp.join(file + args.save_dir, str(SNR) + "real_label.npy"), y)
    import Confusion_matrix
    Confusion_matrix.plot_confusion_matrix(y, y_pred, file + args.save_dir,
                                           normalize=True, cmap=plt.cm.Blues)
    if k_test == 10:
        config = {
            "font.family": 'Times New Roman',
            "font.size": 9.5,
            "mathtext.fontset": 'stix',
        }
    else:
        config = {
            "font.family": 'Times New Roman',
            "font.size": 12,
            "mathtext.fontset": 'stix',
        }
    plt.rcParams.update(config)
    fig = plt.figure(3)
    for i in range(k_test):
        ax = fig.add_subplot(k_test + 1, 1, i + 1)
        ax.plot(xdata[k_test * i, 1, 0, :], label="WiFi_" + str(i))
        plt.legend(loc="upper right", frameon=False, fontsize=10)
        plt.xticks()
        plt.yticks()
        plt.xlim(0, 2220)
    ax = fig.add_subplot(k_test + 1, 1, k_test + 1)
    # ax.set_title("WiFi_pred-" + str(y_pred[-1]) + "_real-" + str(y[-1]), fontsize=14)
    ax.plot(xdata[-1, 1, 0, :])
    plt.xlim(0, 2220)
    plt.xticks()
    plt.yticks()
    plt.show()


if __name__ == '__main__':
    main()
    print('进行-' + str(n_train) + '-shot ' + str(k_train) + '-way 训练   ' + str(n_test) + '-shot ' + str(
        k_test) + '-way 验证' + "distance = " + distance)
    print(args.model)
