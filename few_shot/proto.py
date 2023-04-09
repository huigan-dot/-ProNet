import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable

from few_shot.utils import pairwise_distances


def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples  以1-shot, 5-way, 1-qury
    embeddings = model(x)  # embeddings  torch.Size([10, 38400])
    # print("embeddings.shape:",embeddings.shape) [1250, 128]

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    # print('support.shape', y[:n_shot * k_way]) # [250, 128]
    # print('queries.shape', queries.shape) [1000, 128])
    prototypes = compute_prototypes(support, k_way, n_shot)
    # print(prototypes.shape)#  等于way的维度  以及网络输出嵌入
    # print(prototypes.shape) #torch.Size([5, 38400]) 将所有的同类的support 求和取平均
    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)  # 采用qury数据集实现模型参数的更新

    # Calculate log p_{phi} (y = k | x)   计算新样本在embedding space中到已知原型的距离，将之后通过softmax将距离转化为one hot 编码
    log_p_y = (-distances).log_softmax(dim=1)
    # print(log_p_y) # log_p_y是一个5-wayX5-way的矩阵 
    # print(y)#tensor([0, 1, 2, 3, 4], device='cuda:0')
    # print(log_p_y)
    loss = loss_fn(log_p_y, y)  # 把log_p_y的输出与Label对应的那个值拿出来，再去掉负号，再求均值  就是log_p_y对角线元素相加取反求均值
    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)
    del embeddings
    # print(y_pred.shape)# 行数等于q_shot*k_way  列数k_way
    if train:
        # Take gradient step
        # loss.requires_grad_(True)
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)  # 在n shot 的维度上取平均 数据是按照顺序的先是第一类
    return class_prototypes
