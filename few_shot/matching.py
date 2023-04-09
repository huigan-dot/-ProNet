import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss

from config import EPSILON
from few_shot.core import create_nshot_task_label
from few_shot.utils import pairwise_distances


def matching_net_episode(model: Module,
                         optimiser: Optimizer,
                         loss_fn: Loss,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         n_shot: int,
                         k_way: int,
                         q_queries: int,
                         distance: str,
                         fce: bool,
                         train: bool):
    """Performs a single training episode for a Matching Network.

    # Arguments
        model: Matching Network to be trained.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between support and query set samples
        fce: Whether or not to us fully conditional embeddings
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Matching Network on this task
        y_pred: Predicted class probabilities for the query set on this task

    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model.encoder(x)
    # print(x.shape)  #torch.Size([150, 2, 4800])
    # print(embeddings.shape)# torch.Size([150, 448])

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    # 可选择计算完整上下文嵌入 (FCE)。考虑到支持集，LSTM 将原始嵌入作为输入和输出修改后的嵌入。
    # Optionally apply full context embeddings
    if fce:
        # LSTM requires input of shape (seq_len, batch, input_size). `support` is of
        # shape (k_way * n_shot, embedding_dim) and we want the LSTM to treat the
        # support set as a sequence so add a single dimension to transform support set
        # to the shape (k_way * n_shot, 1, embedding_dim) and then remove the batch dimension
        # afterwards

        # Calculate the fully conditional embedding, g, for support set samples as described
        # in appendix A.2 of the paper. g takes the form of a bidirectional LSTM with a
        # skip connection from inputs to outputs
        # print(support.unsqueeze(1)) #torch.Size([8, 1, 448])
        print(support.shape) # torch.Size([8, 448])
        # x_emb.to(torch.float32)
        support, _, _ = model.g(support)#
        print(support.shape)
        support = support.squeeze(1)
        print(support.shape)

        # Calculate the fully conditional embedding, f, for the query set samples as described
        # in appendix A.1 of the paper.
        queries = model.f(support, queries)

    # Efficiently calculate distance between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, support, distance)
    print(distances.shape)# torch.Size([2, 8])

    # Calculate "attention" as softmax over support-query distances
    attention = (-distances).softmax(dim=1)# 注意力权重系数
    print(attention.shape)# torch.Size([2, 8])
    # Calculate predictions as in equation (1) from Matching Networks
    # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
    y_pred = matching_net_predictions(attention, n_shot, k_way, q_queries)

    # Calculated loss with negative log likelihood
    # Clip predictions for numerical stability
    # clamp函数用于约束返回值到A和B之间，若value小于min，则返回min；若value大于max，则返回max，起到上下截断的作用。
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    loss = loss_fn(clipped_y_pred.log(), y)
    # print(y)# tensor([0, 1], device='cuda:0') 因为n=2所以真实标签就是有两类
    # print(clipped_y_pred.log())
    if train:
        # Backpropagate gradients
        loss.backward()
        # I found training to be quite unstable so I clip the norm
        # of the gradient to be at most 1
        clip_grad_norm_(model.parameters(), 1)
        # Take gradient step
        optimiser.step()

    return loss, y_pred


def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.

    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.

    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
        n: Number of support set samples per class, n-shot
        k: Number of classes in the episode, k-way
        q: Number of query samples per-class

    # Returns
        y_pred: Predicted class probabilities
    """
    if attention.shape != (q * k, k * n):
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, k)

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    y = create_nshot_task_label(k, n).unsqueeze(-1)
    print(y)
    # scatter是一个非常实用的映射函数，其将一个源张量（source）中的值按照指定的轴方向（dim）和对应的位置关系（index）逐个填充到目标张量（target）中
    y_onehot = y_onehot.scatter(1, y, 1)# 将1映射到y_onehot上，dim=1表示逐列进行行填充，按照轴方向，y表示在target张量中需要填充的位置
    print(y_onehot)
    # torch.mm(a, b)是矩阵a和b矩阵相乘
    y_pred = torch.mm(attention, y_onehot.cuda())#.double()

    return y_pred