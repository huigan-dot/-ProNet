import torch


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    #sum1=y_pred.shape[0]
    #y_pred.argmax(dim=-1) 取出y_pred中每行最大数据的索引 最为模型预测的结果
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}
