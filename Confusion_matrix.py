from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
from matplotlib import rcParams


def plot_confusion_matrix(true_data, pre_data, path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(8, 6.5))  # 其中figsize提供整数元组则会以该元组为长宽，单位是英寸。
    # 1英寸 = 2.54厘米
    cm = confusion_matrix(true_data, pre_data)
    if title is not None:
        plt.title(title)
    tick_marks = np.arange(max(set(true_data)) + 1)

    if len(set(true_data)) < 31:
        plt.xticks(tick_marks, fontsize=14, rotation=45)  #
        plt.yticks(tick_marks, fontsize=14)
    elif len(set(true_data)) < 60 & len(set(true_data)) > 31:
        plt.xticks(tick_marks, fontsize=12, rotation=90)  #
        plt.yticks(tick_marks, fontsize=12)
    elif len(set(true_data)) > 79:
        plt.xticks(tick_marks, fontsize=5.5, rotation=90)  #
        plt.yticks(tick_marks, fontsize=5.7)
    else:
        plt.xticks(tick_marks, fontsize=10, rotation=90)  #
        plt.yticks(tick_marks, fontsize=10)

    if normalize:
        cm2 = np.array(cm, dtype=np.float32)
        num = np.sum(cm2, axis=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm2[i][j] = 1.0 * cm2[i][j] / num[i]
                cm2[i][j] = round(float(cm2[i][j]), 3)
        plot = plt.imshow(cm2, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(plot)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        print('Confusion matrix, without normalization')
    plt.ylabel('True label', fontsize=18, labelpad=4.5)  #
    plt.xlabel('Predicted label', fontsize=18, labelpad=4.5)
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.16, left=0.15)
    acc_count = 0
    for i in range(len(true_data)):
        if true_data[i] == pre_data[i]:
            acc_count += 1
    acc = acc_count / len(true_data) * 1.0
    acc = format(acc, '.4%')
    print("Mean accuracy:", acc)
    plt.show()
    if path != None:
        if normalize:
            fig.savefig(path + 'Confusion_matrix_' + str(acc) + '.svg')
        else:
            fig.savefig(path + 'Confusion_matrix_without normalization_' + str(acc) + '.svg')
