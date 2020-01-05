import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def correlation_heat_map_1(a, columns, index):
    # a = np.random.rand(31, 31)
    fig, ax = plt.subplots(figsize=(62, 62))
    # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
    # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
    sns.heatmap(pd.DataFrame(np.round(a, 2), columns=columns ,index=index),
                annot=True, vmax=0.2, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
    # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
    #            square=True, cmap="YlGnBu")
    ax.set_title('二维数组热力图', fontsize=18)
    ax.set_ylabel('数字', fontsize=18)
    ax.set_xlabel('字母', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的

    plt.show()


def correlation_heat_map(a, columns, index):
    fig, ax = plt.subplots(figsize=(31, 31))

    mask = np.zeros_like(a)
    mask[np.triu_indices_from(mask)] = True

    # sns.heatmap(pd.DataFrame(np.round(a, 2), columns=columns, index=index),
    #             mask=mask, annot=True, vmax=0.2, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")

    sns.heatmap(pd.DataFrame(np.round(a, 2), columns=columns, index=index),
                annot=True, vmax=0.2, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=45, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')

    plt.show()


def correlation_heat_map_save(a, columns, index, path):
    fig, ax = plt.subplots(figsize=(31, 31))

    # mask = np.zeros_like(a)
    # mask[np.triu_indices_from(mask)] = True

    # sns.heatmap(pd.DataFrame(np.round(a, 2), columns=columns, index=index),
    #             mask=mask, annot=True, vmax=0.2, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")

    sns.heatmap(pd.DataFrame(np.round(a, 2), columns=columns, index=index),
                annot=True, vmax=0.2, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=45, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')

    plt.savefig(path, format='png', dpi=300)
    plt.clf()


# correlation_heat_map(None, list(range(31)), list(range(31)))