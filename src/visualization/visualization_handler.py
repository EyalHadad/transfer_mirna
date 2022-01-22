import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import *
from src.models.models_handler import list_files, create_dir


def create_heatmaps(cross_org_data_path: Path):
    graph_dir_path = create_dir(MODELS_GRAPHS)
    for file in list_files(cross_org_data_path):
        heatmap_dict = acc_heatmap_dict if 'ACC' in file.stem else f1_heatmap_dict
        data = pd.read_csv(file, index_col=0)
        draw_heatmap(data=data, img_path=graph_dir_path / f"{file.stem}_heatmap.png", **heatmap_dict)


def draw_heatmap(data, img_path, **kwargs):
    ax = sns.heatmap(data, **kwargs)
    plt.xlabel('Testing Dataset', fontsize=10)
    plt.ylabel('Training Dataset', fontsize=10)
    ax.xaxis.label.set_color('purple')
    ax.yaxis.label.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(img_path)
    plt.clf()


def create_transfer_graphs(transfer_table_path, metrics):
    graph_dir_path = create_dir(MODELS_GRAPHS)
    for metric in metrics:
        graph_dict = dict()
        for file in list_files(transfer_table_path):
            if metric in file.stem:
                data = pd.read_csv(file).iloc[:, :-1]
                data = data.set_index(data['src_org'] + "_" + data['dst_org'])
                data = data.drop(['src_org', 'dst_org'], axis=1)
                graph_dict[file.stem] = data

        transfer_model_index = graph_dict[list(graph_dict.keys())[0]].index
        for ind in transfer_model_index:
            plot_df = pd.DataFrame()
            for k, trans_df in graph_dict.items():
                plot_df = plot_df.append(pd.Series(data=trans_df.loc[ind], name=k))
            draw_transfer_graph(plot_df, ind, graph_dir_path / f"{ind}.png")


def draw_transfer_graph(data, org_names, img_path):
    sns.set_theme(style="darkgrid")
    labels = [x.replace("base_", "miRNA_NET_") for x in data.index]
    ax = sns.lineplot(data=data.T, dashes=False, linewidth=2.5)
    ax.set(yticks=np.linspace(0, 1, 11))
    ax.legend(labels)
    plt.title(f"{org_names}", fontsize=15)
    plt.xlabel('#Target observations', fontsize=10)
    plt.ylabel('AUC', fontsize=10)
    plt.legend(loc='lower right')
    ax.title.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(img_path)
    plt.clf()
