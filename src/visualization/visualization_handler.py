import statistics
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from src.analytics.analytics import find_most_distinguish_cols, update_dataset_diff_file
from constants import *
from constants import MODELS_STATISTICS_PATH


def get_important_features(datasets_tuple, n=5):
    f_path = os.path.join(MODELS_FEATURE_IMPORTANCE, 'datasets_diff_features.csv')
    dataset_diff_file = pd.read_csv(f_path)
    file_line = dataset_diff_file[
                    (dataset_diff_file['src'] == datasets_tuple[0]) & (
                            dataset_diff_file['dst'] == datasets_tuple[1])].iloc[0, :]
    col_names = [str(x) for x in range(n + 1)]
    important_deatures = [file_line[x] for x in col_names]
    return important_deatures


def create_imp_features_file(dataset_list, imp_features):
    for d in dataset_list:
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, f"{d[0][0]}_train.csv"), index_col=False)
        train = train.drop(FEATURES_TO_DROP, axis=1)
        test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, f"{d[0][1]}_test.csv"), index_col=False)
        test = test.drop(FEATURES_TO_DROP, axis=1)
        dist_list = find_most_distinguish_cols(train, test, n=len(imp_features), df_col=imp_features, return_all=True)
        # dist_list = [str(x) for x in dist_list]
        col_names = [x[0] for x in dist_list]
        values = [str(x[2]) for x in dist_list]
        update_dataset_diff_file(values, src_org=d[0][0], dst_org=d[0][1],
                                 output_f_name='datasets_diff_features_lineplot.csv', file_col=col_names)


def draw_dataset_feature_importance_lineplot():
    f_path = os.path.join(MODELS_FEATURE_IMPORTANCE, 'datasets_diff_features_lineplot.csv')
    org_data = pd.read_csv(f_path, index_col=False)
    for d in DATASETS:
        for f in ['src', 'dst']:
            data = org_data.copy()
            data = data[data[f] == d]
            # data = data.iloc[[0, 10, 20, 30, 40, 48], :]
            # data = data.iloc[[0, 10, 18, 28, 38, 43], :]
            data.drop(['src', 'dst'], axis=1, inplace=True)
            plt.clf()
            sns.set_theme(style="whitegrid")
            ax = sns.lineplot(data=data, linewidth=2.5)
            plt.title('Features correlations between datasets', fontsize=15)
            plt.xlabel("Sorted models by AAC (0 is the lowest)", fontsize=13)
            plt.ylabel("Correlation", fontsize=13)
            ax.title.set_color('purple')
            plt.legend(title='Features', loc='upper left')
            plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
            # plt.show()
            plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, f"Features correlations between datasets_{d}_{f}.png"))


def create_datasets_features_importance_file(f_names):
    data1 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['miRNA_Net']), index_col=0)
    data2 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['xgboost']), index_col=0)
    avg_dataset_dict = [data1 + data2][0].div(2).stack().to_dict()
    dataset_list = [(k, v) for k, v in sorted(avg_dataset_dict.items(), key=lambda item: item[1])]
    cross_org_dataset_list = [t for t in dataset_list if t[0][0][:-1] != t[0][1][:-1]]
    # imp_features = get_important_features(dataset_list[0][0])
    imp_features = GOOD_MODEL_SHAP_FEATURES
    create_imp_features_file(cross_org_dataset_list, imp_features)


def create_heatmaps(f_names, metric, t_type):
    heatmap_dict = {"cmap": "RdBu_r", "square": True, "linewidths": 3, "annot": True, "vmin": 0, "vmax": 1,"cbar_kws":None}
    data1 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['miRNA_Net']), index_col=0)
    data2 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f_names['xgboost']), index_col=0)
    if metric == 'ACC':
        heatmap_dict['vmin'] = 0.5
        heatmap_dict['cmap'] = "RdBu_r"
        heatmap_dict['cbar_kws'] = {'label': 'ACC'}

    else:
        heatmap_dict['vmin'] = 0
        heatmap_dict['cmap'] = sns.diverging_palette(145, 300, s=60, as_cmap=True)
        heatmap_dict['cbar_kws'] = {'label': 'F1 Score'}
    # heatmap_dict['vmin'] = min(data1.values.min(), data2.values.min())
    # heatmap_dict['vmax'] = max(data1.values.max(), data2.values.max())
    draw_heatmap(data=data1, img_name=f"base_{t_type}_heatmap_{metric}.png",
                 img_title=f"base_{t_type}_heatmap_{metric}", **heatmap_dict)
    draw_heatmap(data=data2, img_name=f"xgboost_{t_type}_heatmap_{metric}.png",
                 img_title=f"xgboost_{t_type}_heatmap_{metric}", **heatmap_dict)
    new_data = data1 - data2
    limit_num = max(new_data.values.max(), abs(new_data.values.min()))
    heatmap_dict['vmin'] = -limit_num
    heatmap_dict['vmax'] = limit_num
    draw_heatmap(data=new_data, img_name=f"{t_type}_{metric}_diff.png", img_title=f"{t_type}_{metric}_diff",
                 **heatmap_dict)


def draw_heatmap(data, img_name, img_title, xlabel='Testing Dataset', ylabel='Training Dataset', **kwargs):
    ax = sns.heatmap(data, **kwargs)
    # plt.title(img_title, fontsize=15)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    ax.xaxis.label.set_color('purple')
    # ax.title.set_color('purple')
    ax.yaxis.label.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(os.path.join(MODELS_GRAPHS_HEATMAP, img_name))
    plt.clf()


def get_statistic_str(row, action):
    str_list = [row[str(x)].split(":") for x in TRANSFER_SIZE_LIST]
    num_list = [[float(x) for x in lst] for lst in str_list]
    if action == 'std':
        num_list = [round(statistics.stdev(v), 2) for v in num_list]
    elif action == 'avg':
        num_list = [round(statistics.mean(v), 2) for v in num_list]
    r_list = ','.join([str(x) for x in num_list])
    return r_list


def save_statistic_file(model_type, action):
    data = pd.read_csv(os.path.join(MODELS_OBJECTS_TRANSFER_TABLES, f"{model_type}_transfer.csv"))
    out_f = os.path.join(MODELS_STATISTICS_PATH, f"{model_type}_{action}.csv")
    with open(out_f, 'w') as f:
        f_header = ['model'] + TRANSFER_SIZE_LIST + ['\n']
        f.write(','.join(str(x) for x in f_header))
        data.apply(
            lambda row: f.write("{0}_{1},{2}\n".format(row['src_org'], row['dst_org'], get_statistic_str(row, action))),
            axis=1)


def file_exists(model, file_type):
    f_path = os.path.join(MODELS_STATISTICS_PATH, f"{model}_{file_type}.csv")
    return os.path.exists(f_path)
