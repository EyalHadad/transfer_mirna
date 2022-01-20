from src.models.models_handler import *
from src.visualization.graphs_handler import create_transfer_graphs, create_intra_transfer_graphs
from src.visualization.visualization_handler import *


def run_visualization():
    # cross_org_tables = {'miRNA_Net': r"base_11_11_2021 16_18_16.csv", 'xgboost': r"xgboost_11_11_2021 15_57_33.csv"}
    # TODO change loading tables
    # for t_type in ['8X8','4vs4','4vs4_hu','4vs4_mu','4vs4_mu_hu']:
    #     metric = 'ACC'
    #     cross_org_tables = {'miRNA_Net': f"base_{metric}_{t_type}.csv", 'xgboost': f"xgboost_{metric}_{t_type}.csv"}
    #     create_heatmaps(cross_org_tables,metric,t_type)
    for t_type in ['4vs4']:
        for metric in ['ACC','F1_score']:
            cross_org_tables = {'miRNA_Net': f"base_{metric}_{t_type}.csv", 'xgboost': f"xgboost_{metric}_{t_type}.csv"}
            create_heatmaps(cross_org_tables,metric,t_type)

    # cross_org_tables = {'miRNA_Net': r"base_ACC.csv", 'xgboost': r"xgboost_ACC.csv"}
    # create_heatmaps(cross_org_tables)
    # create_datasets_features_importance_file(cross_org_tables)
    # draw_dataset_feature_importance_lineplot()
    model_list = ['base_20','xgboost_20','base_baseline','xgboost_baseline']
    # create_transfer_graphs(model_list)
    # create_intra_transfer_graphs(model_list) # should be the same for aa1




if __name__ == '__main__':
    run_visualization()
