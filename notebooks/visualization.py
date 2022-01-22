from src.visualization.visualization_handler import *
from src.visualization.visualization_handler import create_transfer_graphs


def visualization_main(run_heatmap=False, run_transfer_graphs=False, cross_org_dir_path=None, transfer_table_path=None):
    if run_heatmap:
        if cross_org_dir_path is None:
            cross_org_dir_path = list_files(MODELS_PATH / "cross_org_tabels")[-1]
        create_heatmaps(cross_org_dir_path)

    if run_transfer_graphs:
        if transfer_table_path is None:
            transfer_table_path = list_files(MODELS_PATH / "transfer_tables")[-1]
        create_transfer_graphs(transfer_table_path, ['ACC', 'F1_score'])


if __name__ == '__main__':
    visualization_main(run_heatmap=True, run_transfer_graphs=True)
