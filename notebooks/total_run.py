from notebooks.training import run_training
from notebooks.transferring import run_transfer

if __name__ == '__main__':
    # model_type = 'xgboost'

    for model_type in ['base', 'xgboost']:
        run_training(model_type)
        run_transfer(model_type=model_type)
        # run_intra_transfer(model_type = model_type)
    # create_transfer_graphs(compare_to_xgboost = True)
