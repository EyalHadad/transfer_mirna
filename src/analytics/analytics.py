from src.models.models_handler import *
from src.data.data_handler import *
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import shap


def create_error_file(org_name, target_dataset, x, y):
    for model_type in ['base', 'xgboost']:
        model = load_trained_model(model_type, org_name)
        print(f"Predicting {target_dataset} using {model_type}_{org_name}")
        pred = model.predict(x)
        total_frame = x.copy()
        total_frame["index"] = x.index
        total_frame["actual"] = y
        total_frame["predicted"] = np.round(pred)
        incorrect = total_frame[total_frame["actual"] != total_frame["predicted"]]
        incorrect.to_csv(
            os.path.join(MODELS_WRONG_PREDICTIONS, f"{target_dataset}_{org_name}_{model_type}_incorrect.csv"),
            index=False)


def get_target_test_data(target_dataset):
    test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(target_dataset)), index_col=False)
    test['sequence'] = test.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
    y = test['label']
    X = test.drop(FEATURES_TO_DROP, axis=1)
    X.drop('sequence', axis=1, inplace=True)
    x = X.astype("float")
    return x, y


def create_error_datasets(target_dataset):
    x_test, y_test = get_target_test_data(target_dataset)
    for org_name in DATASETS:
        print(f"Analysis {org_name}")
        create_error_file(org_name, target_dataset, x_test, y_test)


def analysis_error_datasets(target_dataset):
    csv_files = [x for x in os.listdir(MODELS_WRONG_PREDICTIONS) if f"{target_dataset}_" in x]
    wrong_pred_dict = create_wrong_dict(csv_files)
    write_statistics_file(target_dataset, wrong_pred_dict)


def write_statistics_file(target_dataset, wrong_pred_dict):
    f_name = os.path.join(MODELS_ERROR_STATISTICS, f"{target_dataset}_statistics.csv")
    with open(f_name, 'w') as file:
        file.write("s_org,stat,miRNA_Net,Xgboost\n")
        for t_org, v in wrong_pred_dict.items():
            base_wrong = wrong_pred_dict[t_org]['base']
            xg_wrong = wrong_pred_dict[t_org]['xgboost']
            amount_row = f"{t_org},#mis predictions,{base_wrong.shape[0]},{xg_wrong.shape[0]}\n"
            file.write(amount_row)
            fp_row = f"{t_org},#false positive,{base_wrong[base_wrong['predicted'] == 1].shape[0]},{xg_wrong[xg_wrong['predicted'] == 1].shape[0]}\n"
            file.write(fp_row)
            u_miss_base = base_wrong[~base_wrong.index.isin(xg_wrong.index)].shape[0]
            u_miss_xg = xg_wrong[~xg_wrong.index.isin(base_wrong.index)].shape[0]
            unique_rows = f"{t_org},#unique mistakes,{u_miss_base},{u_miss_xg}\n"
            file.write(unique_rows)


def create_wrong_dict(csv_files):
    wrong_pred_dict = {}
    for file in csv_files:
        data = pd.read_csv(os.path.join(MODELS_WRONG_PREDICTIONS, file), index_col=['index'])
        t_org = file.split('_')[1]
        if t_org not in wrong_pred_dict.keys():
            wrong_pred_dict[t_org] = {}
        model_type = file.split('_')[2]
        wrong_pred_dict[t_org][model_type] = data
    return wrong_pred_dict


def find_most_distinguish_cols(df1,df2,n=1,df_col=None,return_all=False):
    distinguish_list = []
    if df_col is None:
        df_col = df1.columns
    for c in df_col:
        res = ks_2samp(df1[c], df2[c])
        distinguish_list.append([c,res.statistic,res.pvalue])
    distinguish_list.sort(key=lambda tup: tup[2])
    if return_all:
        return distinguish_list
    return [x[0] for x in distinguish_list[0:n]]


def plot_distinguish_cols(f1_path,f2_path,col_name,s_org,d_org):
    network_worng_pred = pd.read_csv(f1_path)
    xgboost_wrong_pred = pd.read_csv(f2_path)
    target_dataset = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(d_org)), index_col=False)
    network_unique_misstake = network_worng_pred[~network_worng_pred.index.isin(xgboost_wrong_pred.index)]
    xgboost_unique_misstake = xgboost_wrong_pred[~xgboost_wrong_pred.index.isin(network_worng_pred.index)]
    network_worng_pred.loc[:,'dataset'] = 'Network'
    xgboost_wrong_pred.loc[:,'dataset'] = 'Xgboost'
    target_dataset.loc[:,'dataset'] = f"{s_org}_test_distribution"
    network_unique_misstake['dataset'] = 'Network unique mistakes'
    xgboost_unique_misstake['dataset'] = 'Xgboost unique mistakes'
    new_data = network_worng_pred.append(xgboost_wrong_pred).append(target_dataset).append(network_unique_misstake).append(xgboost_unique_misstake)
    sns.set_theme(style="darkgrid")
    sns.displot(data=new_data, x=col_name, hue='dataset',kind="kde")
    # plt.title(f"{s_org} model predict {d_org} interactions", fontsize=10)
    plt.savefig(os.path.join(MODELS_ERROR_STATISTICS, f"{d_org}_{col_name}_wrong_prediction.png"))
    plt.clf()


def model_diff(s_org,d_org):
    dict_names = {'base': f"base_{s_org}_{d_org}_incorrect.csv",
                  'xgboost': f"xgboost_{s_org}_{d_org}_incorrect.csv"}

    nn_path = os.path.join(MODELS_PREDICTION_PATH, dict_names['base'])
    xg_path = os.path.join(MODELS_PREDICTION_PATH, dict_names['xgboost'])
    col_names = find_most_distinguish_cols(nn_path, xg_path, n=10)
    for c in col_names:
        plot_distinguish_cols(nn_path,xg_path, c, s_org, d_org)


def dataset_diff():
    dataset_list = copy.deepcopy(DATASETS)
    for src_org_name in dataset_list:
        train = pd.read_csv(os.path.join(PROCESSED_TRAIN_PATH, "{0}_train.csv".format(src_org_name)),
                           index_col=False)
        train = train.drop(FEATURES_TO_DROP, axis=1)
        rest = copy.deepcopy(DATASETS)
        rest.remove(src_org_name)
        for dst_org_name in rest:
            test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(dst_org_name)),
                               index_col=False)
            test = test.drop(FEATURES_TO_DROP, axis=1)
            col_names = find_most_distinguish_cols(train, test,n=10)
            update_dataset_diff_file(col_names, src_org_name,dst_org_name,'datasets_diff_features.csv',map(str, range(11)))

            # draw_dataste_feature_diff(train,test,[col_names[0]], src_org_name, dst_org_name)


def draw_dataste_feature_diff(train,test,col_names, src_org, dst_org):
    src_data = train.copy()
    dst_data = test.copy()
    src_data.loc[:, 'dataset'] = f"{src_org}_train_distribution"
    dst_data.loc[:, 'dataset'] = f"{dst_org}_test_distribution"
    all_data = src_data.append(dst_data)
    for c in col_names:
        sns.set_theme(style="darkgrid")
        sns.displot(data=all_data, x=c, hue='dataset', kind="kde")
        plt.savefig(os.path.join(MODELS_FEATURE_DIFF, f"{src_org}_{dst_org}_{col_names[0]}_dataset_diff.png"))
        plt.clf()


def update_dataset_diff_file(col_names, src_org, dst_org, output_f_name, file_col):
    f_path = os.path.join(MODELS_FEATURE_IMPORTANCE, output_f_name)
    with open(f_path, 'a') as file:
        if file.tell() == 0:
            file.write("src,dst," + ','.join(file_col) + "\n")
        row_list = [src_org, dst_org] + col_names + ['\n']
        file.write(",".join(row_list))


def create_shap_global_plots():
    dataset_list = copy.deepcopy(DATASETS)
    for src_org_name in dataset_list:
        xgb_model = load_trained_model("xgboost", src_org_name)
        base_model = load_trained_model("base", src_org_name)
        rest = copy.deepcopy(DATASETS)
        rest.remove(src_org_name)
        for dst_org_name in rest:
            test = pd.read_csv(os.path.join(PROCESSED_TEST_PATH, "{0}_test.csv".format(dst_org_name)),
                                index_col=False)
            test = test.drop(FEATURES_TO_DROP, axis=1)
            model_shap_plot(test, xgb_model,'xgboost',src_org_name,dst_org_name,'Energy_MEF_Duplex')
            model_shap_plot(test, base_model,'base',src_org_name,dst_org_name,'Energy_MEF_Duplex')


def model_shap_plot(data, model, model_name,s_org,d_org,dependence_feature=None):
    if model_name == 'base':
        shap_values = shap.DeepExplainer(model, data.values.astype('float')).shap_values(data.values.astype('float'))[0]
    else:
        shap_values = shap.TreeExplainer(model).shap_values(data.values.astype('float'))
    shap.summary_plot(shap_values, data,show=False,max_display=10,feature_names=data.columns)
    plt.title(f"{model_name}_{s_org}_{d_org}_summary_plot")
    plt.savefig(os.path.join(MODELS_FEATURE_SUMMARY, f"{model_name}_{s_org}_{d_org}_summary_plot.png"), bbox_inches='tight')
    plt.clf()
    if dependence_feature is not None:
        shap.dependence_plot(dependence_feature, shap_values, data,show=False)
        plt.title(f"{model_name}_{s_org}_{d_org}_dependence_plot")
        plt.savefig(os.path.join(MODELS_FEATURE_DEPENDENCE, f"{model_name}_{s_org}_{d_org}_dependence_plot.png"),
                    bbox_inches='tight')
        plt.clf()

