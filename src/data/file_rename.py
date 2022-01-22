import os

from constants import *

os.chdir('../..')


def change_file_names():
    dir_path = EXTERNAL_PATH
    conv_dict = {'positive': '_pos.csv', 'negative': '_neg.csv', 'dataset1': '1', 'dataset2': '2', 'dataset3': '3'}
    conv_dict_names = {'cattle': 'cow', 'human': 'human', 'celegans': 'worm', 'mouse': 'mouse'}
    files = os.listdir(dir_path)
    for index, file in enumerate(files):
        if '.csv' in file:
            s_file = file.split('_')
            new_f_name = conv_dict_names[s_file[0]] + conv_dict[s_file[1]] + conv_dict[s_file[3]]
            os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_f_name))
            print(f"file name {file} converted to {new_f_name}")


if __name__ == '__main__':
    change_file_names()
