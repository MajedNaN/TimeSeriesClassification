import argparse
import os
from airdata import airdata

if __name__ == '__main__':
    #TODO: add the description for the args
    parser = argparse.ArgumentParser(description='smartair dataset arguments')

    parser.add_argument('-p', '--data_path', type=str, default='../data', help='')
    parser.add_argument('-f', '--file_name', type=str, default='df_minimal_clean.csv', help='')
    parser.add_argument('-s', '--save_path', type=str, default='../data', help='')
    parser.add_argument('--window_size', type=int, default='30')
    parser.add_argument('--threshold_val', type=float, default='180')

    args = parser.parse_args()

    datacls = airdata()

    file_name = args.file_name
    window_size = args.window_size
    threshold_val = args.threshold_val

    datacls.init_paths(path=args.data_path, file_name=args.file_name)
    df = datacls.read_df_file()

    df_new = datacls.indexing_data_window(df, group_name='deviceid_int', window_size=window_size, threshold_val=threshold_val)

    df_new.to_csv(os.path.join(args.save_path, '{}_window_{}.csv'.format(file_name[:file_name.rfind('.')], window_size)), index=False)
