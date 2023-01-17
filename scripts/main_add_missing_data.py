import argparse
import os
from airdata import airdata

if __name__ == '__main__':
    #TODO: add the description for the args
    parser = argparse.ArgumentParser(description='smartair dataset arguments')

    parser.add_argument('-p', '--data_path', type=str, default='/afs/tu-chemnitz.de/home/urz/a/abom/internship/', help='')
    parser.add_argument('-f', '--file_name', type=str, default='df_limit_large.csv', help='')
    parser.add_argument('-s', '--save_path', type=str, default='/afs/tu-chemnitz.de/home/urz/a/abom/internship/', help='')
    parser.add_argument('--window_size', type=float, default='120')
    parser.add_argument('--min_diff_multip', type=float, default='1.5')
    parser.add_argument('--max_diff_multip', type=float, default='3.2')

    args = parser.parse_args()

    datacls = airdata()

    file_name = args.file_name
    window_size = args.window_size
    min_diff_multip = args.min_diff_multip
    max_diff_multip = args.max_diff_multip

    datacls.init_paths(path=args.data_path, file_name=args.file_name)
    df = datacls.read_df_file()
    df_new = datacls.add_missing_steps(df,
                                       group_name='deviceid_int',
                                       window_size=window_size,
                                       min_diff_multip=min_diff_multip,
                                       max_diff_multip=max_diff_multip)
    
    df_new = datacls.remove_columns(df_new, ['indx', 'date', 'time'])
    #df_new = datacls.add_datetime_columns(df_new, datetime_column='datetime')
    
    df_new.to_csv(os.path.join(args.save_path, '{}_clean.csv'.format(file_name[:file_name.rfind('.')])), index=False)
