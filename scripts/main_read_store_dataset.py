import argparse
import os
from airdata import airdata
import pandas as pd

if __name__ == '__main__':
    #TODO: add the description for the args
    parser = argparse.ArgumentParser(description='smartair dataset arguments')

    parser.add_argument('-p', '--data_path', type=str, default='/scratch/smartairsense/h5Files', help='')
    parser.add_argument('-f', '--file_name', type=str, default='part1.h5', help='')
    parser.add_argument('-s', '--save_path', type=str, default='~/internship', help='')
    parser.add_argument('--flag_minimal', action='store_true', help='')
    parser.add_argument('--flag_metadata', action='store_true', help='')
    parser.add_argument('-r', '--remove_list', nargs='*', default=['h2s'], help='')
    parser.add_argument('--flag_remove_devices', action='store_true', help='')
    parser.add_argument('--file_name_devices', type=str, default='./devices_to_keep.txt', help='')
    parser.add_argument('--flag_date_range', action='store_true', help='')
    # TODO: add rules for the arguments, size 2 for date_range, with first element < second element
    parser.add_argument('--date_range', type=str, default=['2021-06', '2022-07'], help='')

    args = parser.parse_args()

    datacls = airdata()
    datacls.init_paths(path=args.data_path, file_name=args.file_name)
    
    df = datacls.load_all_files(file_extension='.h5')

    if args.flag_remove_devices:
        with open(args.file_name_devices) as file:
            devices_list = file.readlines()
            devices_list = [d.rstrip() for d in devices_list]

        df = datacls.select_rows(df, devices_list, key='deviceid')

    if args.flag_date_range:

        df = datacls.select_date_range(df, key='datetime', min_range=args.date_range[0], max_range=args.date_range[1])

    df_s = datacls.sort_by(df, all_by='deviceid', group_by='datetime')
    df_s = datacls.add_datetime_columns(df_s, datetime_column='datetime')

    if args.flag_metadata:
        df_meta = datacls.extract_metadata(df_s, nans_flag=True, minmax_flag=False, extra_info_flag=False)
        datacls.save_df(df_meta, path_to_save=args.save_path, file_name='df_meta_all')

        # df_meta = datacls.print_metadata(df_s, path_to_save=args.save_path, file_name='df_meta_all')

    list_to_remove = args.remove_list

    df_s = datacls.remove_columns(df_s, list_to_remove)

    list_columns = ['deviceid', 'deviceid_int', 'timestamp', 'datetime', 'date', 'time']

    if args.flag_minimal == True:

        list_parameters = ["humidity",
                           "temperature",
                           "tvoc",
                           "oxygen",
                           "co2",
                           "co",
                           "pressure",
                           "o3",
                           "sound" ]

    else:

        list_parameters = [
            "pressure",
            "temperature",
            "sound",
            "tvoc",
            "oxygen",
            "humidity",
            "humidity_abs",
            "co2",
            "co",
            "so2",
            "no2",
            "o3",
            "pm2_5",
            "pm10",
            "pm1",
            "sound_max",
            "dewpt"
        ]


    # if args.flag_minimal == True:

    #     list_parameters = ["humidity_abs",
    #                        "temperature",
    #                        "tvoc",
    #                        "oxygen",
    #                        "co2",
    #                        "co",
    #                        "no2",
    #                        "o3", ]

    # else:

    #     list_parameters = [
    #         "pressure",
    #         "temperature",
    #         "sound",
    #         "tvoc",
    #         "oxygen",
    #         "humidity_abs",
    #         "co2",
    #         "co",
    #         "so2",
    #         "no2",
    #         "o3",
    #         "pm2_5",
    #     ]

    df_new = datacls.select_columns(df_s, list_columns + list_parameters)

    # ### interpolating NAN values
    df_new.reset_index(drop=True,inplace=True)
    device_ids= df_new['deviceid'].unique()
    count = 0
    for device_id in device_ids:
        idx = list(df_new.loc[df_new['deviceid'] == device_id].index)
        for column in list_parameters:
            df_temp = df_new.loc[idx,column].interpolate(method='linear')
            df_new.loc[idx,column] = df_temp.values
        count += 1
        print(count)
    # ### removing NaN values at beginning and end

    df_new = datacls.remove_nan(df_new, from_list=list_parameters)


    df_new.to_csv(os.path.join(args.save_path, 'df_limit_{}.csv'.format('minimal' if (args.flag_minimal == True) else 'large')), index=False)
