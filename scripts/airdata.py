import os
import h5py
import numpy as np
import datetime
import math
import pandas as pd
import argparse

class airdata():
    def __init__(self):
        print('airdata class is initiated')

    def init_paths(self, path='./', file_name='part.h5'):
        self.path = path
        self.file_name = file_name


    def load_file(self):
        f = []
        df_type = ''
        self.file_path = os.path.join(self.path, self.file_name)
        if self.file_path.endswith('.h5'):
            f = h5py.File(self.file_path, 'r')
            df_type = 'h5'
        return f, df_type

    # def load_data(self):
    #     self.df = self.df['df']

        # df1['block0_values']
        # h[0][:]
        # df = pd.read_hdf('./part1.h5')
        # hh = df.sort_values('deviceid')

    def process_data(self):
        self.df, df_type = self.load_file()
        if df_type == 'h5':
            self.load_data()

    def load_all_files(self, file_extension='.h5'):

        files_list = [f for f in os.listdir(self.path) if f.endswith(file_extension)]
        df_list = [pd.read_hdf(os.path.join(self.path, f)) for f in files_list]
        df = pd.DataFrame()
        for df_temp in df_list:
            # df = df.append(df_temp, ignore_index=True)
            df = pd.concat([df,df_temp])
        return df

    def sort_by(self, df, all_by='deviceid', group_by='datetime'):
        df = df.sort_values(all_by)
        df.insert(df.keys().to_list().index(all_by) + 1, '{}_int'.format(all_by), df[all_by].factorize()[0], True)
        return (df.groupby('{}_int'.format(all_by), group_keys=False).apply(lambda s: s.sort_values(by=group_by, ascending=True)))


    def add_datetime_columns(self, df, datetime_column='datetime'):

        df.insert(df.keys().to_list().index(datetime_column) + 1, 'date', df[datetime_column].dt.date, True)
        df.insert(df.keys().to_list().index(datetime_column) + 2, 'time', df[datetime_column].dt.time, True)
        return df

    def extract_metadata(self, df, nans_flag=True, minmax_flag=False, extra_info_flag=False):

        deviceid_int = np.unique(df['deviceid_int'].to_numpy()).tolist()
        df_keys = df.keys().to_list()

        if not('time' in df_keys):
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = self.add_datetime_columns(df, datetime_column='datetime')

        df_keys = df.keys().to_list()
        sensor_indx = df_keys.index('time') + 1
        sensor_keys = df_keys[sensor_indx:]

        meta_data = []
        for i in deviceid_int:
            df_temp = df.loc[df['deviceid_int'] == i].copy()
            dict_temp = {}
            dict_temp['deviceid_int'] = i

            deviceid = df_temp['deviceid'].to_numpy()

            if (deviceid[0] == deviceid).all():
                dict_temp['deviceid'] = deviceid[0]
            else:
                print('Not all device IDs are the same, but the integer values assigned to them are.')
                print('Device ID integer {} should be checked'.format(i))
                dict_temp['deviceid'] = '-1'

            dict_temp['num_datapoints'] = len(df_temp.index)
            dict_temp['date_min'] = df_temp['date'].min()
            dict_temp['date_max'] = df_temp['date'].max()
            dict_temp['recorded_time'] = df_temp['datetime'].max() - df_temp['datetime'].min()
            dict_temp['num_days'] = pd.unique(df_temp['date']).size

            if nans_flag:
                for k in sensor_keys:
                    dict_temp['{}_isNaN'.format(k)] = df_temp[k].isnull().values.any()
                    dict_temp['{}_numNaN'.format(k)] = df_temp[k].isnull().sum()
                    dict_temp['{}_ratioNaN'.format(k)] = (dict_temp['{}_numNaN'.format(k)] / dict_temp['num_datapoints']) * 100.0

            if minmax_flag:
                for k in sensor_keys:
                    dict_temp['{}_min'.format(k)] = df_temp[k].min()
                    dict_temp['{}_max'.format(k)] = df_temp[k].max()

                    if extra_info_flag:
                        dict_temp['{}_mean'.format(k)] = df_temp[k].mean()
                        dict_temp['{}_median'.format(k)] = df_temp[k].median()
                        dict_temp['{}_std'.format(k)] = df_temp[k].std()

            meta_data.append(dict_temp)

        df_meta = pd.DataFrame(meta_data, columns=list(dict_temp.keys()))

        return df_meta

    def save_df(self, df, path_to_save='./', file_name='df_meta'):
        df_path = os.path.join(path_to_save, '{}.csv'.format(file_name))
        df.to_csv(df_path, index=False)
        print('{} file saved at {}'.format(file_name, df_path))


    def remove_columns(self, df, list_to_remove):
        for col in list_to_remove:
            if col in df.columns:
                df.drop(col, inplace=True, axis=1)

        return df

    def select_rows(self, df, list_to_keep, key='deviceid'):
        # TODO: add multiple conditions (or keys) with their corresponding list, so it applies all of them in one-go

        return df[df[key].isin(list_to_keep)]

    def select_date_range(self, df, key='datetime', min_range='2021-01', max_range='2022.10'):
        df[key] = pd.to_datetime(df[key])
        return df.loc[df[key].between(min_range, max_range)]


    def select_columns(self, df, list_to_select):

        list_inconsist = [l for l in list_to_select if not (l in list(df.keys()))]

        if not(not list_inconsist):
            print('The following item(s) doesn\'t exist in the given list, {}'.format(list_inconsist))
            for e in list_inconsist:

                if e in list_to_select:
                    list_to_select.remove(e)

        return df[list_to_select]

    def remove_nan(self, df, from_list=None):
        if from_list is None:
            df = df.dropna(axis=0)
        else:
            df = df.dropna(subset=from_list)
        return df

    def add_missing_steps(self, df, group_name='deviceid_int', window_size=2*60, min_diff_multip=1.5, max_diff_multip=3.2):
        df.insert(loc=0, column='indx', value=list(df.index))
        df['datetime'] = pd.to_datetime(df['datetime'])
        # TODO: reset index
        df_list = []
        added_entries = 0
        for i in df[group_name].unique().tolist():
            temp = df.loc[df[group_name] == i].copy()
            diff_temp = temp['datetime'].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(-1).astype('int64')
            diff_vec = diff_temp.to_numpy()
            mask_vec = \
            np.where((diff_vec > min_diff_multip * window_size) & (diff_vec < max_diff_multip * window_size))[0]
            temp_list_indices = diff_temp.index.to_list()
            temp.reset_index(drop=True, inplace=True)

            step_vec = np.array([math.ceil(e) if e % 1 > 0.5 else math.floor(e) for e in diff_vec / window_size]) - int(min_diff_multip)
            step_vec_pos_indx = np.where(step_vec > 0)
            step_vec_pos = step_vec[step_vec_pos_indx]
            num_entries = np.sum(step_vec_pos[step_vec_pos < max_diff_multip])
            added_entries += num_entries
            print('{} entries should be added to the device ID {}'.format(num_entries, i))

            for m in mask_vec:
                frac_size = diff_vec[m] / window_size
                if frac_size % 1 > 0.5:
                    frac_size = math.ceil(frac_size)
                else:
                    frac_size = math.floor(frac_size)
                steps = int(frac_size)
                for n in range(0, steps - 1):
                    time_added = temp.loc[temp['indx'] == temp_list_indices[m] - 1].iloc[0]['datetime'] + \
                                 pd.to_timedelta((n + 1) * int(diff_vec[m] / steps), unit='s')
                    # n could be removed from the index as we simply want to add that many number of the same rows in the place.
                    # -1 as the index could also be 0, as there would be no difference between the rows added in between.
                    row_index = temp.index[temp['indx'] == temp_list_indices[m]].tolist()[-1]  # -n
                    temp = insert_value_row(row_index,
                                      temp,
                                      temp.loc[temp['indx'] == temp_list_indices[m] - 1].iloc[0])

                    # temp['datetime'][row_index] = time_added
                    temp.at[row_index, 'datetime'] = time_added

                    temp.reset_index(drop=True, inplace=True)

            df_list.append(temp)

        print('total of {} entries have been added to the dataframe'.format(added_entries))

        return pd.concat(df_list)

    def indexing_data_window(self, df, group_name='deviceid_int', window_size=2*60, threshold_val=2*60*1.5):
        df.insert(loc=0, column='indx', value=-1)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # TODO: reset index
        df_list = []
        indx = 0

        for i in df[group_name].unique().tolist():
            df_temp = df.loc[df[group_name] == i].copy()
            diff_temp = df_temp['datetime'].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(-1).astype(
                'int64')
            diff_vec = diff_temp.to_numpy()
            df_temp.reset_index(drop=True, inplace=True)

            counter = 0

            while counter < len(diff_vec) - window_size:
                diff_temp = diff_vec[counter:counter + window_size]
                cond_temp = (diff_temp > 0) & (diff_temp < threshold_val)
                if (counter == 0) & (cond_temp[0] == False):
                    cond_temp[0] = True
                # elif (counter != 0) & (cond_temp[0] == False):
                #     # TODO: this should be changed (there could be two consecutive values that are huge, i.e. only 2 min per hour or day is recorded)
                #     print(
                #         'this should not happen, only the first element should be -1 in the diff_vec. Check what happened.')

                if cond_temp.all():
                    temp = df_temp.iloc[counter:counter + window_size].copy()
                    temp.at[temp.index, 'indx'] = indx
                    df_list.append(temp)
                    indx += 1
                    counter += window_size
                else:
                    counter += np.where(cond_temp == False)[0][-1] + 1

        return pd.concat(df_list)

    def clean_df(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.reset_index(drop=True, inplace=True)
        return df

    def read_df_file(self, path=None, file_name=None):
        if (path is None) | (file_name is None):
            file_path = os.path.join(self.path, self.file_name)
        elif (not(path is None)) | (not(file_name is None)):
            file_path = os.path.join(path, file_name)
        else:
            print('please provide the proper path to the file.')
            file_path = ''

        if os.path.exists(file_path):
            if (file_path.endswith('.csv')):
                return pd.read_csv(file_path)
            else:
                print('not the proper file is expected to be loaded.')
                return 0
        else:
            print('directory/path/file does not exist.')
            return 0

    def load_data(self, data_path=None, data_dict=None):

        df = []

        if data_path is None:
            print('please provide a path to the preprocessed data or a path to the data for preprocessing.')
        else:
            if os.path.exists(data_path):
                if (data_path.endswith('.csv')):
                    if data_dict is None:
                        # load the file
                        print('loading the preprocessed file...')
                        # TODO: these files are huge, there should be a better way to stream them
                        df = pd.read_csv(data_path)
                    else:
                        # load the file for preprocessing
                        print('loading the file to be preprocessed...')
                        df_temp = pd.read_csv(data_path)
                        # TODO: these files are huge, there should be a better way to stream them, same for preprocessing
                        print('preprocessing the data, it may take a while...')
                        # example of data_dict = {'group_name':'deviceid_int', 'window_size':30, 'threshold_val':180}

                        df = self.indexing_data_window(df_temp,
                                                       group_name=data_dict['group_name'],
                                                       window_size=data_dict['window_size'],
                                                       threshold_val=data_dict['threshold_val'])
                else:
                    print('the file extension is not csv, please provide the path to the proper file.')

            else:
                print('the file does not exist, please provide the proper path.')

        return df

def insert_value_row(row_index, df, df_value):

    first_part = [*range(0, row_index, 1)]
    second_part = [*range(row_index, df.shape[0], 1)]

    second_part = [x.__add__(1) for x in second_part]

    indices = first_part + second_part
    df.index = indices

    #TODO: this is weird, check if there is a better way!
    df.loc[row_index] = df_value.values#[0, :]
    df = df.sort_index()

    return df