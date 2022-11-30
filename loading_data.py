from imports import *

data_folder = '/scratch/smartairsense/data_annotated'

prqt_file1 = os.path.join(data_folder,'annotated-office.parquet')
prqt_file2 = os.path.join(data_folder,'2022-week39.parquet')
prqt_file3 = os.path.join(data_folder,'2022-week40.parquet')
prqt_file4 = os.path.join(data_folder,'2022-week41.parquet')
prqt_file5 = os.path.join(data_folder,'2022-week42.parquet')
prqt_file6 = os.path.join(data_folder,'2022-week43.parquet')
prqt_file7 = os.path.join(data_folder,'2022-week44.parquet')
prqt_file8 = os.path.join(data_folder,'2022-week45.parquet')
prqt_file9 = os.path.join(data_folder,'2022-week46.parquet')
prqt_file10 = os.path.join(data_folder,'2022-week47.parquet')

data1 = pd.read_parquet(prqt_file1 , engine='fastparquet')
data2 = pd.read_parquet(prqt_file2 , engine='fastparquet')
data3 = pd.read_parquet(prqt_file3 , engine='fastparquet')
data4 = pd.read_parquet(prqt_file4 , engine='fastparquet')
data5 = pd.read_parquet(prqt_file5 , engine='fastparquet')
data6 = pd.read_parquet(prqt_file6 , engine='fastparquet')
data7 = pd.read_parquet(prqt_file7 , engine='fastparquet')
data8 = pd.read_parquet(prqt_file8 , engine='fastparquet')
data9 = pd.read_parquet(prqt_file9 , engine='fastparquet')
data10 = pd.read_parquet(prqt_file10 , engine='fastparquet')

d1 = data1.copy()
d2 = concat(data2,data3,data4,data5,data6,data7,data8,data9,data10)


d1.reset_index(drop = True,inplace = True)
d2.reset_index(drop = True,inplace = True)

d1.columns=data1.columns
d2.columns=data1.columns

# shift columns person and window_open to the last

last_index = len(d1.columns)

person = d1.pop('person')
d1.insert(last_index-1, 'person', person)
window_open = d1.pop('window_open')
d1.insert(last_index-1, 'window_open', window_open)
person = d2.pop('person')
d2.insert(last_index-1, 'person', person)
window_open = d2.pop('window_open')
d2.insert(last_index-1, 'window_open', window_open) 

data = concat(d1,d2)
data.reset_index(drop = True,inplace=True)
data.columns=d1.columns


print(f'data shape: {data.shape}\n {d2}')