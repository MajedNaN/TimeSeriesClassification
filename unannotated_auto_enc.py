from imports import * 
from functions import *
from archs import *
from mylearner import *

### loading data
data_folder = '/scratch/smartairsense/data/'
csv_file = os.path.join(data_folder,'df_minimal_clean.csv')
df_chunks = pd.read_csv(csv_file,chunksize=100000,usecols=['humidity_abs','temperature','tvoc','oxygen','co2','co','no2','o3'],dtype=np.float32)
n_features = 8
### autoencoder
autoencoder = AutoEncoder(c_in=n_features) 
### statistics over all chunks
history = dict(train=[], val=[])
n_x_train = 0
n_x_valid =0
n_epochs = 0
n_chunks = 0

valid_epochs = 0

with open('config.yaml') as f:  ### loading hyperparameters
    hyperparams = yaml.load(f,SafeLoader)
under_window = hyperparams['sample_segment']['under_window']
seq_len = hyperparams['sample_segment']['seq_len']
stride = hyperparams['sample_segment']['stride']
sliding_mode = hyperparams['sample_segment']['sliding_mode']
epochs = hyperparams['model']['epochs']
bs = hyperparams['model']['bs']
num_workers = hyperparams['model']['num_workers']
### for time 
start = time.time()

for df in df_chunks:
    
    ### impute NaN
    impute_NaN(df) ## polynomial interpolation with degree > 1 uses index, also convert dtype to float to work
    df.reset_index(drop=True,inplace= True)
    df.columns=df.columns


    ### sliding 
    X= sliding(seq_len,stride,df,mode=sliding_mode)  

    ## splitting and standardization
    splits = TrainValidTestSplitter(valid_size=0.01)(X[:,0,0]) ##### we DON'T have test set here
    x_train = np.zeros(X[splits[0]].shape,dtype=np.float32)
    x_valid = np.zeros(X[splits[1]].shape,dtype=np.float32)
    scalers = {}
    for i in range(x_train.shape[1]): ## n_features
        scalers[i] = StandardScaler()
        scalers[i].fit(np.unique((X[splits[0]])[:, i, :]).reshape(-1,1)) ### as we have overlapping samples
        x_train[:, i, :] = scalers[i].transform((X[splits[0]])[:, i, :].reshape(-1,1)).reshape(x_train.shape[0],x_train.shape[-1])
        x_valid[:, i, :] = scalers[i].transform((X[splits[1]])[:, i, :].reshape(-1,1)).reshape(x_valid.shape[0],x_valid.shape[-1])

    ###training/validation dataloaders
    Tsets = TSDatasets(x_train, inplace=True)
    Vsets = TSDatasets(x_valid, inplace=True)
    dls   = TSDataLoaders.from_dsets(Tsets, Vsets, bs = bs, num_workers=num_workers)
    #####training
    autoencoder, history_chunk = train_autoencoder(
    autoencoder,
    dls.train,
    dls.valid,
    n_epochs= epochs
    )
    ### accumulate train/valid loss
    history['train'].extend(history_chunk['train'])
    history['val'].extend(history_chunk['val'])
    ### total number of sequences x_train, x_valid
    n_x_train += x_train.shape[0]
    n_x_valid += x_valid.shape[0]
    ## n_epochs, n_chunks
    n_epochs += epochs
    n_chunks += 1

    ### save the autoencoder
    torch.save(autoencoder.state_dict(), f'models/{autoencoder._get_name()}_{seq_len}.pt')

    elapsed = (time.time()-start)/3600  ### in hours
    print(f'elapsed time = {elapsed} hours, # of chunks= {n_chunks}, # of epochs= {n_epochs}, # of train sequences= {n_x_train}, # of valid sequences= {n_x_valid}')

### plotting distribution of data
x_lbs = ['train set','valid set']
y_nmrs = [n_x_train,n_x_valid]

fig, ax = plt.subplots()    
ind = np.arange(len(y_nmrs))  # the x locations for the groups
bars = ax.bar(ind, y_nmrs, color="blue")
ax.set_xticks(ind)
ax.set_xticklabels(x_lbs, minor=False)
plt.title('Distribution of datset')
# plt.xlabel('x')
# plt.ylabel('y')
ax.bar_label(bars)
plt.savefig(f'distribution_auto_enc_{seq_len}.png')

### plot train/valid losses
plt.figure()
x = list(range(1,n_epochs+1))

plt.plot(x,history['train'],label='train_loss')
plt.plot(x,history['val'],label = 'valid_loss')
plt.xlabel('epochs')
plt.legend()
plt.savefig(f'losses_auto_enc_{seq_len}.png')

##############predict for autoencoder##################
# predictions, pred_losses = predict_autoencoder(autoencoder, dls.valid)
# plt.figure()
# sns.distplot(pred_losses, bins=50, kde=True)
# plt.title('Distribution of reconstruction error of predictions')

# ### correct predictions based on threshold for reconstruction error
# threshold = 0.2
# correct = sum(l <= threshold for l in pred_losses)
# print(f'Correct normal predictions: {correct}/{x_valid.shape[0]}')
