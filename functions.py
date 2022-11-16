from imports import *
# ****************************
# standardize function *****************
# *******************************

def standardize(x,mode=0):
    
    size = x.shape[1]
    if mode==1: #normalize
        if x.ndim == 3: ### in case of segments
            mean = []
            std = []
            for i in range(size):
                mean.append(np.unique(x[:,i,:]).mean())
                std.append(np.unique(x[:,i,:]).std())
            x=(x-np.array(mean).reshape(1,size,1))/np.array(std).reshape(1,size,1) # to be applied tp features correctly

        elif x.ndim == 2: ### in case of data before segmentation
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            x=(x-mean.reshape(1,size))/std.reshape(1,size)
        return x.astype(np.float32),mean,std
    elif mode==2: #minmax
        if x.ndim == 3:
            min = []
            max = []
            for i in range(size):
                min.append(np.unique(x[:,i,:]).min())
                max.append(np.unique(x[:,i,:]).max())
            x=(x-np.array(min).reshape(1,size,1))/((np.array(max)-np.array(min)).reshape(1,size,1)) 

        elif x.ndim == 2:
            min = x.min(axis=0)
            max = x.max(axis=0)
            x=(x-min.reshape(1,size))/((max-min).reshape(1,size) )

        return x.astype(np.float32),min,max
    return x.astype(np.float32)

def standardize_with_params(x,**k):
    if len(k)==2:
        size = x.shape[1]
        if 'mean' in k and 'std' in k: #normalize
            mean = k['mean']
            std = k['std']
            if x.ndim == 3: ### in case of segments
                x=(x-np.array(mean).reshape(1,size,1))/np.array(std).reshape(1,size,1) # to be applied tp features correctly
            elif x.ndim == 2: ### in case of data before segmentation
                x=(x-np.array(mean).reshape(1,size))/np.array(std).reshape(1,size)
        elif 'min' in k and 'max' in k: #minmax
            min = k['min']
            max = k['max']
            if x.ndim == 3:
                x=(x-np.array(min).reshape(1,size,1))/((np.array(max)-np.array(min)).reshape(1,size,1)) 
            elif x.ndim == 2:
                x=(x-np.array(min).reshape(1,size))/((np.array(max)-np.array(min)).reshape(1,size) )
    return x.astype(np.float32)

# ****************************
# sliding function *****************
# *******************************

def sliding(window,stride,features,targets,start=0):

    x = []
    y = []
    count = 0
    i=start
    length = window + i
    size = features.shape[0]
    while  length <= size :
        x.append(features.iloc[i:i+window].values.tolist())
        y.append(targets.iloc[i])
        i += stride
        length = window + i
        count += 1
    x = np.array(x,dtype=np.float32)#.reshape((count,features.shape[1], window))
    x = np.transpose(x, (0, 2, 1)) ## to have correct segmentation shape (#samples,#features,seq_len)
    ### to be compatible with torch
    try:
        sh = targets.shape[1]
    except:
        sh = 1
    y = np.array(y,dtype=np.float32).reshape(-1,sh)
    return x,y

#### Grouping 0,1,2,... and stitching then slicing (destroys time series !!!)
# def sliding(window,stride,features,target):

#     df = pd.concat([features,target],axis=1)
#     df_g = df.groupby(df.columns[-1])

#     x=[]
#     y=[]
#     count = 0

#     for key, item in df_g:
#             i = 0
#             length = window
#             size = df_g.get_group(key).shape[0]
#             while  length <= size :
#                 x.append(df_g.get_group(key).iloc[i:i+window,0:-1].values.tolist())
#                 y.append(df_g.get_group(key).iloc[0,-1])
#                 i += stride
#                 length = window + i
#                 count += 1
#     x = np.array(x,dtype=np.float32)#.reshape((count,features.shape[1], window))
#     x = np.transpose(x, (0, 2, 1))
#     y = np.array(y,dtype=np.int)#reshape(-1)
#     return x,y



# ****************************
# under sampling function *****************
# *******************************


def under_sample(data, size ,seq_len,stride):
    x_list = []
    y_list = []

   
    data.reset_index(drop=True,inplace = True)
    lc = data[(data.loc[:,'person']>0) | (data.loc[:,'window_open']>0)]
    indices = lc.index

    while (len(indices) > 0): 
        left = indices[0] # choose the most left index contains 1
        if left - size < 0: # outside values to avoid empty list from left
            window = data.iloc[:left+size+1]
        else:
            window = data.iloc[left-size : left+size+1]
        #get features,targets for sliding
        f = window.drop(columns = ['person','window_open'])
        t = window.filter(['person','window_open'])      


        _x,_y = sliding(seq_len,stride,f,t)
        x_list.append(_x)
        y_list.append(_y)
        
        #drop selected window values, if count < window, finished undersampling
        try:
            data = data.drop(index = range(0,left+size+1))
        except:
            break
        data.reset_index(drop=True,inplace = True)
        
        #recalculate indices of 1s
        lc = data[(data.loc[:,'person']>0) | (data.loc[:,'window_open']>0)]
        indices = lc.index
        
    # convert lists to correct shape  for segmentation and standardization
    X = x_list[0]
    y = y_list[0]

    for i in range(1,len(x_list)):
        X= concat(X,x_list[i])
        y= concat(y,y_list[i])

    return X,y
    
# ****************************
# plot confusion matrix function *****************
# *******************************

def plot_confusion(y_true,y_pred,n_classes,name):
    cf_matrix=confusion_matrix(y_true,y_pred)


    plt.figure()

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(n_classes,n_classes)  # 2: number of classes

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',cbar = False)
    ax.set_title(f'Confusion Matrix of class {name}\n');
    ax.set_xlabel('\nPredicted Classes')
    ax.set_ylabel('Actual Classes ');

## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(['Not Setting','Setting']) ### names of classes starting from 0
    # ax.yaxis.set_ticklabels(['Not Setting','Setting'])

## Display the visualization of the Confusion Matrix.
    plt.show()