from imports import *

# ****************************
# weighted normalized kullback leibler divergence *****************
# *******************************
def normalized_kl(p,q,weight = 0.1): #normalized KL

    # Should be probabilities 
    p = p / torch.sum(p) 
    q = q/ torch.sum(q)

    ## to avoid dividing by zero
    epsilon = 0.00001

    p = p+epsilon
    q = q+epsilon

    kl = weight *  torch.sum(p*torch.log(p/q)) ## weighted distance 
    norm_kl = 1 - torch.exp(-1*kl) ## between '0':min distance, '1':max distance
    
    return (1-norm_kl)    ### reverse values to represent a score

# ****************************
# Ploting PR curve function *****************
# *******************************
def plot_PR_curve(class_name,y_true,y_probas):
    class_name = class_name

    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F1-Score=%.3f' % (thresholds[ix], fscore[ix]))

    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'PR-curve of class {class_name}')
    plt.scatter(recall[ix], precision[ix], marker='o', color='red', label='best threshold')
    plt.annotate(f'{thresholds[ix]:.4f}',xy = (recall[ix]-0.2, precision[ix]-0.1), color='red')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.show()
    return thresholds[ix], fscore[ix]

# ****************************
# visualize embeddings using sklearn and plotly *****************
# *******************************
def visualize_embeddings(features, person_labels, window_labels,n_components = 2, method = 'pca'):
    #### normalize
    # features = StandardScaler().fit_transform(features)

    if method == 'tsne':
        tsne = TSNE(n_components = n_components)
        components = tsne.fit_transform(features)
        name = 't-SNE'
    elif method == 'pca':
        pca = PCA(n_components = n_components)
        components = pca.fit_transform(features)
        total_var = pca.explained_variance_ratio_.sum() * 100
        name = 'PCA'

    if n_components == 2:
        ### class person
        fig = px.scatter(components, x=0, y=1, color=person_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y'}
        )
        fig.update_layout(
            title={
                'text': f'{name} of class person',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig.show()
        ### class window
        fig = px.scatter(components, x=0, y=1, color=window_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y'}
        )
        fig.update_layout(
            title={
                'text': f'{name} of class window',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig.show()


    elif n_components == 3:
        ### class person
        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=person_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y', '2': 'z'}
        )
        fig.update_layout(
            title={
                'text': f'{name} of class person',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig.show()
        ### class window
        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=window_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y', '2': 'z'}
        )
        fig.update_layout(
            title={
                'text': f'{name} of class window',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig.show()
    else:
        print('Can only visualize in 2D and 3D !')
        return

    #####################################################################
    ### matplotlib for person class
    # plt.figure()
    # for label in range(2):
    #     indices = test_lbls[:,0]==label  ###test_lbls are 0s and 1s
    #     plt.scatter(components[indices,0],components[indices,1], label = person_dict[label])
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.legend()
    # plt.show()

    return components


# ****************************
# standardize function *****************
# *******************************

def standardize(x,mode=1):
    
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

def sliding(window,stride,features,targets,mode='end',start=0):

    x = []
    y = []
    count = 0
    i=start
    length = window + i
    size = features.shape[0]
    while  length <= size :
        x.append(features.iloc[i:length].values.tolist())
        if mode == 'start':
            y.append(targets.iloc[i])
        elif mode == 'end':
            y.append(targets.iloc[length-1])
        elif mode == 'mean':
            y.append(np.array(targets.iloc[i:length]).mean(axis=0).round())
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


def under_sample(data, size ,seq_len,stride,sliding_mode='end'):
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


        _x,_y = sliding(seq_len,stride,f,t,mode=sliding_mode)
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

# Ticket labels - List must be in alphabetical order
    if name == 'person':
        ax.xaxis.set_ticklabels(['No person','People']) ### names of classes starting from 0
        ax.yaxis.set_ticklabels(['No person','People'])
    elif name == 'window':
        ax.xaxis.set_ticklabels(['Closed','Open']) ### names of classes starting from 0
        ax.yaxis.set_ticklabels(['Closed','Open'])

## Display the visualization of the Confusion Matrix.
    plt.show()

###################################################################
### plot distribution of targets vs predictions in a given test set
def plot_distribution(y_true,y_pred,name):
    plt.figure(figsize=(10,4))

    if name == 'person': 
        plt.scatter(range(1,y_pred.shape[0]+1),y_pred,marker = '.', label='Predictions')
        plt.scatter(range(1,y_true.shape[0]+1),y_true,marker = '.',label='Targets')

        plt.yticks([0,1],['No person','People'])
    elif name == 'window': 
        plt.scatter(range(1,y_pred.shape[0]+1),y_pred,marker = '.', label='Predictions')
        plt.scatter(range(1,y_true.shape[0]+1),y_true,marker = '.',label='Targets')
        plt.yticks([0,1],['Closed','Open'])

    plt.title(f'distribution of wrong prediction of class {name}')
    plt.xlabel('Timeline')
    plt.legend()
    plt.show()

###################################################################
### plot distribution of FP, FN in a given test set

### false positives
def plot_fp_fn(y_true,y_pred,name):
    ### false positives
    fp = (y_true==False) & (y_pred==True)
    fp_indices = torch.where(fp == True)[0]
    fp = np.ones(len(fp_indices)) ## to plot it

    ### false negatives
    fn = (y_true==True) & (y_pred==False)
    fn_indices = torch.where(fn == True)[0]
    fn = np.zeros(len(fn_indices)) ## to plot it

    plt.figure(figsize=(10,4))
    
    plt.scatter(fp_indices+1,fp,marker = '.',c='blue')
    plt.scatter(fn_indices+1,fn,marker = '.',c='blue')

    plt.title(f'distribution of FP, FN of class {name}')
    plt.yticks([0,1],['FN','FP'])
    
    x1= (fp_indices+1).tolist()
    x2 = (fn_indices+1).tolist()

    if len(x1)!= 0 or len(x2)!=0:
        plt.xticks(range(min(x1+x2), max(x1+x2)+1, 1),labels =[])
        plt.xlabel('Timeline [frequency = 1]')

    plt.show()
    #################################################################################