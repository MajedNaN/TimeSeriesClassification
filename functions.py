from imports import *
from archs import *

#******************************************

def smoothing_predictions(test_preds,error_max):
    '''Smoothing predictions for max error forward and backward, notice zero is passed first or one to other private function
    '''
    one = [1 for i in range(2)] 
    zero = [0 for i in range(2)]
    #### forward direction
    test_preds_smoothed = __smoothing_predictions(test_preds,one,error_max)
    test_preds_smoothed = __smoothing_predictions(test_preds_smoothed,zero,error_max)
    #### backward direction
    # test_preds_smoothed = __smoothing_predictions(np.flip(test_preds_smoothed),one,error_max)
    # test_preds_smoothed = np.flip(__smoothing_predictions(test_preds_smoothed,zero,error_max))
    return test_preds_smoothed

def __smoothing_predictions(test_preds,pattern:list,error_max):
    '''Smoothing predictions for max error 
    '''
    if type(test_preds)==torch.Tensor:
        test_preds_smoothed = test_preds.clone().numpy()
    else:
        test_preds_smoothed = test_preds.copy()


    for j in range(1,error_max):  ### error  in a row
        __pattern = pattern.copy() ### to be renewed next time
        for i in range(j):
            if pattern[0]==0: 
                __pattern.insert(-1,1) ### insert 1 before end  
            else:
                __pattern.insert(-1,0) ### insert 0 before end
        N = len(__pattern)

        possibles_person = np.where(test_preds_smoothed[:,0] == __pattern[0])[0]
        possibles_window = np.where(test_preds_smoothed[:,1] == __pattern[0])[0]

        solns_person = []
        for p in possibles_person:
            check = list(test_preds_smoothed[:,0][p:p+N])
            if np.all(check == __pattern):
                solns_person.append(p)
                
        solns_window = []
        for p in possibles_window:
            check = list(test_preds_smoothed[:,1][p:p+N])
            if np.all(check == __pattern):
                solns_window.append(p)

        idx = []
        for i in range(1,N): ### cause we have different error frames, to change in between error values
            if __pattern[i] != __pattern[0]:
                idx.append(i)

        for i in idx:
            if len(solns_person) >0:
                test_preds_smoothed[:,0][np.array(solns_person)+i] = __pattern[0]
            if len(solns_window) >0:
                test_preds_smoothed[:,1][np.array(solns_window)+i] = __pattern[0]

    return test_preds_smoothed

#******************************************
def encode_classes(data):
    '''Encode classes person and window into 0,1
    '''
    df = data.copy()
    le = LabelEncoder()
    labels = ['person','window_open']
    for label in labels:
        le.fit(df[label])
        df[label]=le.transform(df[label])
        print(df[label].value_counts())
    print(df.shape)
    return df
# *******************************
def plot_missing(df):
    '''plot missing values
    '''
    missing = df.isnull().sum(0).reset_index()
    missing.columns = ['column', 'count']

    missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]
    missing['percentage'] = missing['count'] / float(df.shape[0]) * 100
    ind = np.arange(missing.shape[0])
    

    fig, ax = plt.subplots()
    rects = ax.barh(ind, missing.percentage.values, color='r')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing.column.values, rotation='horizontal')
    ax.set_xlabel("Precentage of missing values %")
    # plt.grid(linestyle = '--')
    plt.show()
    print ('maximum number of missing values per column= ',missing['count'].max(),'\n')
    print(missing)
#**************************************
def impute_NaN(df):
    '''interpolate or drop NaN
    '''
    df_new = df.copy()
    list_parameters = [c for c in list(df_new.columns) if c != 'datetime']
    df_new.reset_index(drop=True,inplace=True)
    for column in list_parameters:
        df_temp = df_new.loc[:,column].interpolate(method='linear')
        df_new.loc[:,column] = df_temp.values
    # df_new.interpolate(method='polynomial', order=1,inplace=True) ## reset index in case of polynomial with degree > 1
    
    ## incase NaNs are at begining or end we drop those rows
    df_new.dropna(axis=0,inplace=True)
    df_new.reset_index(drop=True,inplace=True)
    return df_new

#*************************************************
def freeze(learn):
    '''
    freezing layers except head
    '''
    assert hasattr(learn.model, "head"), f"you can only use this with models that have .head attribute"
    for p in learn.model.parameters():
        p.requires_grad=False
    for p in learn.model.head.parameters():
        p.requires_grad=True
# *******************************
def unfreeze(learn):
    '''
    unfreezing layers 
    '''
    for p in learn.model.parameters():
        p.requires_grad=True

# *******************************
def predict_autoencoder(model, test_dls):
    '''
    predict autoencoder
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions, losses = [], []
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in test_dls:
            seq_true = seq_true[0].to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred)
            losses.append(loss.item())
        predictions = torch.cat(predictions).detach().cpu().numpy()
    return predictions, losses

# *******************************
def train_autoencoder(model, train_dls, val_dls, n_epochs,lr = 1e-3):
    '''
    train autoencoder
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)
    history = dict(train=[], val=[])
    best_model_wts = deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        batch_count = 0
        dls_length = len(train_dls)
        for seq_true in train_dls:
            optimizer.zero_grad()
            seq_true = seq_true[0].to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            batch_count += 1
            if batch_count % 1000 ==0 :
                print(f'epoch #: {epoch}, training batch #:{batch_count}/{dls_length}, train loss: {train_losses[-1]}')

        val_losses = []
        batch_count = 0
        dls_length = len(val_dls)
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dls:
                seq_true = seq_true[0].to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

                batch_count += 1
                if batch_count % 1000 == 0:
                    print(f'epoch #: {epoch}, validating batch #: {batch_count}/{dls_length}, val loss: {val_losses[-1]}')

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history

# *******************************
def normalized_kl(p,q,weight = 0.1): #normalized KL
    '''
    weighted normalized kullback leibler divergence
    '''
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

# *******************************
def plot_PR_curve_both(y_true,y_probas):
    '''
    Ploting PR curve function
    '''
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    class_name = 'person'
    precision, recall, thresholds_person = precision_recall_curve(y_true[:,0], y_probas[:,0])
    ### replace zero precision and zero recall at same time to avoid infinity fscore 
    zero_indices = np.where(((precision==0)==True) & ((recall==0)==True) == True)[0].tolist()
    precision[zero_indices] = 0.00001
    recall[zero_indices] = 0.00001
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix_person = np.argmax(fscore)
    print('class: {class_name}: Best Threshold =%f, F1-Score =%.3f' % (thresholds_person[ix_person], thresholds_person[ix_person]))

    ax1.plot(recall, precision, marker='.', label=f'PR-curve of class {class_name}')
    ax1.scatter(recall[ix_person], precision[ix_person], marker='o', color='red', label='best threshold')
    ax1.annotate(f'{thresholds_person[ix_person]:.4f}',xy = (recall[ix_person]-0.2, precision[ix_person]-0.1), color='red')
    # axis labels
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ###################################
    class_name = 'window'
    precision, recall, thresholds_window = precision_recall_curve(y_true[:,1], y_probas[:,1])
    ### replace zero precision and zero recall at same time to avoid infinity fscore 
    zero_indices = np.where(((precision==0)==True) & ((recall==0)==True) == True)[0].tolist()
    precision[zero_indices] = 0.00001
    recall[zero_indices] = 0.00001
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix_window = np.argmax(fscore)
    print('class: {class_name}: Best Threshold =%f, F1-Score =%.3f' % (thresholds_window[ix_window], thresholds_window[ix_window]))

    ax2.plot(recall, precision, marker='.', label=f'PR-curve of class {class_name}')
    ax2.scatter(recall[ix_window], precision[ix_window], marker='o', color='red', label='best threshold')
    ax2.annotate(f'{thresholds_window[ix_window]:.4f}',xy = (recall[ix_window]-0.2, precision[ix_window]-0.1), color='red')
    # axis labels
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()
    ########################
    # show the plot
    plt.show()
    return thresholds_person[ix_person], thresholds_window[ix_window]
# *******************************

def plot_PR_curve(class_name,y_true,y_probas):
    '''
    Ploting PR curve function
    '''
    class_name = class_name

    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

    ### replace zero precision and zero recall at same time to avoid infinity fscore 
    zero_indices = np.where(((precision==0)==True) & ((recall==0)==True) == True)[0].tolist()
    precision[zero_indices] = 0.00001
    recall[zero_indices] = 0.00001
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


# *******************************
def visualize_embeddibgs(network,features, person_labels, window_labels,features_unannotated=None,unannotated_labels = None,combined=False,n_components = 2, method = 'pca'):
    '''
    visualize embeddings using sklearn and plotly
    '''
    label = ''

    ### normalize
    __features = StandardScaler().fit_transform(features)
    if features_unannotated is not None:
        __features_unannotated = StandardScaler().fit_transform(features_unannotated)

    if method == 'tsne':
        tsne = TSNE(n_components = n_components)
        components = tsne.fit_transform(__features)
        if features_unannotated is not None:
            components_unannotated = tsne.fit_transform(__features_unannotated)
        name = 't-SNE'
    elif method == 'pca':
        pca = PCA(n_components = n_components)
        components = pca.fit_transform(__features)
        total_var = pca.explained_variance_ratio_.sum() * 100
        if features_unannotated is not None:
            components_unannotated = pca.fit_transform(__features_unannotated)
        name = 'PCA'

    if n_components == 2:
        ### class person
        fig1 = px.scatter(components, x=0, y=1, color=person_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
                    labels={'0': 'x', '1': 'y'})
        ### class window
        fig2 = px.scatter(components, x=0, y=1, color=window_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y'})
    elif n_components == 3:
        ### class person
        fig1 = px.scatter_3d(
            components, x=0, y=1, z=2, color=person_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y', '2': 'z'})
        ### class window
        fig2 = px.scatter_3d(
            components, x=0, y=1, z=2, color=window_labels,
            # title=f'Total Explained Variance: {total_var:.2f}',
            labels={'0': 'x', '1': 'y', '2': 'z'})
    else:
        print('Can only visualize in 2D and 3D !')
        return
    if features_unannotated is not None:
        if n_components == 2:
            fig3 = px.scatter(components_unannotated, x=0, y=1, color=unannotated_labels, 
                            labels={'0': 'x', '1': 'y'}).update_traces(marker=dict(color='orange'))
        elif n_components == 3:
            fig3 = px.scatter_3d(components_unannotated, x=0, y=1,z=2, color=unannotated_labels, 
                            labels={'0': 'x', '1': 'y', '2': 'z'}).update_traces(marker=dict(color='orange'))
        if combined:
            fig1 = go.Figure(data = fig1.data + fig3.data)
            fig2 = go.Figure(data = fig2.data + fig3.data)
            label = 'and unannotated data'
        else:

            fig3.update_layout(
            title={
                'text': f'{name} of class Ununnotated data for {network}',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
            fig3.show()

    fig1.update_layout(
        title={
            'text': f'{name} of class person {label} for {network}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig1.show()
    
    fig2.update_layout(
        title={
            'text': f'{name} of class window {label} for {network}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig2.show()
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


# *******************************

def standardize(x,mode=1):
    '''
    standardize function
    '''

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
    '''
    standardize with parameters
    '''

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

# *******************************

def sliding(window,stride,features,targets=None,mode='end',start=0):
    '''
    sliding function
    '''
    x = []
    y = []
    count = 0
    i=start
    length = window + i
    size = features.shape[0]
    while  length <= size :
        x.append(features.iloc[i:length].values.tolist())
        if targets is not None:
            if mode == 'start':
                y.append(targets.iloc[i])
            elif mode == 'end':
                y.append(targets.iloc[length-1])
            elif mode == 'mean':
                y.append(np.array(targets.iloc[i:length]).mean(axis=0).round())
        i += stride
        length = window + i
        count += 1
    
    if len(x)==0: 
        x = np.array(x,dtype=np.float32).reshape((count,features.shape[1], window))
    else:
        x = np.array(x,dtype=np.float32)#.reshape((count,features.shape[1], window))
        x = np.transpose(x, (0, 2, 1)) ## to have correct segmentation shape (#samples,#features,seq_len)

    ### to be compatible with torch
    if targets is not None:
        try:
            sh = targets.shape[1] ### more than 1 feature (has 2nd dimension)
        except:
            sh = 1
        y = np.array(y,dtype=np.float32).reshape(-1,sh)
        return x,y
    return x

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

# *******************************

def under_sample(data, size ,seq_len,stride,sliding_mode='end'):
    '''
    under sampling function
    '''

    x_list = []
    y_list = []

    __data = data.copy()
    __data.reset_index(drop=True,inplace = True)
    lc = __data[(__data.loc[:,'person']>0) | (__data.loc[:,'window_open']>0)]
    indices = lc.index

    while (len(indices) > 0): 
        left = indices[0] # choose the most left index contains 1
        if left - size < 0: # outside values to avoid empty list from left
            window = __data.iloc[:left+size+1]
        else:
            window = __data.iloc[left-size : left+size+1]
        #get features,targets for sliding
        f = window.drop(columns = ['person','window_open'])
        t = window.filter(['person','window_open'])      


        _x,_y = sliding(seq_len,stride,f,t,mode=sliding_mode)
        x_list.append(_x)
        y_list.append(_y)
        
        #drop selected window values, if count < window, finished undersampling
        try:
            __data = __data.drop(index = range(0,left+size+1))
        except:
            break
        __data.reset_index(drop=True,inplace = True)
        
        #recalculate indices of 1s
        lc = __data[(__data.loc[:,'person']>0) | (__data.loc[:,'window_open']>0)]
        indices = lc.index
        
    # convert lists to correct shape  for segmentation and standardization
    X = x_list[0]
    y = y_list[0]

    for i in range(1,len(x_list)):
        X= concat(X,x_list[i])
        y= concat(y,y_list[i])

    del __data
    return X,y
    
# *******************************

def plot_confusion(y_true,y_pred,n_classes,name):
    '''
    plot confusion matrix function
    '''

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

#******************************************

def plot_confusion_both(y_true,y_pred,n_classes):
    '''plot confusion matrix function for both classes
    '''
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))

    cf_matrix_person=confusion_matrix(y_true[:,0],y_pred[:,0])
    cf_matrix_window=confusion_matrix(y_true[:,1],y_pred[:,1])

    ## person
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix_person.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix_person.flatten()/np.sum(cf_matrix_person)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(n_classes,n_classes)  # 2: number of classes

    sns.heatmap(cf_matrix_person, annot=labels, fmt='', cmap='Blues',cbar = False, ax =ax1)
    ax1.set_title(f'Confusion Matrix of class person\n')
    ax1.set_xlabel('\nPredicted Classes')
    ax1.set_ylabel('Actual Classes ')

# Ticket labels - List must be in alphabetical order
    ax1.xaxis.set_ticklabels(['No person','People']) ### names of classes starting from 0
    ax1.yaxis.set_ticklabels(['No person','People'])


    ### window

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix_window.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix_window.flatten()/np.sum(cf_matrix_window)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(n_classes,n_classes)  # 2: number of classes

    sns.heatmap(cf_matrix_window, annot=labels, fmt='', cmap='Blues',cbar = False,ax =ax2)
    ax2.set_title(f'Confusion Matrix of class window\n')
    ax2.set_xlabel('\nPredicted Classes')
    ax2.set_ylabel('Actual Classes ')

# Ticket labels - List must be in alphabetical order
    ax2.xaxis.set_ticklabels(['Closed','Open']) ### names of classes starting from 0
    ax2.yaxis.set_ticklabels(['Closed','Open'])



## Display the visualization of the Confusion Matrix.
    plt.show()
#******************************************
def plot_distribution(y_true,y_pred,name):
    '''
    plot distribution of targets vs predictions in a given test set
    '''

    plt.figure(figsize=(10,4))

    if name == 'person': 
        plt.scatter(range(1,y_pred.shape[0]+1),y_pred,marker = '.', label='Predictions')
        if y_true is not None:
            plt.scatter(range(1,y_true.shape[0]+1),y_true,marker = '.',label='Targets')
        plt.yticks([0,1],['No person','People'])
    elif name == 'window': 
        plt.scatter(range(1,y_pred.shape[0]+1),y_pred,marker = '.', label='Predictions')
        if y_true is not None:
            plt.scatter(range(1,y_true.shape[0]+1),y_true,marker = '.',label='Targets')
        plt.yticks([0,1],['Closed','Open'])

    plt.title(f'distribution of predictions of class {name}')
    plt.xlabel('Timeline')
    plt.legend()
    plt.show()
#*************************************************

def plot_distribution_both(y_true,y_pred,network):
    '''
    plot distribution of targets vs predictions in a given test set
    '''

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(15,8))
    class_name = 'person'
    ax1.plot(range(1,y_pred.shape[0]+1),y_pred[:,0], label='Predictions')
    if y_true is not None:
        ax1.plot(range(1,y_true.shape[0]+1),y_true[:,0],label='Targets')
    ax1.set_yticks([0,1],['No person','People'])
    ax1.set_title(f'Distribution of predictions for {network}')
    ax1.set_xlabel('Timeline')
    ax1.legend()

    class_name = 'window'
    ax2.plot(range(1,y_pred.shape[0]+1),y_pred[:,1], label='Predictions')
    if y_true is not None:
        ax2.plot(range(1,y_true.shape[0]+1),y_true[:,1],label='Targets')
    ax2.set_yticks([0,1],['Closed','Open'])
    ax2.set_xlabel('Timeline')
    ax2.legend()


    plt.show()
#*************************************************

def plot_fp_fn(y_true,y_pred,name):
    '''plot distribution of FP, FN in a given test set
    '''

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
    #*******************************************
