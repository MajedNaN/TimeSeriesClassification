from imports import *
from functions import *
#### ******************************************
###### pytorch Dataset and Dataloaders
### *************************

# tfms = transforms.Compose(
#     [transforms.ToTensor()])
#      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class MyDataSet(Dataset):
    def __init__(self,X, y, transform=None):
        self.n_samples = X.shape[0]
        self.x = X
        self.y = y
        self.transform = transform
    def __getitem__(self, index: Any):
        sample = self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.n_samples
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MyLearner:
    def __init__(self,x_train,y_train,x_valid,y_valid,model,bs = 64, num_workers = 0,lr=0.001,tfms = None,epochs = 10,device = None,loss_func=nn.BCELoss,optimizer=optim.Adam):
        
        # X_train,p1,p2 = standardize(X[splits[0]],stand_mode)
        # if stand_mode == 1:
        #     X_valid = standardize_with_params(X[splits[1]],mean=p1,std=p2)
        # elif stand_mode == 2:
        #     X_valid = standardize_with_params(X[splits[1]],min=p1,max=p2)
        
        self.tsets = MyDataSet(x_train, y_train, transform=tfms) #train dataset
        self.vsets = MyDataSet(x_valid, y_valid, transform=tfms) #valid dataset
        self.tls = DataLoader(self.tsets, batch_size=bs, shuffle=False, num_workers=num_workers)
        self.vls = DataLoader(self.vsets, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        self.num_workers = num_workers
        self.bs = bs

        if device == None: 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)

        self.criterion = loss_func()
        self.optimizer = optimizer(self.model.parameters(), lr)
    
    def fit(self,epochs=10):
        tloss_list = []
        vloss_list = []
        acc_list = []
        for epoch in range(epochs):

            self.model.train()
            train_loss =0
            for inputs, labels in self.tls:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = loss.item()

            self.model.eval()
            valid_loss =0
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
            
                for inputs, labels in self.vls:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                # max returns (value ,index)
                    predicted = torch.round(outputs)
                    n_samples += labels.shape[0]
                    n_correct += torch.all((predicted == labels),dim=1).sum() # for entire row accuracy use torch.all(....)
                    vloss = self.criterion(outputs, labels)
                valid_loss = vloss.item()
            acc = 100.0 * n_correct / n_samples # for entire row accuracy divide by double of samples
            acc_list.append(acc.cpu())
            tloss_list.append(loss.item())
            vloss_list.append(vloss.item())
            print (f'Epoch [{epoch+1}], Training_Loss: {train_loss:.4f},valid_Loss: {valid_loss:.4f}, valid_accuracy: {acc:.4f}%')

        print('Finished Training')
        # PATH = './cnn.pth'
        # torch.save(model.state_dict(), PATH)
        x_axis = range(1,epochs+1)
        plt.figure()
        plt.plot(x_axis,vloss_list,label='valid_loss')
        plt.plot(x_axis,tloss_list,label='train_loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show 
        plt.figure()
        plt.plot(x_axis,acc_list,label='valid_acc')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
    def eval(self,X,y):
        
        #for person
        yp_true = []
        yp_pred = []

        # for window
        yw_true = []
        yw_pred = []

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            test_set = MyDataSet(X,y)
            test_ls = DataLoader(test_set,batch_size=self.bs, shuffle=False, num_workers=self.num_workers)
            for inputs, labels in test_ls:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                predicted = torch.round(outputs)
                n_samples += labels.shape[0]
                n_correct += torch.all((predicted == labels),dim=1).sum().item()
            
                # for confusion matrix

                yp_pred.append(predicted[:,0].cpu().numpy())
                yp_true.append(labels[:,0].cpu().numpy())

                yw_pred.append(predicted[:,1].cpu().numpy())
                yw_true.append(labels[:,1].cpu().numpy())

                # for i in range(labels.shape[0]):
                #     if torch.unique(labels[i]).size(0)==1:
                #          y_true.append(1.0)
                #     else:
                #         y_true.append(0.0)
                #     if torch.unique(predicted[i]).size(0)==1:
                #          y_pred.append(1.0)
                #     else:
                #         y_pred.append(0.0)
                        
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy: {acc:.4f}%')

        yp_pred = [item for sublist in yp_pred for item in sublist]
        yp_true = [item for sublist in yp_true for item in sublist]

        yw_pred = [item for sublist in yw_pred for item in sublist]
        yw_true = [item for sublist in yw_true for item in sublist]

        ### person class
        ### number of classes in validation set
        n_classes = len(set(yp_true))
        plot_confusion(yp_true,yp_pred,n_classes,name = 'person')
        print(f'f1 score: {f1_score(yp_true,yp_pred)}\n') # to get f1 score for each class use (average=None)


        ### window class
        ### number of classes in validation set
        n_classes = len(set(yw_true))
        plot_confusion(yw_true,yw_pred,n_classes,name = 'window')
        print(f'f1 score: {f1_score(yw_true,yw_pred)}\n') # to get f1 score for each class use (average=None)


