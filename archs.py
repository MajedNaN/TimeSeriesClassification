from imports import *
import torch.nn.functional as F

class ConvNet(Module):
    
    def __init__(self,n_features,n_classes):
        super(ConvNet,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, 32, kernel_size=3, stride=1,padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Dropout(0.3))

        self.fc1 = torch.nn.Linear(6 * 32, 10, bias=True) ## 15 (window size) - 2 (cause of filter) / 2 (pooling) = 6
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer2 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.7))
        self.fc2 = nn.Linear(10,n_classes)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
       
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.flatten(x,start_dim=1)
        x = self.layer2(x)
        out = self.fc2(x)
        out = nn.Sigmoid()(out) # do not use if loss function includes the sigmoid
        return out


###################################################################################3

class MyFCN(FCN):
    def __init__(self, c_in, c_out, layers=[16,16], kss=[3,2]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0],padding=(1,))
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1],padding=(0,))
        # self.convblock3 = ConvBlock(layers[1], layers[2], kss[2],padding=(0,))

        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        # x = self.convblock3(x)
        x = self.gap(x)
        out = self.fc(x)
        # out = nn.Sigmoid()(out) # do not use if loss function includes the sigmoid
        return out

# class MyFCN(FCN):
#     def __init__(self,c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
#         # super(MyFCN,self).__init__(c_in, c_out, layers, kss)
#         assert len(layers) == len(kss)
#         self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
#         self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
#         self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(layers[-1], c_out)
#     def forward(self, x):
#         x = self.convblock1(x)
#         x = self.convblock2(x)
#         x = self.convblock3(x)
#         x = self.gap(x)
#         x = torch.flatten(x,start_dim=1)
#         out = self.fc(x)
#         out = nn.Sigmoid()(out) # do not use if loss function includes the sigmoid
#         return out
###################################################################################3

class MyLSTM(LSTM):
    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        # return nn.Sigmoid()(output)
        return output
        
