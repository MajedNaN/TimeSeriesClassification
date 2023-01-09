from imports import *


######################################################################################
class Encoder(Module):
    def __init__ (self,c_in, embed_size = 2, hidden_size=100,nf=64,kernel_size=3, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, init_weights=True):

            
        self.conv = ConvBlock(c_in, nf,kernel_size,padding= 0,act = nn.LeakyReLU)
        self.rnn1 = nn.LSTM(nf, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.rnn2 = nn.LSTM((1+bidirectional)*hidden_size, embed_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=False)

                
        if init_weights: self.apply(self._weights_init)
        
    def _weights_init(self, m): 
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
    def forward (self, x):
        # x = self.encoder(x)
        x = self.conv(x)
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        x, _ = self.rnn1(x)
        x, (hn,_) = self.rnn2(x)
        # x = x[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        # x = out.view(out.shape[0],1,out.shape[1])  ### [batch_size x 1 x hidden_size * (1 + bidirectional)]
        return x   ### or return hn.transpose(1,0) -> shape (n_layers=1,embed_size)
#####################################################################################

class Decoder(Module):
    def __init__ (self, c_out,embed_size = 2, hidden_size=100,nf=64,kernel_size=3, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, init_weights=True):

            
        # self.conv_len = conv_len
        # self.c_out = c_out 
        # self.embed_size = embed_size
        self.rnn1 = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.rnn2 = nn.LSTM((1+bidirectional)*hidden_size, nf, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=False)
        self.conv = nn.ConvTranspose1d(nf, c_out,kernel_size)
        # self.linear = nn.Linear(hidden_size, c_out)
                
        if init_weights: self.apply(self._weights_init)
        
    def _weights_init(self, m): 
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
    def forward (self, x):

        #### if get last step only from encoder
        # x = x.repeat(self.conv_len,1)    ## to be passed to rnn as seq_len x n_vars "here embed_size"
        # x = x.view((-1,self.conv_len, self.embed_size))
        x, _ = self.rnn1(x)
        x, (hn,_) = self.rnn2(x)
        x = x.transpose(2,1)    # [batch_size x seq_len x n_vars] --> [batch_size x n_vars x seq_len] 
        x = F.leaky_relu(self.conv(x)) 
        return x
#########################################################################################
class AutoEncoder(Module):
  def __init__(self,c_in,embed_size = 2, hidden_size=100,nf=64,kernel_size=3, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, init_weights=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.encoder = Encoder(c_in,embed_size , hidden_size,nf,kernel_size, n_layers, bias, rnn_dropout, bidirectional, init_weights).to(device)
    self.decoder = Decoder(c_in,embed_size , hidden_size,nf,kernel_size, n_layers, bias, rnn_dropout, bidirectional, init_weights).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

    ####################################################################
class ShallowClassifier(Module):
    def __init__(self,c_in,c_out,n_neurons=100,fc_dropout=0):
        self.fc1 = nn.Linear(c_in, n_neurons, bias=True) 
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.linear = nn.Sequential(
                        self.fc1,
                        torch.nn.ReLU(),
                        torch.nn.Dropout(fc_dropout)
                        )
        self.fc2 = nn.Linear(n_neurons,c_out)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    def forward(self, x):
        x = x[:,-1]
        x = self.linear(x)
        out = self.fc2(x)
        return out
####################################################################################
class EncoderClassifier(Module):
    def __init__(self,c_in,c_out,encoder,n_neurons=100,fc_dropout=0,embed_size = 2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.body = encoder.to(device)
        self.head = ShallowClassifier(embed_size,c_out,n_neurons,fc_dropout).to(device)
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
############################################################################################
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
    def __init__(self, c_in, c_out, layers=[16,32], kss=[5,3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0],padding=(0,))
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
#####################################################################################
class MyFCN_body(MyFCN):
    def __init__(self, c_in, c_out, layers=[16,32], kss=[5,3]):
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0],padding=(0,))
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1],padding=(0,))
        # self.convblock3 = ConvBlock(layers[1], layers[2], kss[2],padding=(0,))
        self.gap = GAP1d(1)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        # x = self.convblock3(x)
        out = self.gap(x)
        return out
#####################################################################################
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
class LSTM_body(LSTM):
    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, init_weights=True):
        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        if init_weights: self.apply(self._weights_init)
        
    def _weights_init(self, m): 
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)

    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        return output
##################################################################################
class MyLSTM(LSTM):
    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        # return nn.Sigmoid()(output)
        return output
        
