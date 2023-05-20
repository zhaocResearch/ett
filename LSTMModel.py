import torch

class lstm(torch.nn.Module):

    def __init__(self, input_size=8, hidden_size=32, num_layers=2 , output_size=1 , dropout=0, mode='lstm'):
        """

        :type num_layers: int
        """
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        if mode=='lstm':
            bilstm = False
        else:
            bilstm = True



        self.rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 batch_first=True, dropout=self.dropout,bidirectional=bilstm )
        # torch.nn.GRU

        # 线性层
        if bilstm:
            self.linear1 = torch.nn.Linear(self.hidden_size*2, 32)
        else:
            self.linear1 = torch.nn.Linear(self.hidden_size, 32)
        self.linear2 = torch.nn.Linear(32, self.output_size)
        # print('inputsize',self.inputsize)


        # ReLU 层
        self.relu = torch.nn.ReLU()

        # dropout 层，这里的参数指 dropout 的概率
        # self.dropout = torch.nn.Dropout(0.1)
        # self.dropout2 = torch.nn.Dropout(0.1)

    def forward(self, x):
        #print("x.shape",x.shape)
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size

        out=out[:,-1,:]#取最后一天的做预测
        # print(out.shape,"outshape1") #3,32
        out = torch.nn.functional.leaky_relu(self.linear1(out))
        # print(out.shape, "outshape2") #3,32
        out = self.linear2(out)
        #print(out.shape, "outshape3")#3,1
        return out