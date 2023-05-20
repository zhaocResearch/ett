import torch
import torch.nn.functional as F
import argparse
bilstm=False
am= False
import math
class lstm(torch.nn.Module):

    def __init__(self, input_size=8, hidden_size=32, num_layers=2 , output_size=1 , dropout=0):
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
        self.fc = torch.nn.Linear(hidden_size, 1)


        self.rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 batch_first=True, dropout=self.dropout,bidirectional=bilstm )
        self.w_omiga = torch.randn(8,  hidden_size, 1, requires_grad=True).to(0)
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


    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)  # d_k为query的维度
        #print(d_k) 128

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        #         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x)

        return context, alpha_n

    def forward(self, x):
        #print("x.shape",x.shape)
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # print('out.shape',out.shape) #[8, 3, 128]
        # print('hidden',hidden.shape) #[2, 8, 128]
        # print('cell',cell.shape) #[2, 8, 128]

        #output = out.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]
        if am:
            query = out
            # 加入attention机制
            attn_output, alpha_n = self.attention_net(out, query)
        else:
            attn_output=out
        # print('attn_output',attn_output.shape)#[8, 128]
        # print('alpha_n', alpha_n.shape) #[8, 3, 3]


        out=attn_output[:,-1,:]#取最后一天的做预测
        # print(out.shape,"outshape1") #3,32
        out = torch.nn.functional.leaky_relu(self.linear1(out))
        # print(out.shape, "outshape2") #3,32
        out = self.linear2(out)
        #print(out.shape, "outshape3")#3,1
        return out