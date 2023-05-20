import math
import torch
from torch import nn
import torch.nn.functional as F

t2v_hidden_dim=128

# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         # pe.requires_grad = False
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Time2Vec(nn.Module):
    def __init__(self, activation, hidden_dim=32, batch_size=1, sentence_len=5, in_features=13, out_features_end=13):
        '''

        :param activation: 激活函数（非线性激活函数） sin/cos
        :param hidden_dim: 隐藏（自定义，不影响运行）
        '''
        super(Time2Vec, self).__init__()
        if activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos
        self.out_features = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, out_features_end)

        # 获取x的尺寸信息
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.in_features = in_features
        num=1
        # 初始化权重和偏置
        self.w0 = nn.parameter.Parameter(torch.zeros(batch_size, in_features, 1)*num)
        self.b0 = nn.parameter.Parameter(torch.zeros(batch_size, sentence_len, 1)*num)
        self.w = nn.parameter.Parameter(torch.zeros(batch_size, in_features, self.out_features - 1)*num)
        self.b = nn.parameter.Parameter(torch.zeros(batch_size, sentence_len, self.out_features - 1)*num)
        self.cont=0

    def forward(self, x):

        # 运算
        #print('xwb,xw.shape', x.shape, self.w.shape,self.b.shape,torch.matmul(x, self.w).shape)
        v1 = self.activation(torch.matmul(x, self.w) + self.b)
        self.cont=self.cont+1
        if self.cont%1000==0:
            print(self.cont)
        if self.cont==30000:
            torch.save(self.w, 'selfw')
            print(self.w)

        v2 = torch.matmul(x, self.w0) + self.b0
        v3 = torch.cat([v1, v2], -1)
        # print('v1,v2,v3.shape',v1.shape,v2.shape,v3.shape)
        x = self.fc1(v3)
        return x


class TransAm(nn.Module):
    def __init__(self, d_model=32, n_head=4, dim_feedforward=64, dropout=0.1,
                 activation='relu', batch_first=True, num_layers=6, output_size=1, predays=5,
                 batch_size=1, sentence_len=2, mdevice= 'cuda:0', feature_size=3, mode='ett'):


        super(TransAm, self).__init__()
        if mode=='ett':
            self.time2vec_use = True  # 是否要用T2V
        else:
            self.time2vec_use = False  # 是否要用T2V
        self.device = mdevice
        self.d_model = d_model  # 输入统一为这个格式 因为nhead要被这个整除，不然会报错
        self.num_layers = num_layers  # 整个encoder要重复几次
        self.n_head = n_head  # 多头有几个头,
        self.dim_feedforward = dim_feedforward  # the dimension of the feedforward network model (default=2048).
        self.model_type = 'Transformer'
        self.output_size = output_size  # 输出维度
        self.dropout = dropout  # the dropout value (default=0.1)
        self.src_mask = None
        self.batch_first = batch_first
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.feature_size = feature_size
        self.enc_input_fc = nn.Linear(self.feature_size, self.d_model)
        #print('d_model',d_model,'dim_feedforward',dim_feedforward,'',,)
        # self.encoder = nn.Embedding(feature_size, self.d_model)#官网上这么用
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.activation = activation  # the activation function of intermediate layer, relu or gelu(default=relu).

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,norm=nn.BatchNorm1d(10, eps=1e-10))
                                                            # norm=nn.BatchNorm1d(10, eps=1e-10)
                                                         #norm=nn.LayerNorm(normalized_shape=d_model, eps=1e-10)


        self.decoder = nn.Linear(self.d_model, 16)
        self.tanh=torch.nn.Tanh()
        self.linear1 = torch.nn.Linear(16 , self.output_size)
        self.bn=torch.nn.BatchNorm1d(16)
        self.init_weights()
        if self.time2vec_use:
            # self.time2vec = Time2Vec("sin", hidden_dim=self.feature_size, batch_size=self.batch_size,
            #                          sentence_len=self.sentence_len, in_features=self.feature_size,
            #                          out_features_end=self.feature_size)
            self.time2vec = Time2Vec("sin", hidden_dim=t2v_hidden_dim, batch_size=self.batch_size,
                                     sentence_len=self.sentence_len, in_features=self.d_model,
                                     out_features_end=self.d_model)


        # 自己想的：输入输出要一样才能加在一起，不然像原文那样只加两个，我没试过

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        seq_len = src.shape[1]
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            device = src.device
            mask = self._generate_square_subsequent_mask(seq_len).to(device)
            self.src_mask = mask


        # print('src.shape1', src.shape)  # 1，5，13
        # print('time2vec(src).shape1', self.time2vec(src).shape)  # 1，5，13
        # if time2vec_use:
        #     src = src + 0.001*self.time2vec(src)

        # print(self.time2vec(src).shape)
        #src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.enc_input_fc(src)* math.sqrt(self.d_model)  # 不管进来多少，最后都改成32   1，5，32
        # print('src.shape2', src.shape)
        # src = self.encoder(src) * math.sqrt(self.d_model)#官网上这么用

        src = self.pos_encoder(src)  # 加上位置信息
        # print(src.shape)
        # print(self.time2vec(src).shape)
        if self.time2vec_use:
            src = src + self.time2vec(src)
        # print('src.shape3', src.shape)

        # batch_sizex句长x特征数量
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        # print('output.shape1', output.shape)  # 这里和输入一样，因为encode层要保证输入输出一样1，5，32

        output=output[:,-1,:]

        #print(output.shape)#59,128
        output =(self.decoder(output))  # 1，5，16
        #output=self.bn(output)

        #output = output.reshape(output.shape[0], -1)  # 1，80
        # print('output.shape',output.shape)
        output = (self.linear1(output))  # 1，1
        # print('output.shape2', output.shape)
        return output

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    #     return mask

    def _generate_square_subsequent_mask(self,sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.zeros(sz, sz)
        #return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
