from torch.autograd import Variable
import torch.nn as nn
import torch
import transformerModel
import LSTMModel
import LSTM_AM_Model
import CNNModel
import TCNModel
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
import os

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./logs")

parser = argparse.ArgumentParser()
dataset_name = '沪深300'  # 使用哪个数据集 中证100 沪深300 test
ett=False
# 常改动参数
parser.add_argument('--mode', default='ett')  # 使用哪种模型 transformer lstm bilstm cnn ett lstm_am
parser.add_argument('--gpu', default=0, type=int)  # gpu 卡号
parser.add_argument('--epochs', default=50, type=int)  # 训练轮数
parser.add_argument('--lr', default=0.001, type=float)  # learning rate 学习率
# parser.add_argument('--pre_days', default=3, type=int)  # sequence的长度，默认是用前五天的数据来预测下一天的收盘价,改成去取了
parser.add_argument('--batch_size', default=180, type=int)#中证59 沪深180
parser.add_argument('--test_batch_size', default=180, type=int)
parser.add_argument('--useGPU', default=True, type=bool)  # 是否使用GPU
#parser.add_argument('--batch_first', default=True, type=bool)  # 是否将batch_size放在第一维(全True了）
parser.add_argument('--dropout', default=0.1, type=float)  # 默认0代表不用做dropout
parser.add_argument('--models_path', default='Models')  # 模型保存位置
# parser.add_argument('--show_log', default=False)  # 展示训练时候的损失变化
parser.add_argument('--parameters_path',
                    default='D:/yian/机器学习/量化交易/all_parameters/' + dataset_name + '_parameters')  # 股票常数的路劲
parser.add_argument('--data_path',
                    default='D:/yian/机器学习/量化交易/all_data/train_test_split/' + dataset_name)  # 股票数据的路径
parser.add_argument('--loss_path', default='Loss')  # 每个股票训练时候的损失
parser.add_argument('--show_pic', default=False)  # 展示最终的预测和实际比较折线图
parser.add_argument('--save_predict', default=True)  # 保存预测后的数据

# lstm
parser.add_argument('--layers', default=4, type=int)  # LSTM层数
parser.add_argument('--hidden_size', default=128, type=int)  # 隐藏层的维度

# transformer
parser.add_argument('--d_model', default=256)  # 把输入向量统一映射成这个维度（不然有可能n_heads除不尽）
parser.add_argument('--n_head', default=16)  # 多头有几个头,
parser.add_argument('--dim_feedforward', default=2048)  # the dimension of the feedforward network model (default=2048).
parser.add_argument('--activation', default='relu')  # 激活函数 of intermediate layer, relu or gelu leaky_relu(default=relu).
parser.add_argument('--num_layers', default=4)  # 整个encoder要重复几次
parser.add_argument('--output_size', default=1)  # 输出维度
parser.add_argument('--transfer_learning', default=False)  # 是否开启迁移学习
parser.add_argument('--transformer_pretrained_path', default='Models/transformer_pretrained.pt')  # 开启迁移学习则加载
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device



def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


my_mkdir(args.models_path)
my_mkdir(args.loss_path)


class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform is not None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true + 0.000001)))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def root_mean_square_error(y_test, y_predict):
    y_rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return y_rmse


def train_cnn(train_loader, save_model=False,batch_size=5, sentence_len=6, feature_size=10):
    model = CNNModel.cnn(input_size=feature_size, output_size=1 , pre_days=sentence_len, feature_size=feature_size)
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001
    #optimizer = torch.optim.SGD(model.parameters(), momentum=0.8, lr=args.lr)  # Adam梯度下降  学习率=0.0001
    epoch_loss_list = []
    epoch_loss_test_list = []
    min_loss = 100
    for i in range(1, args.epochs + 1):
        total_loss  = 0
        preds = []
        y_true = []
        for idx, (inputs, labels) in enumerate(train_loader):
            #print("inputs.shape",inputs.shape)
            if args.useGPU:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32).cuda()  # cnn只能接受float 不然会报错
                labels = labels.to(torch.float32).cuda()  # cnn只能接受float 不然会报错
                outputs = model(inputs)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32)  # cnn只能接受float 不然会报错
                labels = labels.to(torch.float32)  # cnn只能接受float 不然会报错
                outputs = model(inputs)

            outputs = outputs.ravel()
            # print("shape:", outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            preds.extend(outputs)
            y_true.extend(labels)
            # writer.add_scalar(str(codes_name), total_loss , i)
        cnn_mse_train = mean_squared_error(np.array(y_true), np.array(preds))
        epoch_loss_list.append(cnn_mse_train)


        preds = []
        y_true = []

        for idx, (x_test_now, label) in enumerate(test_loader):

            if args.useGPU:
                x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
            x_test_now = x_test_now.to(torch.float32)

            pred = model(x_test_now).ravel()  # 出来的是3，1
            # list = pred.data.tolist()

            preds.extend(pred.data.tolist())
            y_true.extend(label)



        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(np.array(y_true), np.array(preds))
        rmse = root_mean_square_error(y_true, preds)
        epoch_loss_test_list.append(mse)

        print(f'epoch:{i:3d} | mse_loss_training:{cnn_mse_train:6.6f} | mse_loss_test:{mse:6.6F}')
        print(f'测试集上：R2:{r2:6.4f} | MSE:{mse:6.6F}\
                | MAE:{mae:6.6F} | RMSE:{rmse:6.6F}')

        # torch.save(model, args.models_path)
        # writer.add_graph(model=model, input_to_model=inputs)
        # torch.save(model, args.models_path)
        if save_model:
            if mse  < min_loss:
                min_loss = mse
                print('save')
                torch.save(model, args.models_path + "/" +args.mode+ ".pt")

                tmp_best_p = {'epoch': i, 'R2': r2, 'MSE': mse,
                              'MAE': mae, 'RMSE': rmse}

                # tmp_model_state_dict = model.state_dict().copy()
                # torch.save({'state_dict': tmp_model_state_dict}, args.models_path + "/LSTM" + ".pt")


    return epoch_loss_list,epoch_loss_test_list, tmp_best_p


def predict_cnn(test_loader, show_pic=False, batch_size=1, sentence_len=2, feature_size=3):
    model = torch.load(args.models_path + "/" +args.mode+ ".pt")


    preds = []
    y_true = []
    for idx, (x_test_now, label) in enumerate(test_loader):

        if args.useGPU:
            x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
        x_test_now = x_test_now.to(torch.float32)

        pred = model(x_test_now).ravel()  # 出来的是3，1
        # list = pred.data.tolist()

        preds.extend(pred.data.tolist())
        y_true.extend(label)
    # cnn_mape=mape(np.array(y_true), np.array(preds))
    cnn_r2 = r2_score(y_true, preds)
    cnn_mae = mean_absolute_error(y_true, preds)
    cnn_mse = mean_squared_error(np.array(y_true), np.array(preds))
    cnn_rmse = root_mean_square_error(y_true, preds)
    print(f'R2:{cnn_r2:6.4f} |MSE:{cnn_mse:6.4F} |RMSE:{cnn_rmse:6.4F} |MAE:{cnn_mae:6.4F}')
    # print(len(preds))
    # print(len(y_true))


    if show_pic:
        plt.figure(figsize=(12, 8))
        plt.plot(preds, 'r', label='y_test_predict')
        plt.plot(y_true, 'b', label='y_test_true')
        plt.legend()
        plt.show()
    return preds


def train_lstm(train_loader, save_model=False):
    model = LSTMModel.lstm(input_size=feature_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
                           dropout=args.dropout,mode=args.mode)
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001
    #optimizer = torch.optim.SGD(model.parameters(),momentum=0.78, lr=args.lr)  # Adam梯度下降  学习率=0.0001
    epoch_loss_list = []
    epoch_loss_test_list = []
    min_loss = 100
    for i in range(1, args.epochs + 1):
        total_loss  = 0
        preds = []
        y_true = []
        for idx, (inputs, labels) in enumerate(train_loader):
            #print("inputs.shape",inputs.shape)
            if args.useGPU:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32).cuda()  # lstm只能接受float 不然会报错
                labels = labels.to(torch.float32).cuda()  # lstm只能接受float 不然会报错
                outputs = model(inputs)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32)  # lstm只能接受float 不然会报错
                labels = labels.to(torch.float32)  # lstm只能接受float 不然会报错
                outputs = model(inputs)

            outputs = outputs.ravel()
            # print("shape:", outputs.shape, labels.shape)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            preds.extend(outputs)
            y_true.extend(labels)
            # writer.add_scalar(str(codes_name), total_loss , i)
        lstm_mse_train = mean_squared_error(np.array(y_true), np.array(preds))
        epoch_loss_list.append(lstm_mse_train)




        preds = []
        y_true = []

        for idx, (x_test_now, label) in enumerate(test_loader):

            if args.useGPU:
                x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
            x_test_now = x_test_now.to(torch.float32)

            pred = model(x_test_now).ravel()  # 出来的是3，1
            # list = pred.data.tolist()

            preds.extend(pred.data.tolist())
            y_true.extend(label)



        lstm_r2 = r2_score(y_true, preds)
        lstm_mae = mean_absolute_error(y_true, preds)
        lstm_mse = mean_squared_error(np.array(y_true), np.array(preds))
        lstm_rmse = root_mean_square_error(y_true, preds)
        epoch_loss_test_list.append(lstm_mse)

        print(f'epoch:{i:3d} ')
        print(f'训练集上:MSE:{lstm_mse_train:6.6f}')
        print(f'测试集上:MSE:{lstm_mse:6.6F} |R2:{lstm_r2:6.4f} |  MAE:{lstm_mae:6.6F} | RMSE:{lstm_rmse:6.6F}')

        # torch.save(model, args.models_path)
        # writer.add_graph(model=model, input_to_model=inputs)
        # torch.save(model, args.models_path)
        if save_model:
            if lstm_mse  < min_loss:
                min_loss = lstm_mse
                print('save')
                torch.save(model, args.models_path + "/" +args.mode+ ".pt")

                tmp_best_p = {'epoch': i, 'R2': lstm_r2, 'MSE': lstm_mse,
                              'MAE': lstm_mae, 'RMSE': lstm_rmse}

                # tmp_model_state_dict = model.state_dict().copy()
                # torch.save({'state_dict': tmp_model_state_dict}, args.models_path + "/LSTM" + ".pt")


    return epoch_loss_list,epoch_loss_test_list, tmp_best_p


def predict_lstm(test_loader, show_pic=False, ):
    model = torch.load(args.models_path + "/" +args.mode+ ".pt")
    # model = LSTMModel.lstm(input_size=feature_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
    #                        dropout=args.dropout)
    # model.to(args.device)
    # checkpoint = torch.load(args.models_path + "/LSTM" + ".pt")
    # model.load_state_dict(checkpoint['state_dict'])

    preds = []
    y_true = []
    for idx, (x_test_now, label) in enumerate(test_loader):

        if args.useGPU:
            x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
        x_test_now = x_test_now.to(torch.float32)

        pred = model(x_test_now).ravel()  # 出来的是3，1
        # list = pred.data.tolist()

        preds.extend(pred.data.tolist())
        y_true.extend(label)
    # lstm_mape=mape(np.array(y_true), np.array(preds))
    lstm_r2 = r2_score(y_true, preds)
    lstm_mae = mean_absolute_error(y_true, preds)
    lstm_mse = mean_squared_error(np.array(y_true), np.array(preds))
    lstm_rmse = root_mean_square_error(y_true, preds)
    print(f'R2:{lstm_r2:6.4f} |MSE:{lstm_mse:6.4F} |RMSE:{lstm_rmse:6.4F} |MAE:{lstm_mae:6.4F}')
    # print(len(preds))
    # print(len(y_true))


    if show_pic:
        plt.figure(figsize=(12, 8))
        plt.plot(preds, 'r', label='y_test_predict')
        plt.plot(y_true, 'b', label='y_test_true')
        plt.legend()
        plt.show()
    return preds


def train_lstm_am(train_loader, save_model=False):
    model = LSTM_AM_Model.lstm(input_size=feature_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
                           dropout=args.dropout)
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001
    #optimizer = torch.optim.SGD(model.parameters(),momentum=0.78, lr=args.lr)  # Adam梯度下降  学习率=0.0001
    epoch_loss_list = []
    epoch_loss_test_list = []
    min_loss = 100
    for i in range(1, args.epochs + 1):
        total_loss  = 0
        preds = []
        y_true = []
        for idx, (inputs, labels) in enumerate(train_loader):
            #print("inputs.shape",inputs.shape)
            if args.useGPU:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32).cuda()  # lstm只能接受float 不然会报错
                labels = labels.to(torch.float32).cuda()  # lstm只能接受float 不然会报错
                outputs = model(inputs)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32)  # lstm只能接受float 不然会报错
                labels = labels.to(torch.float32)  # lstm只能接受float 不然会报错
                outputs = model(inputs)

            outputs = outputs.ravel()
            # print("shape:", outputs.shape, labels.shape)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            preds.extend(outputs)
            y_true.extend(labels)
            # writer.add_scalar(str(codes_name), total_loss , i)
        lstm_mse_train = mean_squared_error(np.array(y_true), np.array(preds))
        epoch_loss_list.append(lstm_mse_train)


        preds = []
        y_true = []

        for idx, (x_test_now, label) in enumerate(test_loader):

            if args.useGPU:
                x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
            x_test_now = x_test_now.to(torch.float32)

            pred = model(x_test_now).ravel()  # 出来的是3，1
            # list = pred.data.tolist()

            preds.extend(pred.data.tolist())
            y_true.extend(label)



        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(np.array(y_true), np.array(preds))
        rmse = root_mean_square_error(y_true, preds)
        epoch_loss_test_list.append(mse)

        print(f'epoch:{i:3d} | mse_loss_training:{lstm_mse_train:6.6f} | mse_loss_test:{mse:6.6F}')
        print(f'测试集上：R2:{r2:6.4f} | MSE:{mse:6.6F}\
                | MAE:{mae:6.6F} | RMSE:{rmse:6.6F}')

        # torch.save(model, args.models_path)
        # writer.add_graph(model=model, input_to_model=inputs)
        # torch.save(model, args.models_path)
        if save_model:
            if mse  < min_loss:
                min_loss = mse
                print('save')
                torch.save(model, args.models_path + "/" +args.mode+ ".pt")

                tmp_best_p = {'epoch': i, 'R2': r2, 'MSE': mse,
                              'MAE': mae, 'RMSE': rmse}

                # tmp_model_state_dict = model.state_dict().copy()
                # torch.save({'state_dict': tmp_model_state_dict}, args.models_path + "/LSTM" + ".pt")


    return epoch_loss_list,epoch_loss_test_list, tmp_best_p


def predict_lstm_am(test_loader, show_pic=False, ):
    model = torch.load(args.models_path + "/" +args.mode+ ".pt")
    # model = LSTMModel.lstm(input_size=feature_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
    #                        dropout=args.dropout)
    # model.to(args.device)
    # checkpoint = torch.load(args.models_path + "/LSTM" + ".pt")
    # model.load_state_dict(checkpoint['state_dict'])

    preds = []
    y_true = []
    for idx, (x_test_now, label) in enumerate(test_loader):

        if args.useGPU:
            x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
        x_test_now = x_test_now.to(torch.float32)

        pred = model(x_test_now).ravel()  # 出来的是3，1
        # list = pred.data.tolist()

        preds.extend(pred.data.tolist())
        y_true.extend(label)
    # lstm_mape=mape(np.array(y_true), np.array(preds))
    r2 = r2_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(np.array(y_true), np.array(preds))
    rmse = root_mean_square_error(y_true, preds)
    print(f'R2:{r2:6.4f} |MSE:{mse:6.4F} |RMSE:{rmse:6.4F} |MAE:{mae:6.4F}')
    # print(len(preds))
    # print(len(y_true))


    if show_pic:
        plt.figure(figsize=(12, 8))
        plt.plot(preds, 'r', label='y_test_predict')
        plt.plot(y_true, 'b', label='y_test_true')
        plt.legend()
        plt.show()
    return preds


def train_transformer(train_loader, save_model=False,
                      batch_size=1, sentence_len=2, feature_size=3):
    tf = transformerModel.TransAm(
        d_model=args.d_model,
        n_head=args.n_head,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,

        num_layers=args.num_layers,
        output_size=args.output_size,
        predays=pre_days,
        batch_size=batch_size,
        sentence_len=sentence_len,
        feature_size=feature_size,
        mdevice=args.device,
        mode=args.mode)

    if args.transfer_learning:
        checkpoint = torch.load(args.transformer_pretrained_path)
        tf.load_state_dict(checkpoint['state_dict'])

    tf.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(tf.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001
    #torch.optim.RMSprop()
    #optimizer = torch.optim.SGD(tf.parameters(), momentum=0.78, lr=args.lr)
    epoch_loss_list = []
    epoch_loss_test_list = []
    min_loss = 100
    for i in range(1, args.epochs + 1):
        total_loss = 0
        tf.train()  # 模型设置为训练模式
        preds = []
        y_true = []
        for idx, (inputs, labels) in enumerate(train_loader):
            # print("inputs.shape",inputs.shape)
            if args.useGPU:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.to(torch.float32)  # lstm只能接受float 不然会报错
            labels = labels.to(torch.float32)  # lstm只能接受float 不然会报错

            outputs = tf(inputs)

            outputs = outputs.ravel()

            # print("shape:", outputs.shape, labels.shape)
            # print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds.extend(outputs)
            y_true.extend(labels)


        transformer_mse_train = mean_squared_error(np.array(y_true), np.array(preds))
        epoch_loss_list.append(transformer_mse_train)
        # writer.add_scalar(str(codes_name), total_loss, i)

        # torch.save(model, args.models_path)
    # writer.add_graph(model=model, input_to_model=inputs)
    # torch.save(model, args.models_path)


    #     if save_model:
    #         torch.save({'state_dict': tf.state_dict()}, args.models_path + "/transformer" + ".pt")
        tf.eval()  # 模型设置为训练模式
        preds = []
        y_true = []

        for idx, (x_test_now, label) in enumerate(test_loader):

            if args.useGPU:
                x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
            x_test_now = x_test_now.to(torch.float32)

            pred = tf(x_test_now).ravel()  # 出来的是3，1
            # list = pred.data.tolist()

            preds.extend(pred.data.tolist())
            y_true.extend(label)


        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(np.array(y_true), np.array(preds))
        rmse = root_mean_square_error(y_true, preds)
        epoch_loss_test_list.append(mse)


        print(f'epoch:{i:3d} | mse_loss_training:{transformer_mse_train:6.6f} | mse_loss_test:{mse:6.6F}')
        print(f'测试集上：R2:{r2:6.4f} | MSE:{mse:6.6F}\
        | MAE:{mae:6.6F} | RMSE:{rmse:6.6F}')



        # torch.save(model, args.models_path)
        # writer.add_graph(model=model, input_to_model=inputs)
        # torch.save(model, args.models_path)
        if save_model:
            if mse < min_loss:
                min_loss = mse
                print('save')
                torch.save(tf, args.models_path + "/" +args.mode+ ".pt")
                tmp_best_p = {'epoch': i, 'R2': r2, 'MSE': mse,
                              'MAE': mae, 'RMSE': rmse}

                # tmp_model_state_dict = model.state_dict().copy()
                # torch.save({'state_dict': tmp_model_state_dict}, args.models_path + "/LSTM" + ".pt")

    return epoch_loss_list, epoch_loss_test_list, tmp_best_p

def predict_transformer(test_loader, show_pic=False,
                        batch_size=1, sentence_len=2, feature_size=3):
    model = torch.load( args.models_path + "/" +args.mode+ ".pt")

    # model = transformerModel.TransAm(
    #     d_model=args.d_model,
    #     n_head=args.n_head,
    #     dim_feedforward=args.dim_feedforward,
    #     dropout=args.dropout,
    #     activation=args.activation,

    #     num_layers=args.num_layers,
    #     output_size=args.output_size,
    #     predays=pre_days,
    #     batch_size=batch_size,
    #     sentence_len=sentence_len,
    #     feature_size=feature_size)
    # model.eval()  # 需要设置为训练模式
    # model.to(device)
    # checkpoint = torch.load(args.models_path + "/transformer" + ".pt")
    # model.load_state_dict(checkpoint['state_dict'])

    preds = []
    y_true = []
    for idx, (x_test_now, label) in enumerate(test_loader):

        if args.useGPU:
            x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
        x_test_now = x_test_now.to(torch.float32)

        pred = model(x_test_now).ravel()  # 出来的是3，1
        # list = pred.data.tolist()

        preds.extend(pred.data.tolist())
        y_true.extend(label)
    transformer_r2 = r2_score(y_true, preds)
    transformer_mae = mean_absolute_error(y_true, preds)
    transformer_mse = mean_squared_error(np.array(y_true), np.array(preds))
    transformer_rmse = root_mean_square_error(y_true, preds)
    # print(len(y_true),y_true[-1])
    print(
        f'R2:{transformer_r2:6.4f} |MSE:{transformer_mse:6.6F} |RMSE:{transformer_rmse:6.6F} |MAE:{transformer_mae:6.6F}')

    print('------------------------------')




    # print(len(preds))
    # print(len(y_true))
    if show_pic:
        plt.figure(figsize=(12, 8))
        plt.plot(preds, 'r', label='y_test_predict')
        plt.plot(y_true, 'b', label='y_test_true')
        plt.legend()
        plt.show()
    return preds

def train_tcn(train_loader, save_model=False):
    # model = LSTMModel.lstm(input_size=feature_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
    #                        dropout=args.dropout)
    kernel_size=3
    model = TCNModel.TCN(pre_days, 1, [64,32,16,8], kernel_size=kernel_size, dropout=args.dropout)
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001
    #optimizer = torch.optim.SGD(model.parameters(),momentum=0.78, lr=args.lr)  # Adam梯度下降  学习率=0.0001
    epoch_loss_list = []
    epoch_loss_test_list = []
    min_loss = 100
    for i in range(1, args.epochs + 1):
        total_loss  = 0
        preds = []
        y_true = []
        for idx, (inputs, labels) in enumerate(train_loader):
            #print("inputs.shape",inputs.shape)
            if args.useGPU:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32).cuda()  # lstm只能接受float 不然会报错
                labels = labels.to(torch.float32).cuda()  # lstm只能接受float 不然会报错
                outputs = model(inputs)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.to(torch.float32)  # lstm只能接受float 不然会报错
                labels = labels.to(torch.float32)  # lstm只能接受float 不然会报错
                outputs = model(inputs)
                
            #print("shape:", outputs.shape, labels.shape)#torch.Size([59, 1]) torch.Size([59])
            outputs = outputs.ravel()
            

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            preds.extend(outputs)
            y_true.extend(labels)
            # writer.add_scalar(str(codes_name), total_loss , i)
        mse_train = mean_squared_error(np.array(y_true), np.array(preds))
        epoch_loss_list.append(mse_train)




        preds = []
        y_true = []

        for idx, (x_test_now, label) in enumerate(test_loader):

            if args.useGPU:
                x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
            x_test_now = x_test_now.to(torch.float32)

            pred = model(x_test_now).ravel()  # 出来的是3，1
            # list = pred.data.tolist()

            preds.extend(pred.data.tolist())
            y_true.extend(label)



        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(np.array(y_true), np.array(preds))
        rmse = root_mean_square_error(y_true, preds)
        epoch_loss_test_list.append(mse)

        print(f'epoch:{i:3d} ')
        print(f'训练集上:MSE:{mse_train:6.6f}')
        print(f'测试集上:MSE:{mse:6.6F} |R2:{r2:6.4f} |  MAE:{mae:6.6F} | RMSE:{rmse:6.6F}')

        # writer.add_graph(model=model, input_to_model=inputs)

        if save_model:
            if mse  < min_loss:
                min_loss = mse
                print('save')
                torch.save(model, args.models_path + "/" +args.mode+ ".pt")

                tmp_best_p = {'epoch': i, 'R2': r2, 'MSE': mse,
                              'MAE': mae, 'RMSE': rmse}

                # tmp_model_state_dict = model.state_dict().copy()
                # torch.save({'state_dict': tmp_model_state_dict},  args.models_path + "/" +args.mode+ ".pt")


    return epoch_loss_list,epoch_loss_test_list, tmp_best_p


def predict_tcn(test_loader, show_pic=False, ):
    model = torch.load(args.models_path + "/" +args.mode+ ".pt")


    preds = []
    y_true = []
    for idx, (x_test_now, label) in enumerate(test_loader):

        if args.useGPU:
            x_test_now = x_test_now.cuda()  # batch_size,seq_len,input_size
        x_test_now = x_test_now.to(torch.float32)

        pred = model(x_test_now).ravel()  # 出来的是3，1
        # list = pred.data.tolist()

        preds.extend(pred.data.tolist())
        y_true.extend(label)
    # lstm_mape=mape(np.array(y_true), np.array(preds))
    lstm_r2 = r2_score(y_true, preds)
    lstm_mae = mean_absolute_error(y_true, preds)
    lstm_mse = mean_squared_error(np.array(y_true), np.array(preds))
    lstm_rmse = root_mean_square_error(y_true, preds)
    print(f'R2:{lstm_r2:6.4f} |MSE:{lstm_mse:6.4F} |RMSE:{lstm_rmse:6.4F} |MAE:{lstm_mae:6.4F}')
    # print(len(preds))
    # print(len(y_true))


    if show_pic:
        plt.figure(figsize=(12, 8))
        plt.plot(preds, 'r', label='y_test_predict')
        plt.plot(y_true, 'b', label='y_test_true')
        plt.legend()
        plt.show()
    return preds


#加载一些常量
codes = np.load(args.parameters_path + '/codes.npy')
codes = codes.tolist()
len_codes = len(codes)
time_list = np.load(args.parameters_path + '/time_list.npy',allow_pickle=True)
time_list = time_list.tolist()
len_train = np.load(args.parameters_path + '/len_train.npy')
len_train = len_train.tolist()
test_size = np.load(args.parameters_path + '/test_size.npy')
test_size = test_size.tolist()
pre_days = int(np.load(args.parameters_path + '/pre_days.npy'))
x_train_all = []
y_train_all = []
x_test_all = []
y_test_all = []
train_time_list = []
test_time_list = []
df = []

houzhui='_biaozhunhua'
#houzhui=''
x_train = pd.read_csv(args.data_path + '/x_train_all_codes'+houzhui+'.csv').values  # _biaozhun
x_test = pd.read_csv(args.data_path + '/x_test_all_codes'+houzhui+'.csv').values
y_train = pd.read_csv(args.data_path + '/y_train_all_codes'+houzhui+'.csv').values.ravel()
y_test = pd.read_csv(args.data_path + '/y_test_all_codes'+houzhui+'.csv').values.ravel()

x_train = x_train.reshape(x_train.shape[0], pre_days, -1)
x_test = x_test.reshape(x_test.shape[0], pre_days, -1)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

if __name__ == "__main__":

    train_data = MyDataset(x_train, y_train)
    test_data = MyDataset(x_test, y_test)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    if ett:
        test_data2 = MyDataset(x_test[-args.batch_size:], y_test[-args.batch_size:])
        test_loader2=DataLoader(dataset=test_data2, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    batch_size = args.batch_size
    _, sentence_len, feature_size = x_train.shape

    # print(batch_size,sentence_len,feature_size)

    # print(feature_size)13
    print('当前mode：'+args.mode)
    if args.mode == 'cnn':
        codes_epochs_loss, epoch_loss_test_list, best_p = train_cnn(train_loader, save_model=True,
                                        batch_size=batch_size, sentence_len=sentence_len, feature_size=feature_size)

        print("在测试集上")
        y_pred = predict_cnn(test_loader,
                              batch_size=batch_size, sentence_len=sentence_len, feature_size=feature_size)
        print("在训练集上")
        y_train_pred = predict_cnn(train_loader,
                                    batch_size=batch_size, sentence_len=sentence_len, feature_size=feature_size)

    if args.mode == 'lstm' or args.mode == 'bilstm':
        codes_epochs_loss, epoch_loss_test_list, best_p = train_lstm(train_loader, save_model=True)

        print("在测试集上")
        y_pred = predict_lstm(test_loader)
        print("在训练集上")
        y_train_pred = predict_lstm(train_loader)

    if args.mode == 'lstm_am':
        codes_epochs_loss, epoch_loss_test_list, best_p = train_lstm_am(train_loader, save_model=True)

        print("在测试集上")
        y_pred = predict_lstm_am(test_loader)
        print("在训练集上")
        y_train_pred = predict_lstm_am(train_loader)
        
    if args.mode == 'tcn':
        codes_epochs_loss, epoch_loss_test_list, best_p = train_tcn(train_loader, save_model=True)
        print("在测试集上")
        y_pred = predict_tcn(test_loader)
        print("在训练集上")
        y_train_pred = predict_tcn(train_loader)

    if args.mode == 'transformer' or args.mode == 'ett':

        codes_epochs_loss, epoch_loss_test_list, best_p = train_transformer(train_loader, save_model=True,
                                              batch_size=batch_size, sentence_len=sentence_len,
                                              feature_size=feature_size)

        # batch_size用于设置T2V，测试时候不需要的
        print("在测试集上")
        y_pred = predict_transformer(test_loader,
                                     batch_size=batch_size, sentence_len=sentence_len, feature_size=feature_size)
        print("在训练集上")
        y_train_pred = predict_transformer(train_loader,
                                           batch_size=batch_size, sentence_len=sentence_len, feature_size=feature_size)


    # writer.close()
    # print("len y_pred",len(y_pred))

    # all_codes_predict = pd.DataFrame(time_list[0][len_train[0]:])
    # all_codes_predict.columns = ["trade_date"]
    # for i in range(len(codes)):
    #     all_codes_predict[str(codes[i])] = y_pred[i::len_codes]
    #
    # if args.save_predict:  # 保存预测后的数据
    #     if args.mode == 'transformer' and args.transfer_learning:
    #         data_dir = os.path.join(args.data_path, ('predict_' + args.mode + '_with_transfer_all_codes' + '.csv'))
    #     else:
    #         data_dir = os.path.join(args.data_path, ('predict_' + args.mode + '_all_codes' + '.csv'))
    #     all_codes_predict.to_csv(data_dir, index=False)


    tmpDF = pd.DataFrame(y_pred)
    tmpDF.to_csv(args.data_path + '/' + args.mode + '_y_prediction_all_codes.csv', index=False)
    tmpDF = pd.DataFrame(y_train_pred)
    tmpDF.to_csv(args.data_path + '/' + args.mode + '_y_train_prediction_all_codes.csv', index=False)


    all_codes_loss = pd.DataFrame()

    all_codes_loss['loss on training'] = codes_epochs_loss
    all_codes_loss['loss on test'] =  epoch_loss_test_list
    all_codes_loss = all_codes_loss.T
    all_codes_loss.index.name = "epoch"
    all_codes_loss.columns = range(1, args.epochs + 1)

    all_codes_loss.to_csv(args.loss_path + '/all_loss_' + args.mode + '.csv', index=True)

    print('最好的一次：',best_p)
        #f'best:epoch:{best_p["epoch"]:3d} | R2:{best_p["R2"]:6.4f}  | MSE:{best_p["MSE"]:6.4F} | RMSE:{best_p["RMSE"]:6.4F} | MAE:{best_p["MAE"]:6.4F}')
