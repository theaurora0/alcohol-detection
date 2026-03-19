import time  # 导入time模块用于计时等
import json  # 导入json模块做参数文件读写
import shutil  # 导入shutil做目录操作，如递归删除
import numpy as np  # 导入numpy进行数值计算

import torch  # 导入pytorch做深度学习
from torch import nn  # 导入nn模块用于定义网络和损失

from model import MLP  # 导入自定义的MLP模型
from utils.data_loading_utils import data_process, data_process_, data_process_1, data_iter  # 导入数据处理相关函数

a, b, c, d = 0, 0, 8, 16  # 选择增益参数范围（a-b控制1550通道，c-d控制1362通道）
parameter_name = "chan_1362_7_3_2"  # 定义本次训练参数保存的文件名
link_parameter = None  # 如果不为None可加载已有参数进行继续训练
train_groups = "7_2"  # 指定训练集合的组别
test_groups = "7_7+7_8"  # 指定测试集合组别（多个用+分隔）
T = "train"  # 选择以train还是test集上的最小损失为模型选择标准
epochs = 300  # 训练轮数
lr = 1e-5  # 学习率
layer_num = [b + d - a - c, 48, 48, 1]  # 每层神经元数（第一项一般为输入特征数）
dropout_num = [0, 0, 0, 0]  # 各层的dropout比例，0表示不做dropout
BN = True  # 是否使用BatchNorm
res = True  # 是否使用ResNet残差结构
ns = 0.1  # LeakyReLU负半轴斜率
process = True  # 是否重新生成归一化的数据集（影响数据加载速度/重做）

if process:
    net_train_path = "Relative_Data/train"  # 归一化后训练数据存放路径
    shutil.rmtree(net_train_path)  # 先删除已存在的
    for train_group in train_groups.split('+'):  # 可支持多组训练
        data_train_path = "Absolute_Data/{}".format(train_group)  # 原始数据存放路径
        data_process_1(data_train_path, net_train_path, a, b, c, d)  # 处理并生成训练样本
        
    net_test_path = "Relative_Data/test"  # 测试样本归一化后存放路径
    shutil.rmtree(net_test_path)  # 先删除已存在的
    for test_group in test_groups.split('+'):  # 可支持多组测试
        data_test_path = "Absolute_Data/{}".format(test_group)  # 原始数据存放路径
        data_process_1(data_test_path, net_test_path, a, b, c, d)  # 处理并生成测试样本

# 选GPU还是CPU训练，根据环境自动选择
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")  # 打印当前使用的设备

# 如果link_parameter为None则新建网络，否则加载之前的参数权重继续训练
if link_parameter == None:
    model = MLP(layer_num, dropout_num, res=res, BN=BN, ns=ns).to(device)  # 新建模型
else:
    model = MLP(layer_num, dropout_num, res=res, BN=BN, ns=ns).to(device)  # 新建结构一致的模型
    if device == "cpu":
        model.load_state_dict(torch.load("parameter_path/{}.pth".format(link_parameter), map_location="cpu"))  # 加载之前保存的参数到cpu
    else:
        model.load_state_dict(torch.load("parameter_path/{}".format(link_parameter)))  # 加载参数到cuda

parameter_sum = sum(p.numel() for p in model.parameters())  # 统计参数总数

loss_fn = nn.MSELoss()  # 定义均方误差作为损失函数
loss_fn = loss_fn.to(device)  # 损失函数部署到对应设备

optimizer = torch.optim.Adam(model.parameters(), lr)  # Adam优化器，指定学习率

# 定义训练过程
def train(train_iter, model, loss_fn, optimizer):
    data_size = len(train_iter.dataset)  # 当前训练集样本总数
    model.train()  # 进入训练模式
    average_error = 0  # 累计平均误差
    for count, (X, y) in enumerate(train_iter):  # 遍历每个batch
        batch_size = len(X)  # 当前batch样本数
        X, y = X.to(device), y.to(device)  # 数据转到GPU或CPU
        y_hat = model(X)  # 前向传播，预测y
        loss = loss_fn(y_hat, y)  # 计算损失
        average_error += abs((y.detach().cpu() - y_hat.detach().cpu()).item())  # 累加本batch绝对误差
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
    average_error = (average_error / data_size) * 100  # 归一化得到均值百分比误差
    print(f"训练误差均值：{average_error:.3f}")  # 打印训练误差
    return average_error  # 返回误差

# 定义测试过程
def test(test_iter, model, loss_fn):
    data_size = len(test_iter.dataset)  # 测试集样本总数
    batch_nums = len(test_iter)  # 总batch数
    model.eval()  # 设置模型为评估/推理模式
    average_error = 0  # 累计平均误差
    with torch.no_grad():  # 禁止梯度
        for X, y in test_iter:  # 遍历测试集
            X, y = X.to(device), y.to(device)  # 数据转设备
            y_hat = model(X)  # 得到预测
            average_error += abs((y.detach().cpu() - y_hat.detach().cpu()).item())  # 累加本batch绝对误差
        average_error = (average_error / data_size) * 100  # 归一化为均值百分比误差
        print(f"测试误差均值：{average_error:.3f}")  # 打印测试误差
        return average_error  # 返回误差

# 加载训练集与测试集迭代器（batch_size=1,因为实验数据较少）
train_iter = data_iter(path="Relative_Data/train", batch_size=1, is_train=True)
test_iter = data_iter(path="Relative_Data/test", batch_size=1, is_train=False)

# 训练主循环
min_loss = 10000  # 最小损失初始化为较大值，用于模型选择
train_loss_t = 0  # 记录最佳模型对应的train损失
test_loss_t = 0  # 记录最佳模型对应的test损失
loss_list = []  # 记录每一轮loss列表
for t in range(epochs):  # 训练若干轮
    print(f"Epoch {t + 1}\n-------------------------------")  # 打印当前轮数
    train_loss = train(train_iter, model, loss_fn, optimizer)  # 训练
    test_loss = test(test_iter, model, loss_fn)  # 测试
    loss_list.append(train_loss)  # 保存当前轮训练误差
    # 保存最佳参数权重
    if T == "test":  # 以测试集最小损失为标准
        if test_loss < min_loss:
            torch.save(model.state_dict(), "parameter_path\\{}.pth".format(parameter_name))  # 保存参数
            min_loss = test_loss
            test_loss_t = test_loss  # 记录本次test loss
            train_loss_t = train_loss  # 记录对应的train loss
    else:  # 以训练集最小损失为标准
        if train_loss < min_loss:
            torch.save(model.state_dict(), "parameter_path\\{}.pth".format(parameter_name))  # 保存参数
            min_loss = train_loss
            test_loss_t = test_loss
            train_loss_t = train_loss

# 结果及超参数写入json文件（para_intro.json）
with open('parameter_path/para_intro.json', 'r') as filename:
    para = filename.read()  # 读取旧内容
    para = json.loads(para)  # 转为字典
    para[parameter_name] = {
        "layer": layer_num,  # 网络层结构
        "BatchNorm": BN,  # BN开关
        "negative_relu": ns,  # LeakyReLU负半轴斜率
        "dropout": dropout_num,  # 每层dropout
        "parameter_sum": parameter_sum,  # 网络参数总量
        "lr": lr,  # 学习率
        "res": res,  # 残差结构开关
        "epochs": epochs,  # 训练轮数
        "train_groups": train_groups,  # 训练组
        "test_groups": test_groups,  # 测试组
        "1550_chan": [a, b],  # 1550通道的增益范围
        "1360_chan": [c, d],  # 1362通道的增益范围
        "min_loss": T,  # 以哪个loss最小为准
        "train_loss": train_loss_t,  # 最优参数对应train loss
        "test_loss": test_loss_t  # 最优参数对应test loss
    }
    para = json.dumps(para, indent=4)  # 美化json
    filename = open('parameter_path/para_intro.json', 'w')  # 重新打开写入
    filename.write(para)  # 写入新内容
    filename.close()  # 关闭文件

# 以下为可选，保存loss随轮次变化曲线，可作画loss曲线用
# import os
# import pandas as pd
#
# df = pd.DataFrame(np.array(loss_list).ravel(order="F"))
# df.to_csv("loss_list.csv", header=False)
