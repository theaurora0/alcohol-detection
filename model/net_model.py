import torch.nn as nn                # 导入 PyTorch 的神经网络模块
import torch                        # 导入 PyTorch
from collections import OrderedDict  # 导入有序字典（这里好像没用到）


# # Define net model
# class MLP(nn.Module):
#     def __init__(self, num_inputs, num_hiddens, dropout):
#         super().__init__()
#         # self.save_hyperparameters()
#         self.net = nn.Sequential(
#             # hidden layer
#             nn.Linear(num_inputs, num_hiddens),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             # output layer
#             nn.Linear(num_hiddens, 1),
#         )
#
#     def forward(self, X):
#         out = self.net(X)
#         return out.squeeze(0).squeeze(0)

class bn(nn.Module):   # 自定义批标准化激活层
    def __init__(self, ns=0.1):  # ns作为LeakyReLU的负斜率
        super(bn, self).__init__()    # 调用父类初始化
        self.net = nn.LeakyReLU(ns)   # LeakyReLU激活函数

    def forward(self, x):             # 前向传播
        x = (x - torch.mean(x)) / torch.std(x)  # 简单的标准化（非正统BN）
        x = self.net(x)               # 通过激活函数
        return x                      # 返回结果


class res_block(nn.Module):   # 定义残差块
    def __init__(self, input_num, output_num, ns=0.1, bias=True): # 输入/输出维度，激活参数，是否有偏置
        super(res_block, self).__init__()        # 父类初始化
        self.linear_1 = nn.Linear(input_num, output_num, bias)        # 第一层线性映射
        self.linear_2 = nn.Linear(output_num, output_num, bias)       # 第二层线性映射
        if input_num != output_num:                                   # 如果输入输出形状不同，需要额外对齐
            self.linear_3 = nn.Linear(input_num, output_num, bias)    # 残差分支线性层
        else:
            self.linear_3 = None                                      # 否则不需要第三层
        self.bn = bn(ns)                                              # 调用自定义bn层

    def forward(self, x):                 # 前向传播
        y = self.bn(self.linear_1(x))     # 先全连接，再bn
        y = self.linear_2(y)              # 再全连接

        if self.linear_3:                 # 如果需要残差变换
            x = self.linear_3(x)          # 用线性层调整输入维度

        return x + y                      # 残差连接返回


# Define net model
class MLP(nn.Module):     # 定义MLP总结构
    def __init__(self, layer_num, dropout_num, res=True, bias=True, BN=True, ns=0.1): # 参数包括各层维度、dropout、是否用残差/BN
        super(MLP, self).__init__()               # 父类初始化
        # self.save_hyperparameters()
        num = len(layer_num) - 1                  # 层数，等于维度数组-1
        self.net = nn.Sequential()                # 使用顺序容器
        for i in range(num):                      # 遍历每层
            if i < (num - 1):                    # 除了最后一层都加激活/BN/dropout
                if res:                          # 如果用残差
                    self.net.add_module("layer_{}".format(i), res_block(layer_num[i], layer_num[i + 1], ns, bias=bias)) # 添加残差块
                else:
                    self.net.add_module("layer_{}".format(i), nn.Linear(layer_num[i], layer_num[i + 1], bias=bias))      # 普通线性层
                if BN:                           # 用BN
                    self.net.add_module("BN_{}".format(i), bn(ns=ns))        # 添加自定义bn层
                else:
                    self.net.add_module("LeakyReLU_{}".format(i), nn.LeakyReLU(negative_slope=ns)) # 只激活
                self.net.add_module("Dropout_{}".format(i), nn.Dropout(dropout_num[i]))            # 添加dropout
            else:
                self.net.add_module("layer_{}".format(i), nn.Linear(layer_num[i], layer_num[i + 1], bias=bias)) # 最后一层只做线性

    def forward(self, X):                       # 前向传播
        out = self.net(X)                       # 顺序结构直接运行
        return out.squeeze(0).squeeze(0)        # 压缩输出的多余维度
