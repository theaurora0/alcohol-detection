import json  # 导入json模块，用于读取和解析json文件
import argparse  # 导入argparse模块，用于处理命令行参数
import shutil  # 导入shutil模块，用于文件和文件夹的操作，比如删除目录
import numpy as np  # 导入numpy，用于数值计算
import pandas as pd  # 导入pandas，用于生成DataFrame和数据处理
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import matplotlib  # 导入matplotlib主模块

matplotlib.rc("font", family='KaiTi')  # 设置全局字体为楷体，适合中文显示

import torch  # 导入torch用于深度学习的训练和推理
from torch import nn  # 从torch库中导入神经网络相关的模块

from model import MLP  # 从自定义的model模块中导入MLP模型
from utils.data_loading_utils import data_process, data_process_, data_process_1, data_iter  # 导入自定义的数据处理函数

# 获取训练所用的设备，如果有GPU可用则用cuda，没有则用cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")  # 打印当前用的设备

def parse_args(s):
    # 创建命令行参数解析器
    parse = argparse.ArgumentParser()
    # 增加para参数，网络训练参数，字符串类型，默认由s提供
    parse.add_argument('--para', default=s, type=str, help='Network training parameters')
    # 增加test_groups参数，测试组，字符串类型，默认"9_3"
    parse.add_argument('--test_groups', default="9_3", type=str, help='The second set of training parameters')
    # 增加save参数，是否保存，布尔类型，默认False
    parse.add_argument('--save', default=False, type=bool, help='save or not')
    # 解析参数并返回
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    # 三组超参数字符串，每个字符串用+隔开，对应不同的测试组
    s = [
        "chan_1550_1362_7_1_1+chan_1550_1362_7_1_2+chan_1550_1362_7_1_3+chan_1550_1362_7_1_4+chan_1550_1362_7_2_1+chan_1550_1362_7_2_2+chan_1550_1362_7_2_3+chan_1550_1362_7_2_4",
        "chan_1550_7_1_1+chan_1550_7_1_2+chan_1550_7_1_3+chan_1550_7_1_4+chan_1550_7_2_1+chan_1550_7_2_2+chan_1550_7_2_3+chan_1550_7_2_4",
        "chan_1362_7_1_1+chan_1362_7_1_2+chan_1362_7_1_3+chan_1362_7_1_4+chan_1362_7_2_1+chan_1362_7_2_2+chan_1362_7_2_3+chan_1362_7_2_4"
    ]
    # 解析命令行参数，这里以s[0]为para参数，表示两个激光器均正常可用
    args = parse_args(s[0]) 
    label = []  # 存放真实值
    value = [[], [], [], [], [], [], [], []]  # 存放预测结果，共8组
    i = 0  # 计数器，表示当前是第几个通道

    # 遍历所有参数名，每个parameter_name对应一个模型
    for parameter_name in args.para.split("+"):
        # 拼接当前参数的模型权重路径
        path = "parameter_path/{}.pth".format(parameter_name)

        # 打开JSON文件，读取模型和数据的配置信息
        filename = open('parameter_path/para_intro.json', 'r')
        para = filename.read()
        para = json.loads(para)
        para = para[parameter_name]

        # 打印当前网络超参数、通道信息和训练组信息等
        print("res:{},layer:{}, BN:{}, negative:{}, dropout:{}, parameter_sum:{}".format(
            para["res"], para["layer"], para["BatchNorm"], para["negative_relu"], para["dropout"], para["parameter_sum"]
        ))
        print("lr:{}, epochs:{}, train_groups:{}, test_groups:{}, 1550_chan:{}, 1360_chan:{}".format(
            para["lr"], para["epochs"], para["train_groups"], para["test_groups"], para["1550_chan"], para["1360_chan"]
        ))
        print("min_loss:{}".format(para["min_loss"]))
        print("train_loss:{}".format(para["train_loss"]))
        print("test_loss:{}".format(para["test_loss"]))

        # 读取所需的模型参数
        layer_num = para["layer"]  # 层数
        dropout_num = para["dropout"]  # dropout概率
        BN = para["BatchNorm"]  # 是否使用批归一化
        ns = para["negative_relu"]  # 是否为带负的ReLU
        res = para["res"]  # 是否有残差
        # 构建MLP模型，并加载权重到选择的device
        model = MLP(layer_num, dropout_num, res=res, BN=BN, ns=ns).to(device)
        if device == "cpu":
            model.load_state_dict(torch.load(path, map_location="cpu"))  # 用cpu加载权重
        else:
            model.load_state_dict(torch.load(path))  # 用cuda加载权重
        # 使用均方误差作为损失函数
        loss_fn = nn.MSELoss()

        # 提取当前通道的数据选择参数
        a, b = para["1550_chan"][0], para["1550_chan"][1]
        c, d = para["1360_chan"][0], para["1360_chan"][1]
        test_groups = args.test_groups  # 测试组参数
        net_test_path = "Relative_Data/test"  # 临时存储测试集的路径
        shutil.rmtree(net_test_path)  # 删除上一次生成的测试集数据
        # 对每个测试组进行数据预处理，将其处理到统一的Relative_Data/test
        for test_group in test_groups.split('+'):
            data_test_path = "Absolute_Data/{}".format(test_group)
            data_process_1(data_test_path, net_test_path, a, b, c, d)
        # 生成测试集迭代器，每次一个样本
        test_iter = data_iter(path="Relative_Data/test", batch_size=1, is_train=False)

        # 开始推理，设置网络为评估模式
        model.eval()

        test_loss = 0  # 初始化测试损失
        with torch.no_grad():  # 不计算梯度，提高效率
            loss_num = []  # 储存每个样本的误差
            for count, (X, y) in enumerate(test_iter):  # 遍历所有测试样本
                X, y = X.to(device), y.to(device)  # 数据转移到指定device
                y_hat = model(X)  # 获取预测值

                if i == 0:
                    label.append(y.item() * 100)  # 只在第一组时保存真实值(保证不重复)
                value[i].append(y_hat.item() * 100)  # 保存预测值
                # 计算当前样本的误差
                dif = np.abs(y_hat.detach().cpu().item() - y.detach().cpu().item()) * 100
                loss_num.append(dif)
                # 打印每个样本的真实值、预测值和误差
                print("测量值: {:.3f}, 预测值: {:.3f}, 误差: {:.6f}".format(
                    y.detach().cpu().item() * 100, y_hat.detach().cpu().item() * 100, dif
                ))
                # 如果只取前30个样本可取消注释
                # if count == 30:
                #     break
            # 打印本组的平均误差和标准差
            print("均值:{} 方差:{}".format(np.mean(loss_num), np.std(loss_num)))
        i += 1  # 完成一组模型

    # 将列表转换为numpy数组，方便后续处理
    label = np.array(label)
    value = np.array(value)

    # 裁剪预测值范围到[30, 80]（防止极端值影响结果，如果小于30设为30，大于80设为80）
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            if value[i, j] > 80:
                value[i, j] = 80
            if value[i, j] < 30:
                value[i, j] = 30

    # 对标签和预测值进行小数点一位四舍五入
    label = np.around(label, 1)
    value = np.around(value, 1)

    # 计算每个样品不同模型预测均值和标准差，便于画误差棒
    value_mean = np.around(value.mean(axis=0), 1)
    value_std = np.around(value.std(axis=0), 1)

    # # 以下三段是不同风格的柱状图，可以取消注释看效果
    # # 第一种，含真实值和预测均值
    # x = np.arange(value.shape[1])  # 样品序列
    # plt.figure(figsize=(2, 5))  # 图像尺寸
    # ax = plt.axes()
    # ax.set_facecolor("white")
    # bar_width = 0.2
    # plt.bar(x, label, bar_width, color="red", label="真实值")
    # plt.bar(x + bar_width, value_mean, bar_width, color="b", label="预测均值")
    # plt.ylabel("浓度(%)")
    # plt.legend(loc=(0, 1))
    # xl = []
    # for i in range(value.shape[1]):
    #     xl.append("样品{}".format(i + 1))
    # plt.xticks(x + bar_width / 2, xl)
    # plt.grid(alpha=0.3)
    # plt.show()

    # # 第二种，逐个显示每组预测值
    # x = np.arange(5) + 1
    # plt.figure(figsize=(5, 5))
    # ax = plt.axes()
    # ax.set_facecolor("white")
    # bar_width = 0.2
    # xl = ["标准值"]
    # plt.bar(x[0], label, bar_width, color="red", label="真实值")
    # for i in range(4):
    #     plt.bar(x[i+1] + bar_width, value[i], bar_width, color="b", label="预测值{}".format(i + 1))
    #     xl.append("预测值{}".format(i + 1))
    # plt.ylabel("浓度(%)")
    # plt.legend(loc=(0, 1))
    # plt.xticks(x + bar_width / 2, xl)
    # plt.grid(alpha=0.3)
    # plt.show()

    # # 第三种，带误差棒，能明显比较不同样品之间的误差范围
    # x = np.arange(value.shape[1])
    # plt.figure(figsize=(4, 5))
    # bar_width = 0.2
    # for n in range(value.shape[1]):
    #     plt.subplot(1, value.shape[1], n + 1)
    #     plt.ylim((value_mean[n] - 3, value_mean[n] + 3))
    #     plt.bar(x[n], label[n], bar_width, color="red", label="标准值")
    #     error_params = dict(elinewidth=2, ecolor='orange', capsize=5)  # 设置误差棒样式
    #     plt.bar(x[n] + bar_width, value_mean[n], bar_width, yerr=value_std[n], error_kw=error_params, color="b", label="预测均值")
    #     plt.xlabel("样品{}".format(n + 1))
    #     if n == 0:
    #         plt.ylabel("浓度(%)")
    #         plt.legend(loc=(0, 1))
    #     plt.grid(alpha=0.3)
    #     plt.xticks([])
    # plt.subplots_adjust(wspace=2, hspace=1)
    # plt.show()

    # # 还可以画折线图，显示预测值和真实值随样品的变化趋势
    # x = np.arange(1, 5, 1)
    # label_T = np.array([label, label, label, label]).T
    # value_T = value.T
    # plt.figure(figsize=(5, 5))
    # plt.ylim(33, 83)
    # for i in range(label_T.shape[0]):
    #     plt.plot(x, label_T[i], "r-")
    #     plt.plot(x, value_T[i], "b.-")
    # plt.show()

    # 计算两组的平均误差（与50%浓度的差距，用于可视化表格中的“误差”列）
    error_1 = np.around(abs((value[0] + value[1] + value[2] + value[3]) / 4 - 50), 1)
    error_2 = np.around(abs((value[4] + value[5] + value[6] + value[7]) / 4 - 50), 1)
    # 计算预测均值与真实值的误差，用于数据分析
    error_data = np.around(np.abs(value_mean - label), 3)
    print(np.std(error_data))  # 打印预测误差的标准差
    s_num = np.arange(value.shape[1])
    s_num = map(str, s_num + 1)
    # 整理所有信息，生成表格所需数据
    data = {
        "真实值": label,
        "预测值1": value[0], "预测值2": value[1], "预测值3": value[2], "预测值4": value[3], "第一组误差": error_1,
        "预测值5": value[4], "预测值6": value[5], "预测值7": value[6], "预测值8": value[7], "第二组误差": error_2
    }
    df = pd.DataFrame(data)  # 构建pandas表格DataFrame

    # 创建一个图像和坐标轴，用于显示表格（不显示坐标轴本身）
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')  # 隐藏坐标轴

    # 在fig的ax上创建表格
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)  # 关闭自动字体大小调整
    table.set_fontsize(10)  # 设置表格字体大小为10
    table.scale(1.2, 1.5)  # 调整表格大小

    plt.savefig("table_picture//1.jpg")  # 保存表格为图片到指定目录
    plt.show()  # 弹出窗口展示表格图片

    # 如果save参数为True，则保存结果到csv文件
    if args.save:
        data_r = np.append([label], value, axis=0)  # 合并真实值与预测值
        data_r_T = data_r.T  # 转置，方便逐样本存储
        df = pd.DataFrame(data_r_T)  # 转为DataFrame格式
        df.to_csv(f"Results\\{args.para}_group_{args.test_groups}.csv")  # 保存为csv文件
