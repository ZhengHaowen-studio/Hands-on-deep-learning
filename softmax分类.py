#!/usr/bin/env python
# coding=utf-8
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display

batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="E:\PythonProgram\Fashion-MNIST\data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="E:\PythonProgram\Fashion-MNIST\data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]) # 这里采用了交叉熵损失 跟之前直接求误差的损失不一样
    # 交叉熵的特点 这里返回的只是序号对应的损失

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module): # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
        # 在训练模型时会在前面加上：
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        print('y',y)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    for epoch in range(num_epochs): # 迭代次数
        train_epoch_ch3(net, train_iter, loss, updater)

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

if __name__=='__main__':
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)