#coding:utf-8
#file: main.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/4/4 16:00
#@desc:
from Net.Net import *
import torch.nn.functional as fn
import tensorflow_hub as hub
from sklearn import metrics
import numpy as np
import torch
from torch.autograd import Variable
from config import getConfig
from torchsummary import summary
import torch, torchvision


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = len(x)//batch_size + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def execute():
    config = getConfig.get_config("./config/config.ini")
    print("构建模型...")
    model=TextAttBiLSTM(config)
    #读取数据
    data=np.load("./data.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    #目标函数和优化器
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.08)
    for epoch in range(config["epochs"]):
        global_batch=1
        data_iter = batch_iter(x_train, y_train, config["batch_size"])
        print("Epoch {}/{}".format(epoch+1,config["epochs"]))
        for n in range(len(x_train)//config["batch_size"]+1):
            text,label=next(data_iter)
            text=Variable(torch.from_numpy(text)).long()
            output=model(text)
            model.zero_grad()
            label=Variable(torch.from_numpy(label))
            loss=fn.cross_entropy(output,label.argmax(dim=1))
            loss.backward()
            optimizer.step()
            acc=metrics.accuracy_score(label.argmax(dim=1),output.argmax(dim=1))
            print("{}/{} [..............................] - accuracy: {}".format(global_batch*256,len(x_train),acc))
            global_batch+=1


if __name__ =="__main__":
    execute()


