#coding:utf-8
#file: Net.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/4/4 11:43
#@desc:

import torch
import torch.nn as nn
from torchsummary import summary
from config import getConfig
from torch.autograd import Variable


class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN, self).__init__()
        #初始化时定义各个组件
        self.embedding=nn.Embedding(config["max_features"],config["embedding_dims"],)
        convs=[
            nn.Sequential(nn.Conv1d(config["embedding_dims"],128,kernel_size),
                          nn.BatchNorm1d(128),
                          nn.ReLU(True),
                          nn.MaxPool1d(kernel_size=(config["maxlen"]-2*kernel_size+2))
                          ) for kernel_size in [3,4,5]
        ]
        self.convs=nn.ModuleList(convs)
        self.fc=nn.Sequential(
            nn.Linear(3*128,128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config["nums_class"])
        )


    def forward(self,input):
        embeds=self.embedding(input)#batch*seq*embeddim
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        conv_out=[conv(embeds.permute(0,2,1)) for conv in self.convs]
        conv_out=torch.cat(conv_out,dim=1).squeeze()
        out=self.fc(conv_out)
        return out


class Text_BiRNN(nn.Module):
    def __init__(self,config):
        super(Text_BiRNN,self).__init__()
        #初始化时定义各个组件
        self.embedding=nn.Embedding(config['max_features'],config['embedding_dims'])
        self.lstm=nn.LSTM(config['embedding_dims'],128,2,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(128*2,config['nums_class'])

    def forward(self,x):
        out=self.embedding(x)#batch*seqlen*embedim
        out,_=self.lstm(out)#每个cell都返回了hiddenstates
        out=self.fc(out[:,-1,:])
        return out


class TextRCNN(nn.Module):
    def __init__(self,config):
        super(TextRCNN,self).__init__()
        self.embedding=nn.Embedding(config["max_features"],config["embedding_dims"])
        self.lstm=nn.LSTM(config["embedding_dims"],128,num_layers=2,batch_first=True,bidirectional=True)
        self.conv_pool=nn.Sequential(nn.Conv1d(config["embedding_dims"]+256,128,kernel_size=1),
                                     nn.ReLU(True),
                                     nn.MaxPool1d(config["maxlen"])
                                     )
        self.fc=nn.Linear(128,config["nums_class"])

    def forward(self,input):
        """
        :param input: [batch,seq_len]
        :return:
        """
        out=self.embedding(input)#[batch,seq_len,embedim]
        current=out
        #lstm输入要求是：batch_first=True时，输入和输出都为[batch,seq_len,embedim]
        out,_=self.lstm(out)
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        out=torch.cat([out,current],dim=2)
        out=self.conv_pool(out.permute(0,2,1))#[batch,nums_kernel,seq_len]
        out=self.fc(out.squeeze())
        return out


class TextAttBiLSTM(nn.Module):
    def __init__(self,config):
        super(TextAttBiLSTM,self).__init__()
        self.embedding=nn.Embedding(config["max_features"],config["embedding_dims"])
        self.lstm=nn.LSTM(config["embedding_dims"],128,2,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(256,config["nums_class"])
        #nn.parameter将attention_w变成可优化的参数，模型的一部分。
        self.attention_w=Variable(torch.rand(size=(256,config["attention_size"])),requires_grad=True)
        self.attention_b=Variable(torch.rand(size=(config["attention_size"],)),requires_grad=True)
        self.attention_u=Variable(torch.rand(size=(config["attention_size"],)),requires_grad=True)

    def forward(self,input):
        #input:[batch,seqlen]
        y=self.embedding(input)#[batch,seqlen,embeding]
        out,_=self.lstm(y)#[batch,seqlen,256]
        #[batch,seqlen,256]*[256,attention_size]
        x=torch.tanh(torch.add(torch.tensordot(out,self.attention_w,dims=1),self.attention_b))#[batch,seqlen,attention_size]
        #[batch,seqlen,attention_size]*[attention_size,]
        x=torch.tensordot(x,self.attention_u,dims=1)
        x=torch.softmax(x,dim=1)
        x=torch.mul(x.unsqueeze(dim=-1),out)
        x=torch.sum(x,dim=1)
        out=self.fc(x)
        return out








