#coding:utf-8
#file: Net.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/3/31 23:00
#@desc:Net

from tensorflow.keras import Input,Model,backend as k,layers
import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding,Conv1D,GlobalMaxPooling1D,\
    Concatenate,Dropout,Bidirectional,LSTM,Lambda

class TextCNN(object):
    def __init__(self,maxlen,max_features,emb_dims,nums_class=5,last_activition="softmax"):
        self.max_len=maxlen
        self.max_features=max_features
        self.nums_class=nums_class
        self.emb_dim=emb_dims
        self.last_activition=last_activition

    def get_model(self):
        input=Input((self.max_len,))
        embedding=Embedding(input_dim=self.max_features,output_dim=self.emb_dim,
                            input_length=self.max_len)(input)#batch*seq*embeddim
        convs=[]
        for kernel_size in [3,4,5]:
            c=Conv1D(128,kernel_size,activation='relu')(embedding)
            c=GlobalMaxPooling1D()(c)
            convs.append(c)
        x=Concatenate()(convs)
        output=Dense(self.nums_class,activation=self.last_activition)(x)
        model=Model(inputs=input,outputs=output)
        return model


class TextRNN(object):
    def __init__(self,maxlen,max_features,emb_dims,nums_class=5,last_activition="softmax"):
        self.max_len=maxlen
        self.max_features=max_features
        self.nums_class=nums_class
        self.emb_dim=emb_dims
        self.last_activition=last_activition

    def get_model(self):
        input=Input(shape=(self.max_len,))
        embedding=Embedding(self.max_features,self.emb_dim,input_length=self.max_len)(input)
        x=LSTM(128)(embedding)
        output=Dense(self.nums_class,activation=self.last_activition)(x)
        model=Model(input,output)
        return model


class TextBiRNN(object):
    def __init__(self,maxlen,max_features,emb_dims,nums_class=5,last_activition="softmax"):
        self.max_len=maxlen
        self.max_features=max_features
        self.nums_class=nums_class
        self.emb_dim=emb_dims
        self.last_activition=last_activition

    def get_model(self):
        input=Input(shape=(self.max_len,))
        embedding=Embedding(self.max_features,self.emb_dim,input_length=self.max_len)(input)
        x=Bidirectional(LSTM(128))(embedding)
        output=Dense(self.nums_class)(x)
        model=Model(input,output)
        return model


class TextRCNN(object):
    def __init__(self,maxlen,max_features,emb_dims,nums_class=5,last_activition="softmax"):
        self.max_len=maxlen
        self.max_features=max_features
        self.nums_class=nums_class
        self.emb_dim=emb_dims
        self.last_activition=last_activition

    def get_model(self):
        input=Input(shape=(self.max_len,))
        input_left=Input(shape=(self.max_len,))
        input_right=Input(shape=(self.max_len,))
        embeder=Embedding(self.max_features,self.emb_dim,input_length=self.max_len)
        embedding=embeder(input)
        embedding_left=embeder(input_left)
        embedding_right=embeder(input_right)
        left=LSTM(128,return_sequences=True)(embedding_left)
        right=LSTM(128,return_sequences=True,go_backwards=True)(embedding_right)
        right=Lambda(lambda x:k.reverse(x,axes=1))(right)
        x=Concatenate(axis=2)([left,embedding,right])
        x=Conv1D(128,kernel_size=1)(x)
        x=GlobalMaxPooling1D()(x)
        output=Dense(self.nums_class,activation=self.last_activition)(x)
        model=Model(inputs=[input_left,input,input_right],outputs=output)
        return model


class TextAttBiRNN(object):
    def __init__(self,maxlen,max_features,emb_dims,nums_class=5,last_activition="softmax"):
        self.max_len=maxlen
        self.max_features=max_features
        self.nums_class=nums_class
        self.emb_dim=emb_dims
        self.last_activition=last_activition

    def get_model(self):
        input=Input(shape=(self.max_len,))
        embedding=Embedding(self.max_features,self.emb_dim,input_length=self.max_len)(input)
        x=Bidirectional(LSTM(128,return_sequences=True))(embedding)
        x=Attention(32)(x)
        output=Dense(self.nums_class,activation=self.last_activition)(x)
        model=Model(input,output)
        return model


class Attention(layers.Layer):
    def __init__(self,attention_size):
        super(Attention,self).__init__()
        self.att_size=attention_size

    def build(self,input_shape):
        self.attention_w = self.add_weight(name="atten_w", shape=(input_shape[-1], self.att_size),
                                           initializer=tf.random_uniform_initializer(), dtype="float32", trainable=True)
        self.attention_u = self.add_weight(name="atten_u", shape=(self.att_size,),
                                           initializer=tf.random_uniform_initializer(), dtype="float32", trainable=True)
        self.attention_b = self.add_weight(name="atten_b", shape=(self.att_size,),
                                           initializer=tf.constant_initializer(0.1), dtype="float32", trainable=True)

    def call(self,input):
        #tensordot可以用于维度不同的矩阵相乘,axis=1表示input的后一维乘以attention_w的前一维
        x=tf.tanh(tf.add(tf.tensordot(input,self.attention_w,axes=1),self.attention_b))#[batch,seqlen,embedim]*[embedim,att_size]+[att_size]
        x=tf.tensordot(x,self.attention_u,axes=1)#[batch,seqlen,att_size]*[att_size]
        x=tf.nn.softmax(x)#[batch,seqlen]
        x=tf.multiply(input,tf.expand_dims(x,-1))#[batch,seqlen,embedim]*[batch,seqlen,1]
        out=tf.reduce_sum(x,axis=1)
        return out

