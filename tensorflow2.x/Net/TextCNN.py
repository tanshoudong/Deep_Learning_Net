#coding:utf-8
#file: TextCNN.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/3/31 23:00
#@desc:TextCNN

from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,Embedding,Conv1D,GlobalMaxPooling1D,Concatenate,Dropout

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
                            input_length=self.max_len)(input)
        convs=[]
        for kernel_size in [3,4,5]:
            c=Conv1D(128,kernel_size,activation='relu')(embedding)
            c=GlobalMaxPooling1D()(c)
            convs.append(c)
        x=Concatenate()(convs)
        output=Dense(self.nums_class,activation=self.last_activition)(x)
        model=Model(inputs=input,outputs=output)
        return model
