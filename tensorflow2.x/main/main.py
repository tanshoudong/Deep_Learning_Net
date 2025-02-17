#coding:utf-8
#file: main.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/3/29 21:45
#@desc:
from Net.Net import *
import numpy as np
from config import getConfig
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

def execute():
    config = getConfig.get_config("../config/config.ini")
    print("构建模型...")
    model=TextAttBiRNN(config["maxlen"],config["max_features"],config["embedding_dims"]).get_model()
    model.compile('adam','categorical_crossentropy',metrics=["accuracy"])
    print(model.summary())
    #设置callbacks回调函数
    my_callbacks=[ModelCheckpoint("./cnn_model.h5",verbose=1),
                  EarlyStopping(monitor='val_accuracy',patience=2,mode="max")]
    #读取数据
    data=np.load("../data/data.npz")
    x_train,y_train=data["x_train"],data["y_train"]
    x_test,y_test=data["x_test"],data["y_test"]

    current_test = x_test
    left_test = np.hstack([current_test[:, 0:1], current_test[:, 0:-1]])
    right_test = np.hstack([current_test[:, 1:], current_test[:, -1:]])

    current=x_train
    left=np.hstack([current[:,0:1],current[:,0:-1]])
    right=np.hstack([current[:,1:],current[:,-1:]])

    #拟合数据
    model.fit(x_train,y_train,batch_size=config["batch_size"],
              epochs=config["epochs"],callbacks=my_callbacks,
              validation_data=(x_test,y_test))


if __name__ =='__main__':
    execute()


