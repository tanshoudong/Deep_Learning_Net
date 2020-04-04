#coding:utf-8
#file: data_process.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/3/29 22:05
#@desc:
from config import getConfig
import random
import sys
from collections import Counter
import numpy as np
import os
from utils import *
from tensorflow.keras.preprocessing import sequence
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False
config=getConfig.get_config("../config/config.ini")
# 获得 词汇/类别 与id映射字典
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(config['vocab_file'])

# 全部数据
x, y = read_files(config['data_dir'])
data = list(zip(x,y))
del x,y
# 乱序
random.shuffle(data)
# 切分训练集和测试集
train_data, test_data = train_test_split(data)
# 对文本的词id和类别id进行编码
x_train = encode_sentences([content[0] for content in train_data], word_to_id)
y_train = to_categorical(encode_cate([content[1] for content in train_data], cat_to_id))
x_test = encode_sentences([content[0] for content in test_data], word_to_id)
y_test = to_categorical(encode_cate([content[1] for content in test_data], cat_to_id))

print('对序列做padding，保证是 samples*timestep 的维度')
x_train = sequence.pad_sequences(x_train, maxlen=config['maxlen'])
x_test = sequence.pad_sequences(x_test, maxlen=config['maxlen'])
print('x_train shape:', x_train.shape)
print('y_train shape:',y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
np.savez_compressed("data",x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)

