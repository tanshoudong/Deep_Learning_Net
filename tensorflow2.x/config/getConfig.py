#coding:utf-8
#file: getConfig.py.py
#@author: tan
#@contact: 2252437684@qq.com
#@time: 2020/3/29 21:53
#@desc:

import configparser
def get_config(config_file):
    parser=configparser.ConfigParser()
    parser.read(config_file)
    _conf_str=[(key,value) for key,value in parser.items('strings')]
    _conf_ints=[(key,int(value)) for key,value in parser.items('ints')]
    return dict(_conf_ints+_conf_str)
