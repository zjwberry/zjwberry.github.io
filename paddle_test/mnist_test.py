# -*- coding=utf-8 -*-
'''
LastEditors: jingweizhu
'''

import numpy as np
import paddle

# 读取mnist数据训练集
train_set = paddle.dataset.mnist.train()
# 包装数据读取器，train_reader 是 生成器(PS: python里面yield关键字生成器，生成器是一个函数，该函数返回一个迭代器)
train_reader = paddle.batch(train_set, batch_size=8)


for batch_id, data in enumerate(train_reader()):
    print type(data), len(data)
    print type(data[0]), len(data[0])
    a = data[0][0]
    b = data[0][1]    
    print type(a), len(a), a.shape
    print type(b), b
    print a
    paddle.fluid.dygraph.nn.Linear
    break