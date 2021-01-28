# -*- coding=utf-8 -*-

'''
LastEditors: jingweizhu
'''

import sys
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np

def main():
    # 定义变量
    a = fluid.data(name="a", shape=[None, 1], dtype="int64")
    b = fluid.data(name="b", shape=[None, 1], dtype="int64")

    # 组建网络
    result = layers.elementwise_add(a, b)

    # 准备运行网络
    cpu = fluid.CPUPlace() # 定义运算设备，这里选择在CPU下训练
    exe = fluid.Executor(cpu) # 创建执行器
    #exe.run(fluid.default_startup_program()) # 网络参数初始化

    # 读取输入数据
    data_1 = int(input("Please enter an integer: a="))
    data_2 = int(input("Please enter an integer: b="))
    x = np.array([[data_1]])
    y = np.array([[data_2]])

    # 运行网络
    outs = exe.run(
        feed={'a':x, 'b':y}, # 将输入数据分别赋值给变量a, b
        fetch_list=[result, a, b]
    )

    # 输出运算结果
    print outs
    return 0


if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
    
