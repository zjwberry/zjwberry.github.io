'''
LastEditors: jingweizhu
'''
import sys
import paddle.fluid as fluid

def main():
    # 定义变量
    data = fluid.layers.fill_constant(shape=[3, 4], value=16, dtype="int64")
    print data2
    print type(data)
    
    # 创建打印操作，获取Tensor数值
    data = fluid.layers.Print(data, message="Print data:")
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
    return exe.run()
    
if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
