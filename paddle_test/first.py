'''
LastEditors: jingweizhu
'''
import sys
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np

def main():
    x_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype("float32")
    y_true = np.array([[1.0], [2.0], [3.0], [4.0]]).astype("float32")
    
    x = layers.data(name="x", shape=[1], dtype="float32")
    y = layers.data(name="y", shape=[1], dtype="float32")
    y_predict = layers.fc(input=x, size=1, act=None)
    
    cost = layers.square_error_cost(input=y_predict, label=y)
    avg_cost = layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)
    
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())
    
    for i in range(100):
        outs = exe.run(feed={'x':x_data, "y":y_true}, fetch_list=[y_predict.name, avg_cost])
    
        #print outs[0]
        #print outs[0].shape
        #print outs[1]
        #print outs[1].shape
        print outs[0]
        #print outs[2].shape
    

    """
    a = layers.data(name="a", shape=[1, 2], dtype="float32")
    b = layers.data(name="b", shape=[1, 2], dtype="float32")
    result = layers.sum([a, b])

    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    x = np.array([1, 2])
    print x.shape
    y = np.array([[2], [3]])
    print y.shape
    

    outs = exe.run(feed={"a": x, "b":y}, fetch_list=[a, b, result])

    print outs[0]
    print outs[1]
    print outs[2]
    """
    
if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
