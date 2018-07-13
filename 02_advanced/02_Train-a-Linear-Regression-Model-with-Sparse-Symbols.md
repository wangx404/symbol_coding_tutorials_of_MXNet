
# 使用稀疏符号式编程训练线性回归模型

在之前的两章中，我们介绍了`CSRNDArray`和`RowSparseNDArray`两种用于处理稀疏数据的基本数据格式。MXNet同时还提供了`Sparse Symbol` API，它使得我们能够使用符号表达式处理稀疏数组。在本教程中，我们将首先关注如何使用稀疏操作符组成符号式计算图，然后再使用Module API提供的稀疏符号训练一个线性回归模型。

## 准备工作

为了完成本章的教程，我们需要：

- MXNet. 请参阅[设置和安装](https://mxnet.io/install/index.html)中的操作系统说明。
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- MXNet符号式编程的基础知识。详细内容请参阅[Symbol-神经网络计算图和自动微分](https://mxnet.incubator.apache.org/tutorials/basic/symbol.html)
- 关于CSRNDArray的基础知识。详细内容请参阅[CSRNDArray-压缩的稀疏行存储格式的NDArray](https://mxnet.incubator.apache.org/versions/master/tutorials/sparse/csr.html)。
- 关于RowSparseNDArray的基础知识。详细内容请参阅[RowSparseNDArray-稀疏梯度更新中的NDArray](https://mxnet.incubator.apache.org/versions/master/tutorials/sparse/row_sparse.html)。

## 变量

变量是用来表示数组的占位符。我们也可以用它来表示稀疏数组。

### 变量存储类型

变量的`stype`属性被用来指示数组的存储类型。默认情况下，变量的`stype`属性是“default”，表明变量为默认的密集存储格式。我们可以指定变量的`stype`属性为“csr”或者“row_sparse”来存储稀疏数组。


```python
import mxnet as mx
# Create a variable to hold an NDArray
a = mx.sym.Variable('a')
# Create a variable to hold a CSRNDArray
b = mx.sym.Variable('b', stype='csr')
# Create a variable to hold a RowSparseNDArray
c = mx.sym.Variable('c', stype='row_sparse')
(a, b, c)
```




    (<Symbol a>, <Symbol b>, <Symbol c>)



### 绑定稀疏数组

上面构造的稀疏符号声明了所要保存的稀疏数组的存储类型。为了评估它们，我们需要为这些自由变量提供稀疏数据。

你可以通过`simple_bind`方法将一个稀疏符号实例化为一个executor，实例化过程中它将根据自有变量的存储类型为其分配0。executor提供了一个用于评估的`forward`方法和一个用于获取所有结果的`outputs`属性。稍后，我们将展示如何使用`backward`和其他的方法来计算梯度以及更新参数。首先来看一个简单的示例：


```python
shape = (2,2)
# Instantiate an executor from sparse symbols
b_exec = b.simple_bind(ctx=mx.cpu(), b=shape)
c_exec = c.simple_bind(ctx=mx.cpu(), c=shape)
# Sparse arrays of zeros are bound to b and c
b_exec.forward()
c_exec.forward()
print(b_exec.outputs, c_exec.outputs)
```

    [
    <CSRNDArray 2x2 @cpu(0)>] [
    <RowSparseNDArray 2x2 @cpu(0)>]


你可以通过访问executor的`arg_dict`并为其分配新值来更新变量所拥有的数组。


```python
b_exec.arg_dict['b'][:] = mx.nd.ones(shape).tostype('csr')
b_exec.forward()
# The array `b` holds are updated to be ones
eval_b = b_exec.outputs[0]
{'eval_b': eval_b, 'eval_b.asnumpy()': eval_b.asnumpy()}
```




    {'eval_b': 
     <CSRNDArray 2x2 @cpu(0)>, 'eval_b.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32)}



## 符号组成和存储类型推断

### 基础符号组成

下面的例子使用不同的存储类型构建了一个简单的按元素加法的表达式。你可以在`mx.sym.sparse`包中找到这些稀疏符号。


```python
# Element-wise addition of variables with "default" stype
d = mx.sym.elemwise_add(a, a)
# Element-wise addition of variables with "csr" stype
e = mx.sym.sparse.negative(b)
# Element-wise addition of variables with "row_sparse" stype
f = mx.sym.sparse.elemwise_add(c, c)
{'d':d, 'e':e, 'f':f}
```




    {'d': <Symbol elemwise_add0>,
     'e': <Symbol negative0>,
     'f': <Symbol elemwise_add1>}



### 存储类型推断

稀疏符号的输出存储类型是什么呢？在MXNet中，对于所有的稀疏符号，结果的存储类型是基于输入的存储类型推断出来的。你可以通过阅读[Sparse Symbol API](http://mxnet.io/versions/master/api/python/symbol/sparse.html)文档来了解输出的存储类型是什么。在下面的例子中，我们将尝试一下在行稀疏和压缩行稀疏教程中所介绍的数据存储格式：`default`(稠密)，`csr`和`row_sparse`。


```python
add_exec = mx.sym.Group([d, e, f]).simple_bind(ctx=mx.cpu(), a=shape, b=shape, c=shape)
add_exec.forward()
dense_add = add_exec.outputs[0]
# The output storage type of elemwise_add(csr, csr) will be inferred as "csr"
csr_add = add_exec.outputs[1]
# The output storage type of elemwise_add(row_sparse, row_sparse) will be inferred as "row_sparse"
rsp_add = add_exec.outputs[2]
{'dense_add.stype': dense_add.stype, 'csr_add.stype':csr_add.stype, 'rsp_add.stype': rsp_add.stype}
```




    {'csr_add.stype': 'csr',
     'dense_add.stype': 'default',
     'rsp_add.stype': 'row_sparse'}



### 存储类型回退

对于那些不专注于稀疏数组的操作符，你仍然可以在稀疏输入上调用它们，只不过此时会有一些效率损失。在MXNet中，稠密操作符需要输入和输出都是稠密格式。如果提供的输入是稀疏格式，那么MXNet会暂时将其转换成稠密格式，然后才能调用稠密操作符。如果提供的输出是稀疏格式，那么MXNet会将稠密格式的输出转换为所提供的稀疏格式。当这样的存储类型回退发生时，警告信息会被打印出来。


```python
# `log` operator doesn't support sparse inputs at all, but we can fallback on the dense implementation
csr_log = mx.sym.log(a)
# `elemwise_add` operator doesn't support adding csr with row_sparse, but we can fallback on the dense implementation
csr_rsp_add = mx.sym.elemwise_add(b, c)

fallback_exec = mx.sym.Group([csr_rsp_add, csr_log]).simple_bind(ctx=mx.cpu(), a=shape, b=shape, c=shape)
fallback_exec.forward()

fallback_add = fallback_exec.outputs[0]
fallback_log = fallback_exec.outputs[1]
{'fallback_add': fallback_add, 'fallback_log': fallback_log}
```




    {'fallback_add': 
     [[ 0.  0.]
      [ 0.  0.]]
     <NDArray 2x2 @cpu(0)>, 'fallback_log': 
     [[-inf -inf]
      [-inf -inf]]
     <NDArray 2x2 @cpu(0)>}



### 检查符号计算图的存储类型（建构中）

当环境变量`MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING`被设置为`1`时，MXNet将会记录计算图中操作符的输入和输出的存储类型信息。例如，我们可以如下所示地检查包含稀疏操作符的线性分类网络的存储类型：


```python
# Set logging level for executor
import mxnet as mx
import os
os.environ['MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING'] = "1"
# Data in csr format
data = mx.sym.var('data', stype='csr', shape=(32, 10000))
# Weight in row_sparse format
weight = mx.sym.var('weight', stype='row_sparse', shape=(10000, 2))
bias = mx.symbol.Variable("bias", shape=(2,))

dot = mx.symbol.sparse.dot(data, weight)
pred = mx.symbol.broadcast_add(dot, bias)
y = mx.symbol.Variable("label")
output = mx.symbol.SoftmaxOutput(data=pred, label=y, name="output")
executor = output.simple_bind(ctx=mx.cpu())
```

## 使用Module API进行训练

在下面的小节中，我们将使用稀疏符号和稀疏优化来实现**线性回归**。

你将探索的函数是：*y = x<sub>1</sub>  +  2x<sub>2</sub> + ... 100x<sub>100*，其中*(x<sub>1</sub>,x<sub>2</sub>, ..., x<sub>100</sub>)*为输入的特征，而*y*是对应的标签。

### 准备数据

在MXNet中，[mx.io.LibSVMIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter)和[mx.io.NDArrayIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.NDArrayIter)都支持加载CSR格式的稀疏数据。在本例中我们将使用`NDArrayIter`。

你可能会看到一些来自scipy的警告信息，但是在本例中你不需要担心这些。


```python
# 随机的训练数据
feature_dimension = 100
train_data = mx.test_utils.rand_ndarray((1000, feature_dimension), 'csr', 0.01)
target_weight = mx.nd.arange(1, feature_dimension + 1).reshape((feature_dimension, 1))
train_label = mx.nd.dot(train_data, target_weight)
batch_size = 1
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, last_batch_handle='discard', label_name='label')
```

### 定义模型

下面是一个指定了变量存储类型的线性回归模型的例子。


```python
initializer = mx.initializer.Normal(sigma=0.01)
X = mx.sym.Variable('data', stype='csr')
Y = mx.symbol.Variable('label')
weight = mx.symbol.Variable('weight', stype='row_sparse', shape=(feature_dimension, 1), init=initializer)
bias = mx.symbol.Variable('bias', shape=(1, ))
pred = mx.sym.broadcast_add(mx.sym.sparse.dot(X, weight), bias)
lro = mx.sym.LinearRegressionOutput(data=pred, label=Y, name="lro")
```

上面的网络使用了下列符号：

1. `变量X`:稀疏输入数据的占位符，`csr`表明要保存的数组是CSR格式。
2. `变量Y`:稠密格式标签的占位符。
3. `变量weight`:将要学习的权重的占位符。权重的`stype`属性被指定为`row_sparse`，表明它将被初始化为RowSparseNDArray，之后优化器将根据规则对其执行稀疏更新。`init`属性指定了用于此变量初始化的初始化程序。
4. `变量bias`:将要学习的偏置的占位符。
5. `sparse.dot`:`X`和`weight`的点乘操作，用于处理`csr`和`row_sparse`输入的稀疏实现。
6. `broadcast_add`:添加`bias`的广播相加操作。
7. `LinearRegressionOutput`:输出层将根据提供给它的输入和标签计算*l2*损失。

### 训练模型

完成模型结构的定义之后，下一步创建一个模型并对参数和优化器进行初始化。


```python
# Create module
mod = mx.mod.Module(symbol=lro, data_names=['data'], label_names=['label'])
# Allocate memory by giving the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# Initialize parameters by random numbers
mod.init_params(initializer=initializer)
# Use SGD as the optimizer, which performs sparse update on "row_sparse" weight
sgd = mx.optimizer.SGD(learning_rate=0.05, rescale_grad=1.0/batch_size, momentum=0.9)
mod.init_optimizer(optimizer=sgd)
```

最后，我们使用Module中提供的`forward`，`backward`和`update`方法来训练模型的参数以拟合训练数据。


```python
# Use mean square error as the metric
metric = mx.metric.create('MSE')
# Train 10 epochs
for epoch in range(10):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Metric = %s' % (epoch, metric.get()))
assert metric.get()[1] < 1, "Achieved MSE (%f) is larger than expected (1.0)" % metric.get()[1]    
```

    Epoch 0, Metric = ('mse', 916.63147034091503)
    Epoch 1, Metric = ('mse', 153.48963233946529)
    Epoch 2, Metric = ('mse', 74.462100808457961)
    Epoch 3, Metric = ('mse', 30.766911837883526)
    Epoch 4, Metric = ('mse', 15.282920648318267)
    Epoch 5, Metric = ('mse', 6.6387904795454613)
    Epoch 6, Metric = ('mse', 2.5454614988268887)
    Epoch 7, Metric = ('mse', 1.2938379316335931)
    Epoch 8, Metric = ('mse', 0.65615797590451597)
    Epoch 9, Metric = ('mse', 0.43595944697025807)


### 使用多机训练

想要使用多机训练稀疏模型，请参考[mxnet/example/sparse/](https://github.com/apache/incubator-mxnet/tree/master/example/sparse)中的例子。

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
