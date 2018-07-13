
# 行稀疏NDArray-稀疏梯度更新中的NDArray

## 动机
在现实世界中，很多数据集要处理的是高纬的稀疏特征向量。当使用稀疏的数据集进行模型权重学习时，由此导出的权重梯度也会是稀疏的。

假设我们要执行矩阵``X``和``W``的乘法，其中``X``是一个1x2的矩阵，``W``是一个2x3的矩阵，而``Y``则是两者相乘的结果。


```python
import mxnet as mx
X = mx.nd.array([[1,0]])
W = mx.nd.array([[3,4,5], [6,7,8]])
Y = mx.nd.dot(X, W)
{'X': X, 'W': W, 'Y': Y}
```

你可以看到，

```
Y[0][0] = X[0][0] * W[0][0] + X[0][1] * W[1][0] = 1 * 3 + 0 * 6 = 3
Y[0][1] = X[0][0] * W[0][1] + X[0][1] * W[1][1] = 1 * 4 + 0 * 7 = 4
Y[0][2] = X[0][0] * W[0][2] + X[0][1] * W[1][2] = 1 * 5 + 0 * 8 = 5
```

那么dY/dW，矩阵``W``的梯度呢？我们姑且称之为``grad_W``。首先，``grad_W``的形状和``W``是相同的，都是2x3，因为它是矩阵``W``的导数。接着，我们可以如下求出``grad_W``中的元素值：

```
grad_W[0][0] = X[0][0] = 1
grad_W[0][1] = X[0][0] = 1
grad_W[0][2] = X[0][0] = 1
grad_W[1][0] = X[0][1] = 0
grad_W[1][1] = X[0][1] = 0
grad_W[1][2] = X[0][1] = 0
```

但事实上，你可以通过把矩阵``X``的转置和一个单位矩阵相乘得到``grad_W``。


```python
grad_W = mx.nd.dot(X, mx.nd.ones_like(Y), transpose_a=True)
grad_W
```

正如你所见的那样，``grad_W``的第0行填充的是非零值，而第1行则不是这样。为什么会这样呢？如果你看一下``grad_W``是如何被计算出来的，你会注意到，由于``X``的第1列的元素均为0，所以``grad_W``的第1行的元素也都是0。

在现实世界中，那些和稀疏输入进行交互的参数通常具有这样的梯度，其中很多行切片的元素全部是0。使用默认的稠密矩阵结构储存和操纵这样的具有很多全零元素行切片的稀疏矩阵会造成内存的浪费和大量的零处理。更重要的是，许多基于梯度进行优化的方法，例如SGD，[AdaGrad](https://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf)和[Adam](https://arxiv.org/pdf/1412.6980.pdf)，都能够利用稀疏梯度的特性，从而被证明是高效且有效的。**在MXNet中，``RowSparseNDArray``以``row sparse``的格式存储矩阵，它是为那些大多数行切片元素为零的数组所设计的。**在本教程中，我们将描述行稀疏格式是什么，以及如何在MXNet中利用RowSparseNDArray进行稀疏梯度更新。


## 准备工作

为了完成本教程，我们需要：

- MXNet. 请参阅[设置和安装](https://mxnet.io/install/index.html)中的操作系统说明。
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- 关于MXNet中NDArray的基础知识。详细教程可以参阅[NDArray-CPU/GPU上的命令式张量操作](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html)
- 理解[如何使用autograd进行自动微分](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)
- GPUs：本教程中的一节将会用到GPU。如果你的机器上没有GPU，请简单的将变量`gpu_device`设置为 `mx.cpu()`。

## 行稀疏格式

RowSparseNDArray使用两个分离的1D数组`数据`和`索引`来表示多维的NDArray。

- 数据：形状为`[D0, D1, ..., Dn]`任意数据类型的NDArray。
- 索引：形状为`[D0]`的1D int64 NDArray，其值按照升序排列。

``indices``数组中存储着非零行切片的索引值，而实际数值存储在``data``数组中。使用RowSparseNDArray `rsp`来表示对应的稠密NDArray的形式如下：``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``。

RowSparseNDArray通常用来表示形状为[LARGE0, D1, .. , Dn]的大型NDArray中的非零切片，其中LARGE0远远大于D0，且大多数行切片为0。

如下为一个2D的矩阵：


```python
[[ 1, 2, 3],
 [ 0, 0, 0],
 [ 4, 0, 5],
 [ 0, 0, 0],
 [ 0, 0, 0]]
```

稀疏行表示的形式如下：
- `data`数组中存储着数组的所有非零行切片。
- `indices`数组中存储着具有非零元素的行切片的索引。


```python
data = [[1, 2, 3], [4, 0, 5]]
indices = [0, 2]
```

`RowSparseNDArray`同样支持多维数组，如下为一个3D的张量：


```python
[[[1, 0],
  [0, 2],
  [3, 4]],

 [[5, 0],
  [6, 0],
  [0, 0]],

 [[0, 0],
  [0, 0],
  [0, 0]]]
```

行稀疏表示则是（其中`data`和`indices`的定义如上）：


```python
data = [[[1, 0], [0, 2], [3, 4]], [[5, 0], [6, 0], [0, 0]]]
indices = [0, 1]
```

``RowSparseNDArray``是``NDArray``的子类。如果你查询RowSparseNDArray的**stype**，返回值将会是**"row_sparse"**。

## 数组创建

你可以通过`row_sparse_array`函数把数据和索引转换为`RowSparseNDArray`。


```python
import mxnet as mx
import numpy as np
# Create a RowSparseNDArray with python lists
shape = (6, 2)
data_list = [[1, 2], [3, 4]]
indices_list = [1, 4]
a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
# Create a RowSparseNDArray with numpy arrays
data_np = np.array([[1, 2], [3, 4]])
indices_np = np.array([1, 4])
b = mx.nd.sparse.row_sparse_array((data_np, indices_np), shape=shape)
{'a':a, 'b':b}
```

## 函数概述

和`CSRNDArray`类似，`RowSparseNDArray`有若干个调用方式相同的函数。在下面的代码块中，你可以尝试一下常用功能。

- **.dtype** - 设置数据类型
- **.asnumpy** - 转换为numpy数组以检查其内容
- **.data** - 取出数据数组
- **.indices** - 取出索引数组
- **.tostype** - 设置存储类型
- **.cast_storage** - 转换存储类型
- **.copy** - 复制数组
- **.copyto** - 将数组复制到现有数组中

## 设置数据类型

从另一个`RowSparseNDArray`创建新的`RowSparseNDArray`时你可以通过`dtype`指定元素的数据类型，它能接受numpy的数据类型。默认情况下会使用`float32`。


```python
# Float32 is used by default
c = mx.nd.sparse.array(a)
# Create a 16-bit float array
d = mx.nd.array(a, dtype=np.float16)
(c.dtype, d.dtype)
```

## 检查数组
和`CSRNDArray`一样，你可以通过`asnumpy`函数将`RowSparseNDArray`转换为稠密的`numpy.ndarray`从而检查其中的内容。


```python
a.asnumpy()
```

你也可以通过`indices`和`data`属性来检查RowSparseNDArray的内部存储。


```python
# Access data array
data = a.data
# Access indices array
indices = a.indices
{'a.stype': a.stype, 'data':data, 'indices':indices}
```

## 存储类型转换

你可以通过`tostype`函数将NDArray转换为RowSparseNDArray，反之亦然。


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type from `default` to `row_sparse`
rsp = ones.tostype('row_sparse')
# Cast the storage type from `row_sparse` to `default`
dense = rsp.tostype('default')
{'rsp':rsp, 'dense':dense}
```

你也可以通过`cast_storage`操作转换存储类型。


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type to `row_sparse`
rsp = mx.nd.sparse.cast_storage(ones, 'row_sparse')
# Cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(rsp, 'default')
{'rsp':rsp, 'dense':dense}
```

## 复制

你可以使用`copy`方法来深层复制数据，这个方法将返回一个新的数组。你也可以使用`copyto`方法或者切片`[]`将数据复制到现有的数组中。


```python
a = mx.nd.ones((2,2)).tostype('row_sparse')
b = a.copy()
c = mx.nd.sparse.zeros('row_sparse', (2,2))
c[:] = a
d = mx.nd.sparse.zeros('row_sparse', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```

当源数组的存储类型和目标数组的存储类型不一致时，如果使用的是`copyto`函数或者切片`[]`，那么目标数组的存储类型不会发生改变。在复制之前，源数组的存储类型会暂时转换成所需要的存储类型。


```python
e = mx.nd.sparse.zeros('row_sparse', (2,2))
f = mx.nd.sparse.zeros('row_sparse', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype, 'g.stype':g.stype}
```

## 保留行切片

你可以通过指定行索引的形式从RowSparseNDArray中保留行切片的子集。


```python
data = [[1, 2], [3, 4], [5, 6]]
indices = [0, 2, 3]
rsp = mx.nd.sparse.row_sparse_array((data, indices), shape=(5, 2))
# Retain row 0 and row 1
# 保留行0和行1
rsp_retained = mx.nd.sparse.retain(rsp, mx.nd.array([0, 1]))
{'rsp.asnumpy()': rsp.asnumpy(), 'rsp_retained': rsp_retained, 'rsp_retained.asnumpy()': rsp_retained.asnumpy()}
```

## 稀疏运算符和存储类型推断

我们在`mx.nd.sparse`中实现了专门为稀疏数组而设计的运算符。你可以通过查阅[mxnet.ndarray.sparse API文档](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html)来找到哪些稀疏运算符是可用的。


```python
shape = (3, 5)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
# A csr matrix as lhs 
lhs = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
# A dense matrix as rhs
rhs = mx.nd.ones((3, 2))
# row_sparse result is inferred from sparse operator dot(csr.T, dense) based on input stypes
transpose_dot = mx.nd.sparse.dot(lhs, rhs, transpose_a=True)
{'transpose_dot': transpose_dot, 'transpose_dot.asnumpy()': transpose_dot.asnumpy()}
```

对于任何稀疏运算符，输出数组的存储类型是通过输入来推断的。你可以通过查阅文档或者检查输出数组的stype属性的方式来确定推断出的存储类型格式。


```python
a = transpose_dot.copy()
b = a * 2  # b will be a RowSparseNDArray since zero multiplied by 2 is still zero
c = a + mx.nd.ones((5, 2))  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```

对于那些并非为稀疏数组而设计的运算符，我们仍然可以在稀疏数组上调用它们，只是此时会有性能的损失。在MXNet中，稠密运算符要求所有的输入和输出数组都是稠密格式。

如果提供了稀疏格式的输入，MXNet会将稀疏格式的输入暂时转换为稠密格式，以便可以使用稠密运算符。

如果提供了稀疏格式的输出，MXNet会将稠密运算符生成的稠密输出转化为提供的稀疏格式。


```python
e = mx.nd.sparse.zeros('row_sparse', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd.stype':d.stype, 'e.stype':e.stype} # stypes of a and e will be not changed
```

注意，当发生此类存储回退事件时，警告信息将被打印出来。如果你在使用jupyter notebook，这些警告信息将被打印在终端控制台中。

## 稀疏优化

在MXNet中，只有当权重、状态和梯度全部以`row_sparse`格式存储时才会应用稀疏梯度更新。对权重和状态进行更新时，稀疏优化器只更新那些行索引出现在`gradient.indices`中的行切片。例如，SGD的默认更新规则为：

```
rescaled_grad = learning_rate * rescale_grad * clip(grad, clip_gradient) + weight_decay * weight
state = momentum * state + rescaled_grad
weight = weight - state
```
同时SGD优化器的稀疏更新规则是：

```
for row in grad.indices:
    rescaled_grad[row] = learning_rate * rescale_grad * clip(grad[row], clip_gradient) + weight_decay * weight[row]
    state[row] = momentum[row] * state[row] + rescaled_grad[row]
    weight[row] = weight[row] - state[row]
```


```python
# Create weight
shape = (4, 2)
weight = mx.nd.ones(shape).tostype('row_sparse')
# Create gradient
data = [[1, 2], [4, 5]]
indices = [1, 2]
grad = mx.nd.sparse.row_sparse_array((data, indices), shape=shape)
sgd = mx.optimizer.SGD(learning_rate=0.01, momentum=0.01)
# Create momentum
momentum = sgd.create_state(0, weight)
# Before the update
{"grad.asnumpy()":grad.asnumpy(), "weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```


```python
sgd.update(0, weight, grad, momentum)
# Only row 0 and row 2 are updated for both weight and momentum
{"weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```

在MXNet中，[mxnet.optimizer.SGD](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.SGD)和[mxnet.optimizer.Adam](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.Adam)都支持稀疏更新。

## 高级主题

### GPU支持

默认情况下，RowSparseNDArray操作符在CPU上执行。在MXNet中，对于RowSparseNDArray的GPU支持仍然是实验性质的，只有像cast_storage和dot少数一些操作才支持在GPU上执行。

想要在GPU上创建RowSparseNDArray，我们需要指定context。

**注意**在下一节中，如果GPU不可用，那么代码将会报告一个错误。若想要在CPU上执行此操作，请将gpu_device设置为mx.cpu()。


```python
import sys
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
try:
    a = mx.nd.sparse.zeros('row_sparse', (100, 100), ctx=gpu_device)
    a
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```
