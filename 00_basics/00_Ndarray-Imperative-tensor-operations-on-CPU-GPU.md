# NDArray-CPU/GPU上的命令式张量操作

在MXNet中，`NDArray`是进行所有数学计算的核心数据结构。一个`NDArray`表示了固定大小、多维的同质数组。如果你对python科学计算库[NumPy](http://www.numpy.org/)很熟悉的话，你可能会注意到`mxnet.ndarray`和`numpy.ndarray`非常相似。和对应的Numpy数据结构一样，`NDArray`也支持命令式计算。

所以你可能会很奇怪，为什么不直接使用Numpy呢？因为和numpy相比，MXNet提供了两种引人注目的优点。首先，MXNet的`NDArray`支持在CUP、GPU和多GPU机器等一系列硬件配置上快速执行，同时它还支持部署在云上的分布式系统。其次，因为MXNet的`NDArray`在执行中是lazy形式的，因此它支持在可用硬件上的自动并行化操作。

`NDArray`是拥有相同类型数字的多维数组。我们可以用1D数组表示3D空间中的点坐标，例如`[2, 1, 6]`就是一个形状为3的1D数组。类似地，我们也可以表示2D数组。下面，我们展示了一个0维长度为2，1维长度为3的2D数组。

```
[[0, 1, 2]
 [3, 4, 5]]
```

请注意，这里“维度”的使用是重载的。当我们说一个2D数组的时候，我们的意思是这个数组有两个轴，而不是有两个元素。

每个NDArray都支持一些你可能会经常调用的重要属性。

- **ndarray.shape**: 数组的维度信息。它是一个整数元组，其中数字的大小代表的数组各个维度的长度信息。对于一个拥有`n`行`m`列的矩阵来说，它的`shape`是`(n, m)`。
- **ndarray.dtype**: 一个`numpy`类型的对象，描述了元素的类型。
- **ndarray.size**: 数组的总元素数目，等于`shape`中各元素的乘积。
- **ndarray.context**: 数组存储的设备，例如`cpu()`或者`gpu(1)`。

## 准备工作

为了完成本教程，我们需要：

- MXNet：你可以在[设置和安装](http://mxnet.io/install/index.html)中查看如何在你的操作系统上安装MXNet
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- GPUs：本教程中的一节会用到GPU，如果你的机器上没有GPU，请将变量gpu_device设置为mx.cpu()。

## 数组创建

我们有一系列的方法可以创建一个`NDArray`。

- 我们可以使用`array`函数从一个常规的python列表或者元组创建一个NDArray：


```python
import mxnet as mx
# create a 1-dimensional array with a python list
a = mx.nd.array([1,2,3])
# create a 2-dimensional array with a nested python list
b = mx.nd.array([[1,2,3], [2,3,4]])
{'a.shape':a.shape, 'b.shape':b.shape}
```

- 我们还可以使用`numpy.ndarray`创建一个MXNet NDArray：


```python
import numpy as np
import math
c = np.arange(15).reshape(3,5)
# create a 2-dimensional array from a numpy.ndarray object
a = mx.nd.array(c)
{'a.shape':a.shape}
```

我们可以通过`dtype`选项指定元素的类型，它可以接受numpy的类型，默认情况下使用`float32`。


```python
# float32 is used by default
a = mx.nd.array([1,2,3])
# create an int32 array
b = mx.nd.array([1,2,3], dtype=np.int32)
# create a 16-bit float array
c = mx.nd.array([1.2, 2.3], dtype=np.float16)
(a.dtype, b.dtype, c.dtype)
```

某些情况下我们知道所需的NDArray的形状，但是不知道其中的元素值，MXNet为这种情况提供了几个函数用于创建带有占位符内容的数组：


```python
# 零张量
a = mx.nd.zeros((2,3))
# 单位张量
b = mx.nd.ones((2,3))
# x填充的张量
c = mx.nd.full((2,3), 7)
# 随机数值的张量
d = mx.nd.empty((2,3))
print(c)
print(d)
```

## 打印数组

在检查`NDArray`的内容时，首先使用`asnumpy`函数将其内容转换成`numpy.ndarray`格式会更加方便。Numpy使用以下布局打印输出：

- 最后一个维度从左到右print
- 倒数第二个维度从上到下print
- 剩余的也是从上到下print，中间有空行分割


```python
b = mx.nd.arange(18).reshape((3,2,3))
b.asnumpy()
print(b)
```

## 基本操作

对NDArray执行操作时，标准的算术运算符执行的是按元素操作。返回值为包含了运算结果的新数组。


```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((2,3))
# 按元素相加
c = a + b
# 按元素相减
d = - c
# 按元素平方，sin，然后转置
e = mx.nd.sin(c**2).T
# 数据广播式的大小比较
f = mx.nd.maximum(a, c)
f.asnumpy()
print(f)
```

在mxnet中，*表示的是元素点乘，而mx.nd.dot()表示的则是矩阵乘法。


```python
a = mx.nd.arange(4).reshape((2,2))
b = a * a
c = mx.nd.dot(a,a)
print("b: %s, \n c: %s" % (b.asnumpy(), c.asnumpy()))
```

使用`+=`和`*=`等赋值运算符会修改数组的内容，不会分配新的内存空间用于创建新数组。


```python
a = mx.nd.ones((2,2))
b = mx.nd.ones(a.shape)
b += a
b.asnumpy()
```

## 索引和切片

切片操作符`[]`在0维上对数组执行操作。


```python
a = mx.nd.array(np.arange(6).reshape(3,2))
a[1:2] = 1
a[:].asnumpy()
```

我们可以使用`slice_axis`函数在特定的维度上执行切片操作。


```python
d = mx.nd.slice_axis(a, axis=1, begin=1, end=2)
# 数组，切片的维度，begin index，end index
d.asnumpy()
```

## 形状重塑

使用`reshape`我们可以改变数组的形状，前提是数组的大小需要保持不变。


```python
a = mx.nd.array(np.arange(24))
b = a.reshape((2,3,4))
b.asnumpy()
```

`concat`函数可以沿着0维将多个数组堆叠起来。但是除了0维之外，其他维度的形状需要保持一致。


```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((2,3))*2
c = mx.nd.concat(a,b)
c.asnumpy()
```

## 缩减

像`sum`和`mean`等函数可以将数组缩减为数字标量。


```python
a = mx.nd.ones((2,3))
b = mx.nd.sum(a)
b.asnumpy()
```

我们也可以沿着特定的维度将数组缩减：


```python
c = mx.nd.sum_axis(a, axis=1)
c.asnumpy()
```

## 广播

我们可以对数组进行广播。广播操作沿着数组中长度为1的维度进行值复制。下面的代码就是沿着1维进行广播：


```python
a = mx.nd.array(np.arange(6).reshape(6,1))
b = a.broadcast_to((6,4))  #
b.asnumpy()
```

数组也可以沿着多个维度进行广播。在下面的例子中，我们沿着1维和2维进行广播：


```python
c = a.reshape((2,1,1,3))
d = c.broadcast_to((2,2,2,3))
d.asnumpy()
```

在执行例如`*`和`+`等作时，广播可以沿着形状不同的维度自动执行。


```python
a = mx.nd.ones((3,2))
b = mx.nd.ones((1,2))
c = a + b
c.asnumpy()
```

## 复制

当我们将NDArray分配给另一个python变量的时候，我们只是对*同一个*NDArray的引用进行了复制。但是很多时候，我们需要对数据进行复制，这样我们就可以对新数组进行操作而不会覆盖原始的数值。


```python
a = mx.nd.ones((2,2))
b = a
b is a # will be True
```

`copy`将会对数组的数据进行深层的复制：


```python
b = a.copy()
b is a  # will be False
```

上面的代码为一个新的NDArray分配了内存，并将其分配给*b*。当我们不想分配额外的内存时，我们可以使用`copyto`方法或者`[]`切片操作。


```python
b = mx.nd.ones(a.shape)
c = b
c[:] = a
d = b
a.copyto(d)
(c is b, d is b)  # Both will be True
```

## 高级主题

MXNet的NDArray提供了一些高级特性，这使得它有别于其他的类似库。

### GPU支持

默认情况下，NDArray的操作符在CPU进行执行。但是在MXNet中，你可以很容易的转而使用其他计算设备，例如GPU（如果有的话）。每一个NDArray的设备信息都存储在`ndarray.context`当中。当MXNet使用标志符`USE_CUDA=1`进行编译，且机器上至少拥有一块NVIDIA显卡的时候，我们可以使用通过设置上下文管理器`mx.gpu(0)`或者更简单地使用`mx.gpu()`将所有的计算都放在GPU上进行。当我们拥有两块或者更多的显卡时，可以通过`mx.gpu(1)`表示第二块显卡，诸如此类。

**注意**想要在cpu上执行下述小节的话请将gpu_device设置为mx.cpu()。


```python
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
#gpu_device=mx.cpu()

def f():
    a = mx.nd.ones((100,100))
    b = mx.nd.ones((100,100))
    c = a + b
    print(c)
# in default mx.cpu() is used
f()
# change the default context to the first GPU
with mx.Context(gpu_device):
    f()
```

我们也可以在创建数组时明确指定context。


```python
a = mx.nd.ones((100, 100), gpu_device)
a
```

目前，MXNet需要进行计算的两个数组位于同一个设备上。有几种办法可以在设备之间进行数据复制。


```python
a = mx.nd.ones((100,100), mx.cpu())
b = mx.nd.ones((100,100), gpu_device)
c = mx.nd.ones((100,100), gpu_device)
a.copyto(c)  # copy from CPU to GPU 从cpu复制到gpu
d = b + c
e = b.as_in_context(c.context) + c  # same to above 同上
{'d':d, 'e':e}
```

### 分布式文件系统的序列化

MXNet提供了两种简单的方式用于从硬盘加载数据或者将数据保存到硬盘上。第一种方式就像在python中处理其他对象一样，使用`pickle`。`NDArray`兼容pickle的使用。


```python
import pickle as pkl
a = mx.nd.ones((2, 3))
# pack and then dump into disk
data = pkl.dumps(a)
pkl.dump(data, open('tmp.pickle', 'wb'))
# load from disk and then unpack
data = pkl.load(open('tmp.pickle', 'rb'))
b = pkl.loads(data)
b.asnumpy()
```

第二种方法是使用`save`和`load`方法将数组直接以二进制的形式存储到硬盘上。我们可以保存/加载一个NDArray或者一系列的NDArray。


```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((5,6))
mx.nd.save("temp.ndarray", [a,b])
c = mx.nd.load("temp.ndarray")
c
```

在MXNet中，你也可以以这种实行保存/加载一个NDArray的字典：


```python
d = {'a':a, 'b':b}
mx.nd.save("temp.ndarray", d)
c = mx.nd.load("temp.ndarray")
c
```

这种`load`和`save`的方法在两个方面优于pickle：
- 当使用这类方法的时候，你可以在python的界面中保存数据并在其他语言中调用数据。例如，如果我们在python中保存这些数据：


```python
a = mx.nd.ones((2, 3))
mx.nd.save("temp.ndarray", [a,])
```

我们之后可以在R中加载它：
```
a <- mx.nd.load("temp.ndarray")
as.array(a[[1]])
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

- 当使用 Amazon S3或者Hadoop HDFS这样的分布式文件系统的时候，我们可以直接保存在上面进行数据的保存和加载。

```
mx.nd.save('s3://mybucket/mydata.ndarray', [a,])  # 如果使用USE_S3=1编译
mx.nd.save('hdfs///users/myname/mydata.bin', [a,])  # 如果使用USE_HDFS=1编译
```

### 延迟执行和自动并行化

MXNet使用延迟执行来达到更好的性能。当我们在python中运行`a=b+1`的时候，python线程将操作推入后台引擎然后获取返回的结果。这种方式有两种好处：

- python主线程可以继续执行其他计算直到前一个执行完毕。这对于开销繁重的前端语言非常有用。
- 它让后端引擎更容易进行进一步的优化，例如自动并行化。

后端引擎可以解决数据依赖的问题并直接对计算进行规划。这对于用户也是透明的。我们可以在结果数组上调用`wait_to_read`方法来等待计算完成。将数据从数组复制到其他包上的操作会隐式的调用`wait_to_read`方法。


```python
import time
def do(x, n):
    """push computation into the backend engine"""
    return [mx.nd.dot(x,x) for i in range(n)]
def wait(x):
    """wait until all results are available"""
    for y in x:
        y.wait_to_read()

tic = time.time()
a = mx.nd.ones((1000,1000))
b = do(a, 50)
print('time for all computations are pushed into the backend engine:\n %f sec' % (time.time() - tic))
wait(b)
print('time for all computations are finished:\n %f sec' % (time.time() - tic))
```

除了能够分析数据读取和写入的依赖之外，后端引擎还能够并行规划没有依赖性的计算。例如，在下面的代码中：


```python
a = mx.nd.ones((2,3))
b = a + 1
c = a + 2
d = b * c
```

第二行和第三行的代码可以并行执行。下面的例子先后在CPU和GPU上运行：


```python
n = 10
a = mx.nd.ones((1000,1000))
b = mx.nd.ones((6000,6000), gpu_device)
tic = time.time()
c = do(a, n)
wait(c)
print('Time to finish the CPU workload: %f sec' % (time.time() - tic))
d = do(b, n)
wait(d)
print('Time to finish both CPU/GPU workloads: %f sec' % (time.time() - tic))
```

现在我们发布了所有工作负载。后端引擎将会尝试并行CPU和GPU的计算。


```python
tic = time.time()
c = do(a, n)
d = do(b, n)
wait(c)
wait(d)
print('Both as finished in: %f sec' % (time.time() - tic))
```


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
