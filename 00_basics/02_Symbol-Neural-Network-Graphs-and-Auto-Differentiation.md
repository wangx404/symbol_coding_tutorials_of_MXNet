
# Symbol-神经网络图和自动微分

在[之前的章节](http://mxnet.io/tutorials/basic/ndarray.html)中我们介绍了`NDArray`这个在MXNet中用于操作数据的基本数据结构。仅通过使用NDArray，我们可以执行一系列数学运算。事实上，我们可以通过使用`NDArray`来定义和更新一个完整的神经网络。`NDArray`让我们可以以命令式的方式编写科学计算程序，我们可以在其中充分感受到前端语言的易用性。所以你可能会感到很奇怪，为什么我们不在所有的计算中都使用`NDArray`呢？

MXNet提供了用于符号式编程的接口Symbol API。在符号式编程的过程中，我们首先定义一个*计算图*，而不是一步步的执行计算。计算图中包含了用于代表输入输出数据的占位符。然后通过编译计算图，生成一个函数，和`NDArray`绑定后即可运行。MXNet的Symbol API和[Caffe](http://caffe.berkeleyvision.org/)使用的网络配置以及[Theano](http://deeplearning.net/software/theano/)中的符号式编程很相似。

使用符号式编程的另一个优点是我们可以在执行之前对函数进行优化。举例来说，当我们使用命令式编程执行数学计算时，在每执行完一次计算之后，我们不知道后续将会用到哪些值。但是在符号式变成中，我们事先声明了我们需要的输出。这就意味着在中间步骤中，我们可以通过执行一些操作回收分配的内存。同时，对于同样的网络，Symbol API使用的内存更少。关于这一点想了解更多的话，请参阅[How To](http://mxnet.io/faq/index.html)和[结构](http://mxnet.io/architecture/index.html)。

在我们的设计说明中，我们公布了一份[关于命令式编程和符号式编程比较优势的详细讨论](http://mxnet.io/architecture/program_model.html)。当时在本文档中，我们将集中在如何使用MXNet的Symbol API。在MXNet中，我们可以使用其他的符号，例如计算符（例如简单的矩阵计算符“+”等）或者整个的神经网络层（例如卷积层等），来组成新的符号。计算符可以接受多个输入变量，返回多个输出变量，并且保持内部状态变量。

想要了解这些概念的可视化解释，请参阅[符号式编译和执行](http://mxnet.io/api/python/symbol_in_pictures/symbol_in_pictures.html)。

为了更实在地了解相关内容，让我们亲身体会一下Symbol API，`Symbol`是如何由几种不同的方式来组成的。

## 准备工作

为了完成本教程，我们需要：

- MXNet：在[Setup and Installation](http://mxnet.io/install/index.html)一节中可以了解MXNet的安装。
- [Jupyter Notebook](http://jupyter.org/index.html)和 [Python Requests](http://docs.python-requests.org/en/master/)。

```
pip install jupyter requests
```
- GPU：本教程中的一节将用到GPU，如果你的机器上没有GPU的话，请将变量gpu_device设置为mx.cpu()。

## 基础符号组成

### 基础计算符

下面的示例创建了一个简单的表达式：`a + b`。首先，我们使用`mx.sym.Variable`创建两个占位符，并将其命名为`a`和`b`。然后我们使用计算符`+`来构建一个新的符号。当构建变量的时候，我们不需要为其命名，MXNet会自动地为每个变量生成一个独一无二的名称。就像在下面的示例中，c被自动分配了一个唯一的名称。


```python
import mxnet as mx
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b
(a, b, c)
```

`NDArray`支持的大多数运算符在`Symbol`中也可找到，例如：


```python
# 点乘
d = a * b
# 矩阵乘法
e = mx.sym.dot(a, b)
# reshape
f = mx.sym.reshape(d+e, shape=(1,4))
# 广播
g = mx.sym.broadcast_to(f, shape=(2,4))
# 可视化绘图
mx.viz.plot_network(symbol=g)
```

通过`bind`方法可以将上面声明的计算和输入数据绑定从而进一步对其评估。我们将在[符号操作](#Symbol Manipulation)一节中更进一步的讨论这个问题。

### 基础神经网络

除了基础的计算符之外，`Symbol`还支持一系列的神经网络层。下面的示例构建了一个由两个全连接层组成的的神经网络，在给定输入数据的形状后对网络结构进行可视化。


```python
# 初始将net定义为一个变量，后续的所有操作都是对其的运算，今儿viz可以追溯整个计算过程
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(data=net, name='out')
# 需要定义data的形状才能确定全连接的参数的数目
# 对shape进行赋值时需要使用dict,其中key为变量的名称，形状为tuple
mx.viz.plot_network(net, shape={'data':(100,200)}) 
```

每一个符号都带有一个（独一无二的）字符串名称。NDArray和Symbol都代表一个单一的张量。*计算符*代表的是张量间的运算。运算符接受符号（或者NDArray）作为输入，同时也可能额外接受诸如隐藏神经元个数(*num_hidden*)或者激活类型（*act_type*）这样的超参，最后生成一个输出。

我们可以将符号简单地视为一个接受若干参数的函数。我们可以使用一下方法调用这些参数：


```python
net.list_arguments()
# 可以将所有的参数列出来
```

这些参数是每个符号所需要的输入已经参数。

- *data*：变量*data*所需要的输入
- *fc1_weight* and *fc1_bias*：全连接层*fc1*的权重和偏置
- *fc2_weight* and *fc2_bias*：全连接层*fc2*的权重和偏置
- *out_label*：计算损失所需要的标签

我们也可以明确地为其指明名称：


```python
net = mx.symbol.Variable('data')
w = mx.symbol.Variable('myweight')
net = mx.symbol.FullyConnected(data=net, weight=w, name='fc1', num_hidden=128)
net.list_arguments()
```

在上面的例子中，`FullyConnected`层需要三个输入：数据，权重和偏置。当其中任何一个输入没有被指明的时候，系统会为其自动生成一个变量。

## 更复杂的组成

MXNet为深度学习中经常使用的一些层提供了优化过的符号表示（参阅[src/operator](https://github.com/dmlc/mxnet/tree/master/src/operator)）。我们也可以在python中定义一些新的计算符。下面的示例首先将两个符号按元素相加，然后将其输入到全连接计算符中：


```python
lhs = mx.symbol.Variable('data1')
rhs = mx.symbol.Variable('data2')
net = mx.symbol.FullyConnected(data=lhs + rhs, name='fc1', num_hidden=128)
net.list_arguments()
```

和前例中的描述的单向计算符，我们还可以以一种更灵活的方式构建一个符号。


```python
data = mx.symbol.Variable('data')
net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
net1.list_arguments()
net2 = mx.symbol.Variable('data2')
net2 = mx.symbol.FullyConnected(data=net2, name='fc2', num_hidden=10)
composed = net2(data2=net1, name='composed')
composed.list_arguments()
# 将net1作为data输入进入net2的方式和之前的运算并不本质区别
```

在本例中*net2*被当成一个函数套用在一个已经存在的符号*net1*上，生成的*composed*符号将拥有*net1*和*net2*全部对象。

一旦你要开始构建一些更大的网络，你可能需要使用通用的前缀来命名符号以概述网络的结构。


```python
data = mx.sym.Variable("data")
net = data
n_layer = 2
for i in range(n_layer):
    # 和gluon中with net.name_scope()功能应该是类似的
    # 为每一层的变量赋予了特定的名称
    with mx.name.Prefix("layer%d_" % (i + 1)):
        net = mx.sym.FullyConnected(data=net, name="fc", num_hidden=100)
net.list_arguments()
```

### 深度网络的模块化构建

由于网络的层数很多，因此逐层地构建一个*深度网络*（例如谷歌的inception）可能会非常乏味。所以，对这样的网络，我们经常使用模块化的构建方式。

例如，要构建Google的Inception网络，我们首先需要定义一个factory函数将卷积，批次归一化和线性修正单元(rectified linear unit，ReLU)串联在一起。


```python
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    # 常见的结构中，每一个卷积模块都是由卷积，BN和激活组成
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                  stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.sym.Activation(data=bn, act_type='relu', name='relu_%s%s'
                  %(name, suffix))
    return act
prev = mx.sym.Variable(name="Previous Output")
conv_comp = ConvFactory(data=prev, num_filter=64, kernel=(7,7), stride=(2, 2))
shape = {"Previous Output" : (128, 3, 28, 28)}
mx.viz.plot_network(symbol=conv_comp, shape=shape)
# 
```

然后在`ConvFactory`函数的基础上，我们定义一个函数用于构建inception模块。


```python
def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3,
                      pool, proj, name):
    '''
    通过将基本模块组合成结构部件进行进一步的调用。
    其中pool指定了池化层的类型，一般选用最大池化
    '''
    # 1x1卷积
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 1x1通道缩减 + 3x3卷积
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # 1x1通道缩减+ 3x3卷积×2（但是在原生结构中此处应为5×5卷积）
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # 3×3池化 + 1×1卷积
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.sym.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

prev = mx.sym.Variable(name="Previous Output")
in3a = InceptionFactoryA(prev, 64, 64, 64, 64, 96, "avg", 32, name="in3a")
mx.viz.plot_network(symbol=in3a, shape=shape)
```

最终，我们通过链接多个inception模块构建了整个网络。完整示例请参阅[这里](https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/inception-bn.py)。

### 多符号组

如果想要构建一个拥有多个损失层的神经网络，我们需要使用`mxnet.sym.Group`将多个符号组在一起。下面的例子中两个输出被组在了一起：


```python
net = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

out1 = mx.sym.SoftmaxOutput(data=net, name='softmax') # 输出1
out2 = mx.sym.LinearRegressionOutput(data=net, name='regression') #输出2
group = mx.sym.Group([out1, out2]) # 这个和concat有什么区别？
group.list_outputs()
```

## 和NDArray的关系

就像你已经看到的那样，`Symbol`和`NDArray`均为多维数组提供了类似于`c = a + b`这样的计算符。在这里，我们简要地澄清一下两者间的不同。

`NDArray`提供了命令式编程的接口，其中的计算是逐句评估的。而`Symbol`更加接近与声明式编程，使用时我们需要提前声明计算，然后才使用数据评估。这类例子包括正则表达式和SQL语句。

`NDArray`的优点:
- 直截了当的
- 更易于使用自然语言特性（循环和判别结构等）以及第三方库（numpy等）
- 利于一步步进行代码debug

`Symbol`的优点:
- 提供了NDArray中几乎所有的函数，例如`+`，`*`，`sin`和`reshape`等
- 利于保存加载和可视化
- 利于后端进行计算和内存的优化

## 符号操作

和`NDArray`相比，`Symbol`的一个不同在于我们需要需要提前声明计算，然后才能将计算和数据绑定，运行。

在本节中，我们将介绍直接操作符号的函数。但是请注意，它们大多数都被`module`所包裹。

### 形状和类型推断

对于每个符号，我们都可以查询其参数，辅助状态和输出。我们还可以通过已知的输入数据的形状或者参数的类型来推断输出的形状和类型，这样有利于内存的分配。


```python
arg_name = c.list_arguments()  # 获取输入的名称
out_name = c.list_outputs()    # 获取输出的名称
# 通过给定输入的形状推断输出的形状
arg_shape, out_shape, _ = c.infer_shape(a=(2,3), b=(2,3))
# 通过输入数据的类型推断输出类型
arg_type, out_type, _ = c.infer_type(a='float32', b='float32')

{'input' : dict(zip(arg_name, arg_shape)),
 'output' : dict(zip(out_name, out_shape))}
{'input' : dict(zip(arg_name, arg_type)),
 'output' : dict(zip(out_name, out_type))}
```

### 数据绑定和评估

上面构建的符号`c`声明了如何进行计算。为了评估它，我们需要为参数，即自有变量，提供数据。

上述的符号`c`声明了计算的类型。通过`bind`方法，可以指定计算所在的设备(ctx=)，函数所需的参数(dict()，包含了参数key及value)。通过此方法可以得到一个可执行func，通过调用`forward`方法进行真正的计算。在func上通过.outputs可以得到相应的输出。


```python
ex = c.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3]),
                                'b' : mx.nd.ones([2,3])})
ex.forward()
print('number of outputs = %d\nthe first output = \n%s' % (
           len(ex.outputs), ex.outputs[0].asnumpy()))
```

我们也可以使用不同的数据在GPU上评估同样的符号。

**注意**为了在cpu上执行下面的小节，你需要将gpu_device设置为mx.cpu()。


```python
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
#gpu_device=mx.cpu()

ex_gpu = c.bind(ctx=gpu_device, args={'a' : mx.nd.ones([3,4], gpu_device)*2,
                                      'b' : mx.nd.ones([3,4], gpu_device)*3})
ex_gpu.forward()
ex_gpu.outputs[0].asnumpy()
```

我们也可以通过`eval`方法评估符号。这个方法将`bind`和`forward`结合在了一起。


```python
ex = c.eval(ctx = mx.cpu(), a = mx.nd.ones([2,3]), b = mx.nd.ones([2,3]))
print('number of outputs = %d\nthe first output = \n%s' % (
            len(ex), ex[0].asnumpy()))
```

在神经网络中更常用的方法是```simple_bind```，之后可以通过```forward```和```backward```获取结果和梯度信息。

### 加载和保存

逻辑上讲，符号是和ndarray相对应的。他们都能够表示一个张量，都是操作符的输入或者输出。对`Symbol`对象进行序列化时我们可以使用`picke`或者直接使用在[NDArray教程](http://mxnet.io/tutorials/basic/ndarray.html#serialize-from-to-distributed-filesystems)中讨论过的`save`和`load`方法。

对`NDArray`进行序列化时，我们直接将张量中的数据序列化并以二进制的格式保存在磁盘上。但是符号（编程）使用了图的概念。图是由链式的操作符组成的。它们由输出符号隐式地表示。所以，当对`Symbol`进行序列化时，我们序列化了一张输出符号的图。同时，Symbol使用了更可读的`json`格式来进行序列化操作。将符号转换成`json`字符串时，请使用`json`方法。


```python
# 在json文件的花括号内，使用了key-value的方式描述了计算图
# nodes表明了计算的结点，每个节点包括了（操作，名称，输入）
print(c.tojson())
c.save('symbol-c.json')
c2 = mx.sym.load('symbol-c.json')
c.tojson() == c2.tojson()
```

## 定制符号

为了更好的性能，例如`mx.sym.Convolution`和`mx.sym.Reshape`这样的运算符都是在C++中实现的。MXNet允许用户使用python这样的前段语言编写新的运算符。这样扩展和调试都会更容易一些。想要了解如何在python中实现一个运算符，请参阅[如何创建一个新的运算符](http://mxnet.io/faq/new_op.html)。

## 高级用法

### 类型转换

默认情况下，MXNet使用32位浮点数。但是为了获得更好的精度-性能，我们可以使用低精度的数据类型。例如，Nvidia Tesla Pascal GPU(e.g. P100)提升了16位浮点数的性能，同时GTX Pascal GPU(e.g. GTX 1080)在8位的整数上速度更快。

为了根据要求转换数据类型，我们可以像下面这样使用`mx.sym.cast`运算符：


```python
# 通过cast转换可以将sym默认的数据类型从float32转换为float16,从int32转为uint8
a = mx.sym.Variable('data')
b = mx.sym.cast(data=a, dtype='float16')
arg, out, _ = b.infer_type(data='float32')
print({'input':arg, 'output':out})

c = mx.sym.cast(data=a, dtype='uint8')
arg, out, _ = c.infer_type(data='int32')
print({'input':arg, 'output':out})
```

### 变量共享

为了能够在几个符号之间共享内容，我们可以像下面这样将几个符号和一个数组绑定在一起：


```python
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
b = a + a * a

data = mx.nd.ones((2,3))*2
ex = b.bind(ctx=mx.cpu(), args={'a':data, 'b':data})
ex.forward()
ex.outputs[0].asnumpy()
```


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

