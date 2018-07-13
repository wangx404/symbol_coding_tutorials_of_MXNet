
# 线性回归

在本教程中，我们将使用MXNet的API实现*线性回归*。我们将尝试学习的函数为*y = x<sub>1</sub>  +  2x<sub>2</sub>*，其中*(x<sub>1</sub>,x<sub>2</sub>)*是输入特征，而*y*是对应的标签。

## 准备工作

为了完成本教程，我们需要：

- MXNet。安装请参照[Setup and Installation](http://mxnet.io/install/index.html)；

- [Jupyter Notebook](http://jupyter.org/index.html)。

```
$ pip install jupyter
```

开始之前，下面的代码将导入必要的包。


```python
import mxnet as mx
import numpy as np

import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 准备数据

在MXNet中，数据是通过**Data Iterators**输入的。这里，我们将演示如何将数据集编码成迭代器的形式以便MXNet作为输入使用。本样例中所使用的数据由2维的数据点和对应的标签组成。


```python
#Training data
train_data = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
batch_size = 1

#Evaluation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11,26,16])
```

完成数据的准备工作后，我们将其放入迭代器中，并指定`batch_size`和`shuffle`两个参数。`batch_size`决定了模型训练时每次需要处理的样本的数目；`shuffle`则决定了输入给模型进行训练的数据是否需要打乱处理。


```python
train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
```

在上面的例子中，我们使用了`NDArrayIter`；利用它，我们可以处理numpy的ndarrays和MXNet的NDArrays。一般而言，MXNet提供了不同类型的迭代器，你可以根据自己所要处理的数据类型决定使用哪一种。迭代器的文档可以参考[这里](http://mxnet.io/api/python/io/io.html)。


## MXNet类

1. **IO:** 正如我们看到的那样，IO类处理数据，执行提供数据（以批次或者打乱形式）的操作。

2. **Symbol:** MXNet的神经网络实质上都是由符号组成的。MXNet中包含了不同类型的符号，包括代表输入数据的占位符，神经网络层和对NDArray进行处理的操作符等。

3. **Module:** module类被用来确定整体的计算。通过定义我们将要训练的模型和训练所使用的输入（包括数据和标签），以及一些额外的参数，例如学习率和优化算法和初始化方法等。

## 定义模型

MXNet使用**Symbols**来定义一个模型。Symbols是一些构建中的模块和组成模型的组件。Symbols被用来定义下述内容：

1. **Variables:** Variables是为之后的数据所准备的占位符。这个符号被用来定义有一个点，当我们开始训练时，实际的训练数据和标签将会将其填充。

2. **Neural Network Layers:** 神经网络中的一层或者其他类型的模型都可以使用Symbols进行定义。这样一个Symbol接受一个或者多个之前的Symbols作为输入，进行一定的处理，导出一个或者多个输出。神经网络中的全连接层`FullyConnected`就是这样的一个例子。

3. **Outputs:** 输出symbols是MXNet定义损失的方式。它们都以"Output"作为后缀（例如`SoftmaxOutput`）。你也可以构建你自己的[损失函数](https://github.com/dmlc/mxnet/blob/master/docs/tutorials/r/CustomLossFunction.md#how-to-use-your-own-loss-function)。一些已经定义的损失函数有：`LinearRegressionOutput`，计算输出和输入标签的l2损失；`SoftmaxOutput`，计算输出和输出的交叉熵损失。

上面描述的symbol和其他的symbols串联在一起形成拓扑图的方式是：每一个symbol的输出都是下一个symbol的输入。更多关于不同类型symbols的信息可以在[这里](http://mxnet.io/api/python/symbol/symbol.html)找到。


```python
X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")
```

上面的网络使用了下述的一些层：

1. `FullyConnected`:全连接符号表示的是神经网络中的全连接层（没有进行任何激活）；实质上，它只是输入对象的线性表示。它接受以下两个参数：
    - `data`:本层的输入（也就是指明在这里应该是谁的输出）
    - `num_hidden`:隐含神经元的个数，它和输出数据的维度相同

2. `LinearRegressionOutput`:MXNet计算训练损失的输出层，它是模型预测错误率的度量。训练的目的即为最小化这个训练损失。在我们的例子中，`LinearRegressionOutput`计算它的输入和数据标签之间的l2损失。它接受以下两个参数：
    - `data`:本层的输入（也就是指明在这里应该是谁的输出）
    - `label`:数据标签，亦即我们将用来进行l2损失计算的对象之一

**命名小贴士：** 标签变量的名称应该和我们传递给训练数据迭代器的参数`label_name`相一致。这里的默认值为`softmax_label`，但是在本例中我们将其统一变更为了`lin_reg_label`。所以在`Y = mx.symbol.Variable('lin_reg_label')`和
`train_iter = mx.io.NDArrayIter(..., label_name='lin_reg_label')`中label_name均为`lin_reg_label`。

最终，这个网络变成了*Module*；在这个模型中，我们指明了需要对哪个符号的输出进行最小化（在本例中为`lro`或者`lin_reg_output`），优化过程的学习率又是多少，以及我们需要训练的轮次数。


```python
model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)
```

对我们创建的网络进行可视化的结果如下：


```python
mx.viz.plot_network(symbol=lro)
```

## 训练模型

完成模型结构的定义之后，下一步就是将训练数据和模型结合起来对模型的参数进行训练。通过调用`Module`类中的`fit()`函数可以完成模型的训练。


```python
model.fit(train_iter, eval_iter,
            optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
            num_epoch=2,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(batch_size, 2))	    
```

## 使用一个已经训练好的模型：（测试和预测）

在完成了模型的训练之后，我们可以用它做很多事情。我们既可以用它来进行预测，也可以在测试集上评估它的性能。后者的代码如下所示：


```python
model.predict(eval_iter).asnumpy()
```

我们也可以根据评估函数来评价模型的性能。在本例中，我们评估模型在验证集上的均方根误差(MSE)。


```python
metric = mx.metric.MSE()
model.score(eval_iter, metric)
assert model.score(eval_iter, metric)[0][1] < 0.01001, "Achieved MSE (%f) is larger than expected (0.01001)" % model.score(eval_iter, metric)[0][1]
```

让我们在验证数据上添加一些噪音，看一下MSE是如何变化的。


```python
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11.1,26.1,16.1]) #Adding 0.1 to each of the values
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
model.score(eval_iter, metric)
```

当然，我们也可以自定义一个损失函数，使用它去评估模型。更多的信息可以查阅[API文档](http://mxnet.io/api/python/model.html#evaluation-metric-api-reference)。
