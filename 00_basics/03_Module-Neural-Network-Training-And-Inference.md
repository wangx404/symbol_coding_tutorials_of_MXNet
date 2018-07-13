
# Module-神经网络训练和预测

一般而言，训练一个神经网络包含了非常多的步骤。我们需要指定训练数据的输入，模型参数的初始化，如何进行前向传播和反向传播，如何根据计算得到的梯度更新参数和如何设置模型检查点等等。而在进行预测时，我们需要重复上述的大多数步骤。不管是对于新手还是具有经验的开发者而言，这个任务都是艰巨的。

幸运的是，MXNet中的`module`模块将训练和预测常用的代码进行了模块化处理。`Module`为执行预先定义的模型同时提供了高层次和中层次的接口。我们可以互换地使用这两种接口。在本教程中，我们将展示如何使用这两种接口。

## 准备工作

为了完成本教程，我们需要：

- MXNet.在[Setup and Installation](http://mxnet.io/install/index.html)一节中可以了解MXNet的安装。

- [Jupyter Notebook](http://jupyter.org/index.html)和 [Python Requests](http://docs.python-requests.org/en/master/)。

```
pip install jupyter requests
```

## 初步实施

在本教程中，我们将通过在[UCI字母识别数据集](https://archive.ics.uci.edu/ml/datasets/letter+recognition)上训练一个[多层感知机，（MLP）](https://en.wikipedia.org/wiki/Multilayer_perceptron)来展示如何使用`module`模块。

下面的代码将下载UCI字母识别数据数据集，并将其以80:20的比例切分成训练集和测试机。同时，它还创建了一个批次大小为32的训练数据迭代器和一个测试数据迭代器。


```python
import logging
logging.getLogger().setLevel(logging.INFO) # 更改logger的level
import mxnet as mx
import numpy as np

fname = mx.test_utils.download('http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data')
data = np.genfromtxt(fname, delimiter=',')[:,1:] # len=15的数组
label = np.array([ord(l.split(',')[0])-ord('A') for l in open(fname, 'r')])

batch_size = 32
ntrain = int(data.shape[0]*0.8)
# mxnet.io中为不同类型的数据提供了构造dataiter的func().
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)
```

下面我们定义一下网络并将其可视化。


```python
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mx.viz.plot_network(net) # 对network进行可视化绘图.
```

## 构建一个模型

现在，我们准备开始介绍module。通常而言，我们使用`Module`来创建模型。其中，需要指定的参数如下所示：

- `symbol`: 网络的定义，亦即计算图
- `context`: 运算设备
- `data_names` : 输入数据名称
- `label_names` : 输出变量名称

对于`net`来说，我们有一个数据名为`data`，一个标签名为`softmax_label`；其中，`softmax_label`是根据我们为`SoftmaxOutput`操作符指定的名称自动命名的。


```python
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label']) 

```

## 中间层次接口

我们已经创建了一个模型。下面让我们看一下如何使用`Module`下的中间层级接口完成模型的训练和预测工作。使用这些接口，开发人员可以零活地运行`forward`和`backward`来一步步进行计算。它对于debug同样非常有用。

为了训练一个模型，我们需要执行以下步骤：
- `bind` : 通过bind为参数分配内存空间，进行训练准备工作
- `init_params` : 参数分配和初始化
- `init_optimizer` : 创建优化器，默认为`sgd`.
- `metric.create` : 构建评估函数
- `forward` : 前向计算
- `update_metric` : 根据输出信息评估并更新评估指标
- `backward` : 反向计算
- `update` : 根据优化器参数和上一批次数据中计算得到的梯度进行参数更新

它们的使用如下所示：


```python
# 将train_iter的信息提供给module，可根据shape分配内存的大小
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# 参数初始化
mod.init_params(initializer=mx.init.Uniform(scale=.1))
# lr=0.1的sgd
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# accuracy作为评估指标
metric = mx.metric.create('acc')

for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # 前向计算
        mod.update_metric(metric, batch.label)  # 更新评估指标
        mod.backward()                          # 反向计算
        mod.update()                            # 更新参数
    print('Epoch %d, Training %s' % (epoch, metric.get()))
```

想要了解更多API信息，请访问[Module API](http://mxnet.io/api/python/module/module.html)。

## 高层次接口

### 训练

为了用户的便捷使用，`Module`同样提供了高层次的接口用于训练，预测和评估。通过简单地调用`fit`API，用户不再需要手动执行上节提到的所有步骤，这些步骤都在内部进行自动执行。

调用`fit`函数的代码如下所示：


```python
# 重设迭代器
train_iter.reset()

# 创建模型
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=8)
```

默认状态下，`fit`函数将`eval_metric`设置为`accuracy`，将`optimizer`设置为`sgd`，将优化器参数设置为`(('learning_rate', 0.01),)`。

### 预测和评估

为了使用模型进行预测，我们可以调用`predict()`。它能够自动收集和返回所有的预测结果。


```python
# 使用训练后的mod进行预测时调用mod.predict()即可,函数返回所有的预测值
y = mod.predict(val_iter)
assert y.shape == (4000, 26)
```

如果我们只需要对测试集进行评估而不需要预测的输出结果，那么可以调用`score()`函数来完成。它在验证数据集上进行预测，并根据评估的方式自动评估模型的性能。

使用代码如下所示：


```python
# 在训练结束后还可以使用score()对验证集进行最后的评估
score = mod.score(val_iter, ['acc'])
print("Accuracy score is %f" % (score[0][1]))
assert score[0][1] > 0.77, "Achieved accuracy (%f) is less than expected (0.77)" % score[0][1]
```

其他一些可以使用的评估函数有：`top_k_acc`，`F1`，`RMSE`， `MSE`，`MAE`和`ce`。想要了解更多关于这些评估函数的信息，请访问[Evaluation metric](http://mxnet.io/api/python/metric/metric.html)。

通过调整训练轮数，学习率，优化器参数可以调整参数以获得最佳的评分。

### 保存和加载

在每一轮次的训练后，我们可以调用`checkpoint`将模型的参数保存起来。


```python
model_prefix = 'mx_mlp' 
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint) # 但在很多情况这种记录是没有必要的
```

调用`load_checkpoint`函数可以加载已经保存的模型。将操作符和相应的参数加载以后，我们可以将这些参数加载进入模型当中。

从检查点重新加载参数的步骤：
- 加载检查点数据，得到相应的return
- 构建新的mod，并导入数据得到model


```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3) #(prefix,epoc)
assert sym.tojson() == net.tojson()

mod.set_params(arg_params, aux_params)
```

假如我们想要从检查点中恢复训练，我们可以直接调用`fit()`函数（而不需要调用`set_params()`）将加载的参数传入；这样以来，`fit()`就知道从这些参数开始训练而不是从头开始从随机初始化参数开始。我们还可以设置一下`begin_epoch`参数，这样`fit()`就知道我们是从之前的轮次中恢复训练的。


```python
mod = mx.mod.Module(symbol=sym)
mod.fit(train_iter,
        num_epoch=21,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=3)

assert score[0][1] > 0.77, "Achieved accuracy (%f) is less than expected (0.77)" % score[0][1]        
```
