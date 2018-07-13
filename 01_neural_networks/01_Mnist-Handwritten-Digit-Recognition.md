
# 手写数字识别

在本教程中，我们将手把手教你如何利用MNIST数据集构建一个手写数字分类器。即便对于刚接触深度学习的人来说，这个教程也像"Hello World"一样简单。

MNIST数据集被广泛用于手写数字分类任务。它由70000张已经标记了的28×28像素的手写数字的灰度图片组成。这个数据集通常会被切分成包含60000张图片的训练集和包含10000张图片的测试集，其中包含了十个类别（即十种数字）。目前的任务就是使用训练集中的图片训练一个分类模型，并在验证集的图片上测试其分类准确度。

![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/mnist.png)

**图1：**MNIST数据集中的样本图像

## 准备工作

为了能够完成本教程，我们需要：

- 0.10或更高版本的MXNet.安装请参照[Setup and Installation](http://mxnet.io/install/index.html)

- [Python Requests](http://docs.python-requests.org/en/master/)和[Jupyter Notebook](http://jupyter.org/index.html)

```
$ pip install requests jupyter
```

## 加载数据

在定义模型之前，我们先把[MNIST数据集](http://yann.lecun.com/exdb/mnist/)下载下来。

下面的代码会将MNIST数据集的图像和标签下载下来，并将其加载到内存中。


```python
import mxnet as mx
mnist = mx.test_utils.get_mnist()
```

在运行完上述源代码之后，整个MNIST数据集都应该被完全加载到内存当中。不过请注意，对于大型数据集，我们不太可能预先全部加载。因此我们需要这样一种机制，通过它我们可以快速有效地从数据源中加载数据。MXNet中提供的的数据迭代器正好可以解决这个问题。通过数据迭代器，我们可以将数据源源不断地送入MXNet的训练程序；而且它们非常易于初始化和使用，并且针对加载速度进行了优化。在训练过程中，我们通常会对小批次的数据进行处理；当整个训练过程完成后，每个训练样本都会被处理了很多次。在本教程中，我们将数据迭代器的批次大小设置为100。请记住，每个训练样本都是28×28的灰度图片和对应的标签。

批次图片数据一般由形状为`(batch_size, num_channels, width, height)`的4D数组表示。对于MNIST数据集而言，由于图片是灰度的，因此图片数据的颜色通道数为1。另外，由于图片的分辨率为28×28像素，因此每张图片的宽度和高度均为28。因此输入数据的形状为`(batch_size, 1, 28, 28)`。另一个需要考虑的重要因素是输入样本的顺序。这里的关键是，不要在训练过程中连续输入相同标签的样本。这样干会降低训练的速度。通过随机打乱，数据迭代器很好的处理了这个问题。不过还请注意，我们只需要对训练数据进行打乱，测试数据的顺序并不重要。

以下源代码将MNIST数据集的迭代器初始化。在这里，我们分别为训练集和测试集初始化了一个迭代器。


```python
batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
```

## 训练

我们将介绍几种不同的方法来完成手写数字识别任务。第一种方法利用了一种叫做多层感知机（Multilayer Perceptron，MLP）的传统深度神经网络架构。我们将讨论它的缺点，并以此为契机介绍第二种更为高级的，叫做卷积神经网络（Convolution Neural Network，CNN)的方法。在图像分类任务中，CNN已经被证明性能出众。

### 多层感知机

首先我们用[MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)来解决这个问题。我们将使用MXNet中的符号式接口来定义MLP的结构。首先，我们为输入数据创建一个占位符。在使用MLP的时候，我们需要将28×28的图片展开成一个784（28×28）原始像素值的1D数组结构。只要我们对于所有的图片都使用了相同的展开方法，展开的数组中像素值的顺序就不再重要。


```python
data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)
```

你可能会有疑问，当我们展平图片的时候难道没有丢失信息吗？事实确实如此。在之后讨论CNN（图片的形状没有改变）时我们会更详细地讨论这个问题。现在，让我们接着往下看。

MLP中包含了多个全连接层。在全连接层中，或者（简称为）FC层，一个神经元和前一层中的所有神经元都有连接。从线性代数的角度来看，FC层对*n x m*的输入矩阵*X*应用仿射变换，并输出一个尺寸为*n x k*的矩阵*Y*，其中*k*是FC层中的神经元的个数。*k*也被称为隐含单元的大小。输出矩阵*Y*通过*Y = X W<sup>T</sup> + b*计算得到。FC层有两部分可以学习的参数，即*k x m*大小的权重矩阵*W*和*1 x k*大小的偏置向量*b*。偏置矩阵的加法遵循[`mxnet.sym.broadcast_to()`](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html#mxnet.symbol.broadcast_to)当中的广播规则。从概念上讲，在广播求和之前，偏置向量在行方向上复制得到一个*n x k*的矩阵。

在MLP中，大多数情况下FC层的输出会被送入一个激活函数当中，激活函数对其进行按元素的非线性激活。这一步至关重要，因为它使得神经网络能够对非线性可分的输入进行分类。激活函数的一般选择是sigmoid，tanh或者[线性修正单元(rectified linear unit，ReLU)](https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29)。在本例当中，由于ReLU具有的良好特性，我们将会使用ReLU作为激活函数，而它一般也是激活函数的默认选择。

下面的代码展示了两个分别具有128个和64个神经元的全连接层。此外，这些FC层夹在激活函数层之间，每个激活函数层负责对上一个FC层的输出执行按元素的ReLU激活转换。


```python
# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
```

MLP的最后一个FC层的大小通常和数据集的类别数目相同，这一层的激活函数则为softmax函数。softmax函数将输入映射为输出类别上的概率分布。在训练阶段，损失函数计算神经网络输出的预测分布概率和标签提供的真实分布概率之间的[交叉熵损失](https://en.wikipedia.org/wiki/Cross_entropy)。

下面的代码展示了在最后一个FC层中，神经元的个数为10，也就是数字的类别数目。该层的输出被送入`SoftMaxOutput`层中，一次性执行softmax激活和交叉熵损失的计算。不过请注意，交叉熵损失只在训练阶段才会进行计算。


```python
# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
```

![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png)

**图 2:** MNIST数据集上的MLP网络结构

我们已经定义好了数据迭代器和神经网络，现在我们可以开始进行训练了。MXNet的`module`模块为模型的训练和预测提供了高级的抽象接口，在这里我们来探索一下它的特性。`module`API允许用户指定适当的参数以便控制训练过程。

下面的代码初始化了一个module以便我们训练之前定义的MLP网络。在训练中，我们将使用随机梯度下降法进行参数优化。更确切地讲，我们要使用的是小批量梯度下降法。标准的随机梯度下降法每次只在一个样本上进行训练。在实际使用中，这样训练的速度会非常的慢；因此常常使用小批量样本以加速训练过程。在本例中，我们的样本批次大小选择了较为合理的100。另外一个需要选择的参数是学习率；在优化求解的过程中，学习率决定了每一步的大小。在这里我们将学习率选择为0.1。类似于样本批次大小和学习率大小的设置我们一般称之为超参。我们所设置的超参对于训练得到的模型性能有很大的影响。为了完成本教程，我们将选择一些合理安全的参数值。在其他的教程中，我们将会讨论如何组合这些超参才能达到最佳的模型性能。

通常意义上，我们会一直对模型进行训练直至收敛，此时模型在训练数据集上学习到了一组较好的模型参数（权重+偏置）。在本教程中，我们将在训练10个轮次后停止。一个轮次就是遍历了一遍整个训练数据集。


```python
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes
```

### 预测

训练完成后，我们可以在测试数据上对已训练的模型进行评估。下面的源代码为每一张测试图片计算一个预测得分。*prob[i][j]*代表的是第*i*张测试图片得到分类结果*j*的可能性。


```python
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)
```

由于测试数据集中包含了所有测试图片的标签，因此我们可以像下面的代码那样进行准确率的计算。


```python
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]
```

如果一切顺利的当，我们应该可以看到准确率大约在0.96左右。这意味着我们能够为96%的测试图片准确分类。这已经是一个相当好的结果。但是正如我们下一部分将要看到的，我们还可以做得更好。

### 卷积神经网络

在MLP中，我们简短地讨论了MLP的一个缺点：在将图片输入MLP的首个全连接层之前，我们将图片展平，但这样损失了部分信息。事实证明这确实是一个重要的问题，因为我们没有利用像素之间在纵向和横向上的空间联系。卷积神经网络旨在使用更加结构化的权重表示来解决这个问题。在卷积神经网络中，原来那种将图片展平做矩阵乘法的简单方式被抛弃，它转而采用了一个或者多个卷积层，每个卷积层都在输入图像上进行2D卷积。

一个卷积层可以由一个或者多个滤波器组成，其中每一个都扮演着特征提取的功能。在训练过程中，CNN中的这些滤波器将学习到合适的参数表示。和MLP类似，卷积层的输出也都经过了非线性激活。除了卷积层，CNN另外一个重要的组成就是池化层了。一个池化层可以使得CNN具有平移不变性，也就是当数字在上/下/左/右方向上有几个像素的偏移时，它仍然可以和之前保持一致。池化层将*n x m*大小的区域缩减为一个值，这样一来网络对于空间位置的敏感性更低。在CNN中，池化层通常连接在卷积层（包含激活）之后。

下面的代码定义了一个叫做LeNet的卷积神经网络。LeNet是一个被广泛使用的，在数字分类任务上有良好性能的网络。我们将使用的LeNet和原生的LeNet略有不同，为了得到更好的性能，我们将激活函数从sigmoid更换为了tanh。


```python
data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
```

![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/conv_mnist.png)

**图 3:** LeNet首个卷积+池化层

现在，我们可以使用之前的超参对LeNet进行训练。请注意，如果GPU可用的话，我们推荐使用GPU进行训练。和之前的MLP相比，由于LeNet更加复杂，运算更多，因此使用GPU可以大大加速训练过程。我们只需要将计算设备从`mx.cpu()`改为`mx.gpu()`即可，MXNet会自动完成剩余的设置。就像之前一样，我们在训练10个轮次之后停止训练。


```python
# create a trainable module on CPU, change to mx.gpu() if GPU is available
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)
```

### 预测

最后，我们可以使用已经训练好的LeNet在测试数据上进行预测。


```python
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = lenet_model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.98, "Achieved accuracy (%f) is lower than expected (0.98)" % acc.get()[1]
```

如果一切都没有问题，我们应该可以看到使用LeNet进行预测可以得到更高的准确率。在所有的测试图片上，CNN的预测准确率大概在98%左右。

## 总结

在本教程中，我们学习了如何使用MXNet解决一个标准的计算机视觉问题：对手写数字图片进行分类。你应该已经明白如何使用MXNet构建诸如MLP或者CNN的模型，并在对其进行训练和评估。
