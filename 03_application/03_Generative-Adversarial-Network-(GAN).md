
# 对抗生成网络（GAN）

对抗生成网络（Generative Adversarial Networks，GANs）是无监督学习中的一类算法——训练一个GAN时，你不需要对数据进行标注。

对抗生成网络框架由两个神经网络组成：生成网络和判别网络。

生成器的工作就是获取一系列的随机数并据此生成数据（例如图片或者文本）。

然后，判别器接受该数据以及来自于数据集中的其他样本，并尝试区分该数据是“假的”（由生成网络创建）还是“真的”（来自于原始的数据集）。

训练过程中，两个网络之间相互进行对抗。生成器尝试创建一些“仿真”的数据，这样它就能够欺骗判别器，让判别器误以为它生成的数据来自于原始数据集。与此同时，判别器尝试着不被愚弄——它会学习更好地判别数据是真实的还是虚假的。

由于两个网络在训练的过程中相互战斗，它们可以被视为是相互对抗的，这就是“对抗生成网络”一词的来源。

## 深度卷积对抗生成网络

本教程将介绍深度卷积对抗生成网络（Deep Convolutional Generative Adversarial Networks，DCGAN），它将卷积神经网络（Convolutional Neural Networks，CNNs）和对抗生成网络结合在了一起。

我们将创建一个能够从随机数中生成手写体数字图片的DCGAN。本教程将使用[此文](https://arxiv.org/abs/1511.06434)中概述的神经网络结构和指南，并利用MNIST数据集作为原始数据集。

## 如何使用本教程
你可以遵照本教程的顺序，依次执行出现的python代码来完成本教程。

- 第一个网络就是“生成器”，它从随机数中生成手写体数字图片。
- 第二个网络就是“判别器”，它将判断由生成器创建的图片是真（手写数字的逼真图像）还是假（图像看起来不像是来自于原始数据集）。

除了创建一个DCGAN之外，你还可以学到：

- 如何操作和迭代需要输送给神经网络的批次图像数据。
- 如何创建一个自定义的MXNet数据迭代器，以便从正态分布中生成随机数。
- 如何使用来自于MXNet Module API中的低层次函数，例如.bind()，.forward()和.backward()等，创建自定义的训练流程。DCGAN中的训练过程要比许多其他的神经网络复杂得多，因此我们需要使用这些函数来替代高层的.fit()函数。
- 如何在训练过程中可视化图像

## 准备工作

本教程假设你非常熟悉CNN相关的内容，并在MXNet中实现过相关的网络。同时，你还应该对于逻辑回归的内容比较熟悉。考虑到我们将创建一个自定义的数据迭代器，将随机数作为生成网络的输入进行迭代，因此你需要对MXNet的数据迭代器有些基本的了解。

本示例被设计为在一块GPU上进行训练。由于在CPU上进行训练将会非常的慢，因此推荐你使用GPU进行训练。

为了完成本教程，你需要：
- MXNet
- Python和下列的python库:
    - Numpy-用于矩阵计算
    - OpenCV-用于图像处理
    - Matplotlib-用于输出的可视化

## 数据
我们需要两部分数据对DCGAN进行训练：
1. 来自于MNIST数据集的手写图片数据
2. 来自于正态分布的随机数

生成网络将使用随机数作为输入生成手写数字图像；判别网络将使用来自于MNSIT数据集中的手写数字图像来决定生成器生成的图像是否是逼真的。

MNIST数据集中包含了70000张手写数字图像，每一张图像都是28×28像素。为了生成随机数，我们需要创建一个自定义的MXNet数据迭代器，当我们需要的时候，它能够从正态分布中返回随机数。

## 准备数据

### 准备MNIST数据
让我们首先从准备MNIST数据集中的手写数字开始吧。


```python
import mxnet as mx
import numpy as np

mnist_train = mx.gluon.data.vision.datasets.MNIST(train=True)
mnist_test = mx.gluon.data.vision.datasets.MNIST(train=False)
```


```python
# The downloaded data is of type `Dataset` which are
# Well suited to work with the new Gluon interface but less
# With the older symbol API, used in this tutorial. 
# Therefore we convert them to numpy array first
X = np.zeros((70000, 28, 28))
for i, (data, label) in enumerate(mnist_train):
    X[i] = data.asnumpy()[:,:,0]
for i, (data, label) in enumerate(mnist_test):
    X[len(mnist_train)+i] = data.asnumpy()[:,:,0]
```

使用numpy为数据集的行创建一个随机排序，这样我们就能随机化手写数字。数据集中的每个图像都排列成28×28的网络，网络的每个格子代表一个图像的像素。


```python
#Use a seed so that we get the same random permutation each time
np.random.seed(1)
p = np.random.permutation(X.shape[0])
X = X[p]
```

由于我们正在创建的DCGAN需要64×64的图像作为输入，因此我们使用Opencv将每张图像从28×28缩放至64×64。


```python
import cv2
X = np.asarray([cv2.resize(x, (64,64)) for x in X])
```

64×64图像中的每一个像素都由0-255之间的一个数字表示，像素值的大小表示了像素的强度。但是由于[研究论文](https://arxiv.org/abs/1511.06434)的建议，我们需要将-1到1之间的数字作为DCGAN的输入。为了缩放像素值，我们将其除以(255/2)，这使得像素值被缩放到0-2之间。然后我们将其减去1，就得到了在-1到1之间的数值表示。


```python
X = X.astype(np.float32, copy=False)/(255.0/2) - 1.0
```

图片最终需要通过一个70000x3x64x64的数组输送到神经网络之中，但是他们现在储存在一个70000x64x64的数组中。因此我们需要为这些图像添加3个通道。通常而言，当我们处理图片时，这3个通道表示的是每张图片的红绿蓝（RGB）三种颜色分量。与偶遇MNIST数据集是灰度图像，因此我们只需要一个通道来表示这个数据集。我们将会把通道0上的数据填充到其他通道上：


```python
X = X.reshape((70000, 1, 64, 64))
X = np.tile(X, (1, 3, 1, 1))
```

最后，我们将图片放进MXNet的NDArrayIter中，这样就可以在训练的过程中方便地进行迭代。我们还将会将图片切分成包含64张图片的批次数据。每次进行迭代时，我们都会得到一个大小为(64, 3, 64, 64)的四维数组，表示一个批次的64张图片。


```python
import mxnet as mx
batch_size = 64
image_iter = mx.io.NDArrayIter(X, batch_size=batch_size)
```

### 2.准备随机数
我们需要从正态分布中产生的随机数作为生成网络的输入，因此我们需要创建一个MXNet DataIter用于生成训练所需的批次数据。DataIter是MXNet基本的数据加载API。下面，我们将创建一个DataIter的子类RandIter。在迭代的过程中我们使用MXNet内置的mx.random.normal函数来返回随机的正态分布数据。

**译注：下述迭代器似乎不能正常运行，请参考MXNet教程对其进行修改。**


```python
class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        #Returns random numbers from a gaussian (normal) distribution
        #with mean=0 and standard deviation = 1
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]
```

当我们初始化RandIter时，我们需要提供两个数字：批次的大小以及我们需要从多少个随机数中生成一张图像。这个数字就是Z，我们将其设为100。这个数字来源于我们上面所提到的那篇研究论文。每次我们进行迭代，我们得到一个四维的形状为(batch_size, Z, 1, 1)的数组，在本例中样本的形状为(64, 100, 1, 1)。


```python
Z = 100
rand_iter = RandIter(batch_size, Z)
```


```python
rand_iter.rand
```

## 创建模型

模型包括两个我们同时进行训练的网络——生成网络和判别网络。

### 生成器
然我们首先定义一个生成网络，它使用反卷积层（又叫作fractionally strided layers）从随机数中生成图像：


```python
no_bias = True
fix_gamma = True
epsilon = 1e-5 + 1e-12

rand = mx.sym.Variable('rand')

g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=1024, no_bias=no_bias)
gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=epsilon)
gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias)
gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=epsilon)
gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=epsilon)
gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=epsilon)
gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=3, no_bias=no_bias)
generatorSymbol = mx.sym.Activation(g5, name='gact5', act_type='tanh')
```

生成的图片从我们之前创建的RandIter产生的随机数开始，所以我们为其创建了rand变量。随后我们从反卷积层（有时候也叫作fractionally strided layers）开始创建模型。在反卷积层后我们添加了批次归一化层和ReLU激活层。

我们重复此过程4次，由于其中的反卷积层使用了(2,2)的步长和(1,1)的填充，因此每层的图像尺寸都会翻倍。通过创建这些层，生成网络将会学习到在每层应该如何对输入的随机数向量Z进行上采样，以便网络最后输出一张图片。同时在每一层，我们还将滤波器的数量减半，从而将其维度降低。最后的输出层的形状是64x64x3大小，表示了图片的尺寸和通道数。根据在DCGAN上的论文研究建议，我们在最后一层使用tanh激活而不是常规的relu。最后一层神经元的输出表示的是生成图像的像素值。

请注意，在创建模型的过程中，我们使用了三个变量，它们分别是：no_bias，fixed_gamma和epsilon。网络中的神经元没有偏置，从实践的角度看这样对DCGAN似乎更好。在批次归一化层中，我们设置fixed_gamma=True，这表示对于所有的批次归一化层我们将gama固定为1。epsilon是一个被添加到批次归一化层中的很小的数字，这样我们就不会遇到除零的问题。但是默认情况下，CuDNN要求这个值要比1e-5大，所以我们在这个值上添加了一个更小的数字以确保它仍然很小。

### 判别器
接下来让我们创建一个判别网络，它从MNIST数据集和生成网络中获取手写书体图像作为输入：


```python
data = mx.sym.Variable('data')

d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=epsilon)
dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias)
dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=epsilon)
dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=1024, no_bias=no_bias)
dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=epsilon)
dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
d5 = mx.sym.Flatten(d5)

label = mx.sym.Variable('label')
discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')
```

我们首先创建数据变量，它将储存判别器的输入图像。

之后，判别器将通过五个卷积层，每层中的卷积核均为4x4大小，2x2步长和1×1填充。每经过一层卷积，图像的大小减半一次（初始大小为64×64）。同时模型通过加倍滤波器的数量来增加每层的维度数，起始时滤波器为128，结束时为1024，最后展平输出。

在最后的卷积层后，我们将神经元的输出展平最终得到一个数字作为判别网络的输出。这个数字就是由判别器决定的，图片为真的概率大小。我们使用逻辑回归来决定这个概率值的大小。当我们传递MNIST数据集的“真实”图片时，我们的label为1；当我们传递由生成网络生成的“虚假”图片时，我们的label为0。最终我们可以在判别网络上执行逻辑回归。

### 通过Module API准备模型

到目前为止，我们已经同时为生成器和判别器定义了MXNet符号模型。在我们训练模型之前，我们需要使用Module API将这些符号绑定起来，以便为模型创建一个计算图。它将同时允许我们决定如何对模型进行初始化，以及使用什么类型的优化器。让我们同时为两个模型创建Module吧。


```python
sigma = 0.02
lr = 0.0002
beta1 = 0.5
# Define the compute context, use GPU if available
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

#=============Generator Module=============
generator = mx.mod.Module(symbol=generatorSymbol, data_names=('rand',), label_names=None, context=ctx)
generator.bind(data_shapes=rand_iter.provide_data)
generator.init_params(initializer=mx.init.Normal(sigma))
generator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods = [generator]

# =============Discriminator Module=============
discriminator = mx.mod.Module(symbol=discriminatorSymbol, data_names=('data',), label_names=('label',), context=ctx)
discriminator.bind(data_shapes=image_iter.provide_data,
          label_shapes=[('label', (batch_size,))],
          inputs_need_grad=True)
discriminator.init_params(initializer=mx.init.Normal(sigma))
discriminator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods.append(discriminator)
```

首先，我们为网络创建Module，然后将其与我们在前述步骤中创建的符号绑定在一起。我们使用rand_iter.provide_data为数据形状和生成网络进行绑定。这意味着当我们在生成器Module上进行批次数据的迭代时，RandIter将会通过它的provide_data函数为我们的模型提供随机数。

类似地，我们将判别器Module和image_iter.provide_data绑定在一起，它通过我们之前设置的NDArrayIter为我们提供MNIST中的图像。

请注意，我们使用了高斯初始化，其中sigma等于0.2。这意味着网络中神经元的初始化权重将是来自高斯（正态）分布的随机数，其均值为0，标准差为0.02。

我们同时使用Adam优化器用于随机梯度下降。这里，我们根据DCGAN论文中使用的值设置了两个超参，lr和beta1。我们使用单gpu进行训练。如果你的机器上没有gpu的话，请将设备设置为cpu()。

### 可视化训练过程
在我们训练模型之前，我们首先设置几个辅助函数帮助将生成器生成的图像可视化，同时将其与真实图像对比。


```python
from matplotlib import pyplot as plt

#Takes the images in the batch and arranges them in an array so that they can be
#Plotted using matplotlib
def fill_buf(buf, num_images, img, shape):
    width = buf.shape[0]/shape[1]
    height = buf.shape[1]/shape[0]
    img_width = int(num_images%width)*shape[0]
    img_hight = int(num_images/height)*shape[1]
    buf[img_hight: img_hight+shape[1], img_width: img_width+shape[0], :] = img

#Plots two images side by side using matplotlib
def visualize(fake, real):
    #64x3x64x64 to 64x64x64x3
    fake = fake.transpose((0, 2, 3, 1))
    #Pixel values from 0-255
    fake = np.clip((fake+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    #Repeat for real image
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)

    #Create buffer array that will hold all the images in the batch
    #Fill the buffer so to arrange all images in the batch onto the buffer array
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n*fake.shape[1]), int(n*fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fill_buf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n*real.shape[1]), int(n*real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fill_buf(rbuff, i, img, real.shape[1:3])

    #Create a matplotlib figure with two subplots: one for the real and the other for the fake
    #fill each plot with the buffer array, which creates the image
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(rbuff)
    plt.show()
```

## 模型适配
训练一个DCGAN是一个需要很多步骤的复杂过程。为了能够适配模型，对于MNIST数据集每一批次的数据：

1. 使用包含随机数的向量Z前向通过生成网络。这将输出得到一个“假”的图像，因为图像来自于生成器。

2. 使用假图像作为输入在判别网络中进行一个前向传播和反向传播。我们将标签设置为0以表明这是假的图像。这一步骤将训练判别器学习到假图像长什么样子。我们将反向传播过程中得到的梯度保存起来在下一步使用。

3. 使用来自MNIST数据集中的假真输入在判别网络中进行一个前向传播和反向传播。我们将标签设置为1以表明这是真的图像，这样判决器将会学习到如何识别一个真的图像。

4. 将来自于假图像的梯度和来自于真图像的梯度加合起来对判别器进行参数更新。

5. 这样一来在这一批次的数据上，判别器得到了更新，但是我们仍然需要对生成器进行更新。首先，在参数更新后的判别器上进行前向传播和反向传播，这样我们就得到了新的梯度值。使用这个梯度继续反向传播以更新生成器。

下面是DCGAN的主要训练过程：


```python
# =============train===============
print('Training...')
for epoch in range(1):
    image_iter.reset()
    for i, batch in enumerate(image_iter):
        #Get a batch of random numbers to generate an image from the generator
        rbatch = rand_iter.next()
        #Forward pass on training batch
        generator.forward(rbatch, is_train=True)
        #Output of training batch is the 64x64x3 image
        outG = generator.get_outputs()

        #Pass the generated (fake) image through the discriminator, and save the gradient
        #Label (for logistic regression) is an array of 0's since this image is fake
        label = mx.nd.zeros((batch_size,), ctx=ctx)
        #Forward pass on the output of the discriminator network
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        #Do the backward pass and save the gradient
        discriminator.backward()
        gradD = [[grad.copyto(grad.context) for grad in grads] for grads in discriminator._exec_group.grad_arrays]

        #Pass a batch of real images from MNIST through the discriminator
        #Set the label to be an array of 1's because these are the real images
        label[:] = 1
        batch.label = [label]
        #Forward pass on a batch of MNIST images
        discriminator.forward(batch, is_train=True)
        #Do the backward pass and add the saved gradient from the fake images to the gradient
        #generated by this backwards pass on the real images
        discriminator.backward()
        for gradsr, gradsf in zip(discriminator._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        #Update gradient on the discriminator
        discriminator.update()

        #Now that we've updated the discriminator, let's update the generator
        #First do a forward pass and backwards pass on the newly updated discriminator
        #With the current batch
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        discriminator.backward()
        #Get the input gradient from the backwards pass on the discriminator,
        #and use it to do the backwards pass on the generator
        diffD = discriminator.get_input_grads()
        generator.backward(diffD)
        #Update the gradients on the generator
        generator.update()

        #Increment to the next batch, printing every 50 batches
        i += 1
        if i % 50 == 0:
            print('epoch:', epoch, 'iter:', i)
            print
            print("   From generator:        From MNIST:")

            visualize(outG[0].asnumpy(), batch.data[0].asnumpy())
```

这将对GAN进行训练，并对训练过程中网络的变化进行可视化。每迭代25次之后，我们将调用之前创建的可视化函数，它将把中间结果绘制出来。

左侧的图像显示的是在最近一次迭代中生成器创建的假图像，右侧的图像显示的是同一次迭代中被输入进判别器中的MNIST数据集的原始图像。

随着训练的进行，你将会看到这样的事情发生，生成器将产生越来越逼真的图像。在每一次迭代中，左侧的图像越来越越接近于原始数据图像。

## 总结
现在我们使用MNIST数据集在Apache MXNet上训练了一个深度卷积对抗生成网络（Deep Convolutional Generative Adversarial Neural Networks， DCGAN）。

我们得到了两个神经网络：生成器，能够从随机数中创建手写数字图像；判别器，能够分辨一个手写图像是否为真。

在此过程中，我们学习了与训练深度神经网络相关的图像操作和可视化。我们还学习了如何根据模型使用MXNet的Module API执行高级模型训练功能。

## 致谢

This tutorial is based on [MXNet DCGAN codebase](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/dcgan.py),
[The original paper on GANs](https://arxiv.org/abs/1406.2661), as well as [this paper on deep convolutional GANs](https://arxiv.org/abs/1511.06434).

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
