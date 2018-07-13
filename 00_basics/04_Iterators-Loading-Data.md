
# 迭代器-加载数据
在本教程中，我们将集中精力在如何向一个训练或推断程序中输入数据。在MXNet中，大部分的训练和推断程序都能够接受数据迭代器；这样一来数据加载过程会变得更加简单，当加载大型数据集时尤其如此。在这里我们讨论一下API的使用惯例和几个迭代器的样例。

## 准备条件

为了完成本教程，我们需要：
- MXNet.安装和使用请参考[Setup and Installation](http://mxnet.io/install/index.html)一节。
- [OpenCV Python library](http://opencv.org/opencv-3-2.html),  [Python Requests](http://docs.python-requests.org/en/master/), [Matplotlib](https://matplotlib.org/) and [Jupyter Notebook](http://jupyter.org/index.html).
```
$ pip install opencv-python requests matplotlib jupyter
```
- 将环境变量`MXNET_HOME`设置到MXNet的根目录中。
```
$ git clone https://github.com/dmlc/mxnet ~/mxnet
$ export MXNET_HOME='~/mxnet'
```

## MXNet数据迭代器

*MXNet*中的数据迭代器和Python中的迭代器对象非常类似。在Python中，`iter`函数使我们可以在诸如Python list这样的可迭代对象上通过调用`next()`依次读取数据。迭代器为遍历各种类型的可迭代对象提供了一个抽象的接口，而无需公开底层数据源的详细信息。

在MXNet中，每次对数据迭代器调用`next`后会返回一个批次类似于`DataBatch`的数据。一个`DataBatch`中包含了n个训练的样本和对应的标签。这里的n指的就是迭代器的`batch_size`。在迭代器的末尾，没有更多的数据可以提供，此时迭代器会抛出一个``StopIteration``，就像Python中的`iter`一样。`DataBatch`的结构定义在[这里](http://mxnet.io/api/python/io/io.html#mxnet.io.DataBatch).

训练样本的详细信息，例如名称，形状，数据类型，布局以及对应的标签，都可以通过`DataBatch`中的`provide_data`和`provide_label`属性作为`DataDesc`的数据描述对象存在。`DataDesc`的详细结构信息在[
这里](http://mxnet.io/api/python/io/io.html#mxnet.io.DataDesc)定义。

MXNet中的所有IO都是通过`mx.io.DataIter`和它的子类来处理的。在这个教程中，我们会讨论MXNet中的常见的几种迭代器。

在深入了解之前，让我们先配置一下环境，导入需要使用的库。


```python
import mxnet as mx
%matplotlib inline
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tarfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## 读取内存中的数据

当数据以`NDArray`或者`numpy.ndarray`的形式存在与内存当中时，我们可以通过[__`NDArrayIter`__](http://mxnet.io/api/python/io/io.html#mxnet.io.NDArrayIter)来读取数据，代码如下所示：



```python
import numpy as np
data = np.random.rand(100,3)
label = np.random.randint(0, 10, (100,))
data_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=30)
for batch in data_iter:
    print([batch.data, batch.label, batch.pad])
```

## 从csv文件中读取数据

MXNet中使用[`CSVIter`](http://mxnet.io/api/python/io/io.html#mxnet.io.CSVIter)函数来读取csv中的数据，代码如下所示：


```python
# lets save `data` into a csv file first and try reading it back
np.savetxt('data.csv', data, delimiter=',')
data_iter = mx.io.CSVIter(data_csv='data.csv', data_shape=(3,), batch_size=30)
for batch in data_iter:
    print([batch.data, batch.pad])
```

## 自定义迭代器

当内置的迭代器不能满足你的需求时，你可以创建一个你自己的自定义迭代器。
在MXNet中构造一个迭代器需要满足以下要求：
- 在python2中构造一个`next()`方法或者在python3中构造`__next()__`，其返回的对象是`DataBatch`，在数据流结束后则返回`StopIteration`。
- 构造一个`reset()`方法当数据读取完全后从头开始。
- 包含了`provide_data`对象，是由一系列的`DataDesc`（[这里可以看到更多信息](http://mxnet.io/api/python/io/io.html#mxnet.io.DataBatch)）构成，在这些对象中包含了data的名称，形状，类型和导出信息。
- 包含了`provide_label`对象，是由一系列的`DataDesc`构成，在这些对象中包含了label的名称，形状，类型和导出信息。

当构建一个迭代器的时候，你可以从零开始，也可以重新利用现有的迭代器。例如，在图像字幕应用中，输入的样本是一张图片，而标签是一个句子。所以，我们可以这样构建一个迭代器：
- 通过`ImageRecordIter`构建一个`image_iter`，这样就可以多线程地进行数据预读取和图片增广。
- 通过`NDArrayIter`构建一个`caption_iter`，或者利用rnn中的bucketing迭代器。
- `next()`返回`image_iter.next()`和`caption_iter.next()`的结合。

下面的例子展示了如何构建一个简单的迭代器。


```python
class SimpleIter(mx.io.DataIter):
    # 所有的类都是io.DataIter的子类
    # (name, shape)-->_provide_data-->@property-->provide_data-->+gen-->data-->batch
    def __init__(self, data_names, data_shapes, data_gen,
                         label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = list(zip(data_names, data_shapes))
        self._provide_label = list(zip(label_names, label_shapes))
        self.num_batches = num_batches # 终止判断
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def __next__(self):
        return self.next()
    
    def next(self):
        # 迭代过程条件判断
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            # 将data和label打包成Batch读取的形式
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration # 终止
```


```python
n = 32
num_classes = 10
# 使用了一个lambda函数做个generator
data_iter = SimpleIter(['data'], [(n, 100)],
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['softmax_label'], [(n,)],
                  [lambda s: np.random.randint(0, num_classes, s)])
```


```python
batch = data_iter.next()
batch.label
```

我们使用上面定义的`SimpleIter`来训练一个简单的MLP：


```python
import mxnet as mx
num_classes = 10
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
print(net.list_arguments())
print(net.list_outputs())
```

这里，有四个变量是可学习的参数：全连接层*fc1*和*fc2*中的*weights*和*biases*；有两个变量是输入数据：训练样本的*data*和包含了数据标签以及*softmax_output*的*softmax_label*。

在MXNet的Symbol API中，*data*变量被称之为自由变量。想要真正执行符号，它们需要和数据绑定在一起。[点击这里了解更多关于Symbol的内容](http://mxnet.io/tutorials/basic/symbol.html)。

在MXNet的Module API中，我们通过数据迭代器向神经网络输入训练样本。[点击这里了解更多关于Module的内容](http://mxnet.io/tutorials/basic/module.html)。


```python
import logging
logging.basicConfig(level=logging.INFO)

n = 32
# 使用了一个lambda函数做个generator
data_iter = SimpleIter(['data'], [(n, 100)],
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['softmax_label'], [(n,)],
                  [lambda s: np.random.randint(0, num_classes, s)])

mod = mx.mod.Module(symbol=net)
mod.fit(data_iter, num_epoch=5)
```

Python使用注释：mxnet中的很多方法在Python2.x中使用的是字符串，而在Python3.x中使用的是二进制数据。为了使得本教程可以正常运行，我们在这里定义一个函数，它可以在Python3.x环境中将字符串转换成二进制数据格式。


```python
def str_or_bytes(str):
    """
    A utility function for this tutorial that helps us convert string 
    to bytes if we are using python3.
    ----------
    str : string

    Returns
    -------
    string (python2) or bytes (python3)
    """
    if sys.version_info[0] < 3:
        return str
    else:
        return bytes(str, 'utf-8')
```

## Record IO

Record IO是MXNet中用于数据IO的文件格式。它紧凑地将数据打包以便于从分布式系统（如Hadoop HDFS和AWS S3）中进行高效的写入和读取操作。在[这里](http://mxnet.io/architecture/note_data_loading.html)你可以了解更多关于`RecordIO`的内容。

MXNet为数据的顺序读取和随机读取分别提供了[__`MXRecordIO`__](http://mxnet.io/api/python/io/io.html#mxnet.recordio.MXRecordIO)和[__`MXIndexedRecordIO`__](http://mxnet.io/api/python/io/io.html#mxnet.recordio.MXIndexedRecordIO)两种方式。

### MXRecordIO

首先，让我们了解一下如何使用`MXRecordIO`进行顺序读取。这些文件的扩展名为`.rec`。


```python
record = mx.recordio.MXRecordIO('tmp.rec', 'w')
for i in range(5):
    record.write(str_or_bytes('record_%d'%i))

record.close() # 关闭句柄
```

在打开文件时设置选项参数为`r`，我们可以如下所示的读取数据：


```python
record = mx.recordio.MXRecordIO('tmp.rec', 'r')
while True:
    item = record.read()
    if not item:
        break
    print (item)
record.close()
```

### MXIndexedRecordIO

`MXIndexedRecordIO`支持随机或者顺序地读取数据。我们可以通过如下所示的方式创建索引记录文件和对应的索引文件：


```python
# 此时需要同时创建两个文件，一个是idx，另外一个是rec
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
for i in range(5):
    record.write_idx(i, str_or_bytes('record_%d'%i))

record.close()
```

现在我们可以通过索引键值来单独地访问对应的记录。


```python
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
record.read_idx(3) # 新方法，read)idx()和keys()
```

你也可以将文件的所有索引键值列出来。


```python
record.keys
```

### 打包和解包数据

.rec中的每一条记录都可以包含任意二进制文件。但是，大多数的深度学习任务都要求数据以标签/数据的格式输入。`mx.recordio`为这种操作提供了一些实用的函数，例如，`pack`，`unpack`，`pack_img`和`unpack_img`。

#### 打包和解包二进制文件

[__`pack`__](http://mxnet.io/api/python/io/io.html#mxnet.recordio.pack) 和[__`unpack`__](http://mxnet.io/api/python/io/io.html#mxnet.recordio.unpack)被用来存储浮点标签（或者1d的浮点数组）和二进制数据。这些数据和头文件一起进行打包。头文件的结构请参考[此处](http://mxnet.io/api/python/io/io.html#mxnet.recordio.IRHeader)。


```python
# pack
data = 'data'
label1 = 1.0
# 头文件创建中，flag是任意数字，label是标签，id应为唯一的索引，id2一般为0
# 为每一条记录中data，label和相应头文件就绪后即可打包
header1 = mx.recordio.IRHeader(flag=0, label=label1, id=1, id2=0)
s1 = mx.recordio.pack(header1, str_or_bytes(data))

label2 = [1.0, 2.0, 3.0]
header2 = mx.recordio.IRHeader(flag=3, label=label2, id=2, id2=0)
s2 = mx.recordio.pack(header2, str_or_bytes(data))
```


```python
# unpack
print(mx.recordio.unpack(s1))
print(mx.recordio.unpack(s2))
```

#### 打包和解包图片数据

MXNet提供了[__`pack_img`__](http://mxnet.io/api/python/io/io.html#mxnet.recordio.pack_img)和[__`unpack_img`__](http://mxnet.io/api/python/io/io.html#mxnet.recordio.unpack_img)函数进行图片数据的打包和解包。通过`pack_img`打包的图片数据可以通过`mx.io.ImageRecordIter`进行加载。


```python
data = np.ones((3,3,1), dtype=np.uint8)
label = 1.0
# 应为pack()只能对一维的数组打包，因此对于图片需要调用pack_img()
# 同时打包时需要指定quality和图片格式（方便进行压缩）
header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
s = mx.recordio.pack_img(header, data, quality=100, img_fmt='.jpg')
```


```python
# unpack_img
print(mx.recordio.unpack_img(s))
```

#### im2rec的使用

你可以使用MXNet[src/tools](https://github.com/dmlc/mxnet/tree/master/tools)文件夹中的``im2rec.py``脚本将原始的图片打包成*RecordIO*的格式。具体的使用方式参照下节的`Image IO`。

## 图片IO

在这一节，我们将学习在MXNet中如何预处理和加载图片数据。

在MXNet中提供了四种图片加载的方式：
- 使用[__mx.image.imdecode__](http://mxnet.io/api/python/io/io.html#mxnet.image.imdecode)加载原始图像
- 使用[__`mx.img.ImageIter`__](http://mxnet.io/api/python/io/io.html#mxnet.image.ImageIter)从rec文件和原生图片中加载出iter。
- [__`mx.io.ImageRecordIter`__](http://mxnet.io/api/python/io/io.html#mxnet.io.ImageRecordIter)功能基于C++，功能的灵活性不如ImageIter，但是提供了多种语言的支持。
- 构建自定义的`mx.io.DataIter`

### 图片前处理

图片可以使用不同的方式进行前处理，其中一些罗列如下：
- `mx.io.ImageRecordIter`很快，但是不够灵活。而且它只能在图片分类任务中使用；在更复杂的目标识别和图像分割中不再适用。
- `mx.recordio.unpack_img`（或者`cv2.imread`和`skimage`等等）结合numpy可以更灵活的读取图片，但是由于Python全局解释锁（Global Interpreter Lock，GIL）的存在速度比较慢。
- `mx.image`将图片储存为[__`NDArray`__](http://mxnet.io/tutorials/basic/ndarray.html)的格式，并且由于它利用了MXNet引擎的自动并行化处理且规避了GIL，因此速度会更快。

下面，我们演示一些由`mx.image`提供的一些经常使用的预处理教程。

开始之前，我们先把将要使用的样本图片下载下来。


```python
fname = mx.test_utils.download(url='http://data.mxnet.io/data/test_images.tar.gz', dirname='data', overwrite=False)
tar = tarfile.open(fname)
tar.extractall(path='./data')
tar.close()
```

#### 加载原始图片

`mx.image.imdecode`让我们可以加载图片，其中的`imdecode`的使用方式和``OpenCV``非常类似。

**注意：**你可能会仍然需要使用``OpenCV``（不是CV2 Python库）来代替`mx.image.imdecode`。


```python
img = mx.image.imdecode(open('data/test_images/ILSVRC2012_val_00000001.JPEG', 'rb').read())
plt.imshow(img.asnumpy()); plt.show()
```

#### 图片转换


```python
# resize to w x h
tmp = mx.image.imresize(img, 100, 70)
plt.imshow(tmp.asnumpy()); plt.show()
```


```python
# crop a random w x h region from image
tmp, coord = mx.image.random_crop(img, (150, 200))
print(coord)
plt.imshow(tmp.asnumpy()); plt.show()
```

### 使用图片迭代器加载数据

在学习如何使用两个内置的图片迭代器进行数据的读取之前，让我们先下载一下__Caltech 101__数据集（此数据集中包含了101个类别的物体），并将其打包成rec io格式。


```python
fname = mx.test_utils.download(url='http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz', dirname='data', overwrite=False)
tar = tarfile.open(fname)
tar.extractall(path='./data')
tar.close()
```

让我们先观察以下数据。正如你所看到的那样，在根文件夹(./data/101_ObjectCategories)下每一个标签都包含了一个子文件夹(./data/101_ObjectCategories/yin_yang)。

通过使用`im2rec.py`的脚本我们可以将其转成rec io格式。首先，我们需要创建一个list，其中包含了所有的图片文件和对应的标签。

**译注：以下代码可能无法运行，可以command中输入指令完成list的生成。**


```python
os.system('python %s/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --test-ratio=0.2 data/caltech data/101_ObjectCategories' % os.environ['MXNET_HOME'])
```

生成的list(./data/caltech_train.lst)文件的格式为：编号\t标签(一个或者多个)\t图片路径。在这个例子中，每张图片只有一个标签，但是你可以向list中添加更多的标签用于多标签训练。


```python
os.system("python %s/tools/im2rec.py --num-thread=4 --pass-through data/caltech data/101_ObjectCategories" % os.environ['MXNET_HOME'])
```

现在rec文件保存在了data文件夹中(./data)。

#### 使用ImageRecordIter

[__`ImageRecordIter`__](http://mxnet.io/api/python/io/io.html#mxnet.io.ImageRecordIter)可用于读取保存在rec io格式文件中的数据。想要使用`ImageRecordIter`，我们需要创建一个加载rec文件的实例。


```python
# ImageRecordIter能够接受的参数更多
data_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/caltech.rec", # rec文件
    data_shape=(3, 227, 227), # shape
    batch_size=4, # batch size
    resize=256 # 这是一个比较奇怪的参数
    # 图片增强的参数部分
    )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
```

#### 使用ImageIter

[__ImageIter__](http://mxnet.io/api/python/io/io.html#mxnet.io.ImageIter)是一个较为灵活的接口，它同时支持RecordIO文件和原生格式文件。


```python
# ImageIter除了可以从rec和idx文件中进行数据读取，还可以根据img_list，
# 即储存了图片和位置和label中读取数据
data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 227, 227),
                              path_imgrec="./data/caltech.rec",
                              path_imgidx="./data/caltech.idx" )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
```


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

