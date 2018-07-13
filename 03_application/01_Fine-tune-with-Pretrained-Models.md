
# 使用预训练模型进行微调

计算机视觉领域内许多令人惊叹的深度学习算法都需要大量的数据用于模型训练。例如，最流行的基准数据集[ImageNet](http://www.image-net.org/)中包含了一千个类别的一百万张图片。但是对于任何实际问题，我们所拥有的都是小得多的数据集。在这些情况下，如果我们从随机初始化的参数开始训练神经网络的权重，那么我们将会非常严重地过拟合训练集上。

解决这个问题的一种方法是，首先使用类似于ImageNet这样的大型数据集训练一个深度网络。然后，当我们使用新的数据集完成一个新的任务时，我们可以在这些预训练的权重上开始训练。这个过程一般被称之为_微调_。微调有很多的变种。有时候，这些初始化的神经网络仅仅被用作_特征提取器_。这意味着我们将冻结输出层之前的每一层网络的参数，然后只学习一个新的输出层。在[另一份文档](https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/predict.ipynb)中我们介绍了如何实现这种特征提取。另一种方法就是更新网络的全部权重，我们将在本文档中展示这种方法。

要微调一个网络，我们首先需要将网络最后的全连接层替换为一个新的全连接层，新全连接层的输出数目是我们所需要的类别数目。之后我们随机初始化全连接层的权重，然后正常对其进行训练。不过通常情况下，我们会使用一个较小的学习率，这是因为根据直觉我们此时可能已经很接近较好的结果。

在本展示中，我们将在ImageNet上预训练一个模型，然后在小得多的caltech-256数据集上进行微调。按照本示例，你也可以在其他数据集上进行微调，甚至是在人脸识别等截然不同的应用中。

我们将会证明，即便使用最简单的超参设置，我们也能够在caltech-256数据集上达到甚至超过现有的结果。

```eval_rst
.. list-table::
   :header-rows: 1

   * - Network 
     - Accuracy 
   * - Resnet-50 
     - 77.4% 
   * - Resnet-152 
     - 86.4% 
```

## 准备数据

我们遵循标准协议从每个类别中抽取60张图片作为训练集，其余的作为验证集。我们将图片缩放到256x256大小，然后打包成rec文件。准备数据的脚本如下：

> 为了能够在Windows上成功运行下面的脚本，你需要使用https://cygwin.com/install.html 。

```sh
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar

mkdir -p caltech_256_train_60
for i in 256_ObjectCategories/*; do
    c=`basename $i`
    mkdir -p caltech_256_train_60/$c
    for j in `ls $i/*.jpg | shuf | head -n 60`; do
        mv $j caltech_256_train_60/$c/
    done
done

python ~/mxnet/tools/im2rec.py --list --recursive caltech-256-60-train caltech_256_train_60/
python ~/mxnet/tools/im2rec.py --list --recursive caltech-256-60-val 256_ObjectCategories/
python ~/mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 caltech-256-60-val 256_ObjectCategories/
python ~/mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 caltech-256-60-train caltech_256_train_60/
```

下面的代码会下载预先生成的rec文件，它可能需要几分钟才能完成。


```python
import os, sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urlretrieve(url, filename)
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
```

接下来，我们定义一个函数用于数据数据迭代器：


```python
import mxnet as mx

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec = './caltech-256-60-train.rec',
        data_name = 'data',
        label_name = 'softmax_label',
        batch_size = batch_size,
        data_shape = data_shape,
        shuffle = True,
        rand_crop = True,
        rand_mirror = True)
    val = mx.io.ImageRecordIter(
        path_imgrec = './caltech-256-60-val.rec',
        data_name   = 'data',
        label_name  = 'softmax_label',
        batch_size  = batch_size,
        data_shape  = data_shape,
        rand_crop   = False,
        rand_mirror = False)
    return (train, val)
```

然后我们下载一个50层的预训练的ResNet模型，然后将其加载进内存当中。请注意，如果`load_checkpoint`报错，请删除下载文件并重试。


```python
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)
```

## 训练

我们首先定义一个函数，它将替换掉给定网络的最后一层全连接。


```python
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol 预训练网络符号
    arg_params: the argument parameters of the pretrained model 预训练模型的参数值
    num_classes: the number of classes for the fine-tune datasets 用于微调的数据集类别数目
    layer_name: the layer name before the last fully-connected layer 最后一层全连接之前的层名称
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)
```

现在我们创建了一个模型。请注意，我们通过参数`arg_params`传递已加载模型中的参数。最后一个全连接层中的参数将通过`initializer`进行初始化。


```python
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
        num_epoch=8,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)
```

然后我们开始训练。我们使用了AWS EC2 g2.8xlarge，它包含有8块GPU。


```python
num_classes = 256
batch_per_gpu = 16
num_gpus = 8

(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

batch_size = batch_per_gpu * num_gpus
(train, val) = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
assert mod_score > 0.77, "Low training accuracy."
```

你将会看到在经过了仅仅8轮之后，我们就可以得到78%的验证集准确率。这已经达到了在caltech-256上单独进行训练的结果，例如可参考[VGG](http://www.robots.ox.ac.uk/~vgg/research/deep_eval/)。

下面，我们尝试使用另外一个预训练模型。这个模型是在比ImageNet 1K大10倍的完整ImageNet数据集上进行训练的，同时我们将使用3倍深的Resnet结构。


```python
get_model('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
assert mod_score > 0.86, "Low training accuracy."
```

就像你看到的，在仅仅经过一轮训练之后，它就在验证集上达到了83%的准确率。在8轮训练之后，验证集准确率增加到了86.4%。

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
