
# 矩阵分解

在推荐系统当中，存在着一组用户（user）和一组项目（item）。其中每个用户都对系统当中的某些项目进行了评分，那么我们希望能够预测用户将会给那些他们尚未评分的项目打多少分，这样一来我们就能够对用户进行项目推荐。

矩阵分解是推荐系统主要使用的算法之一。它主要用于发现隐藏在有相互作用的不同实体间的隐特征。

假设我们为每个用户分配一个k维的向量，为每一个项目也分配一个k维的向量，而它们两者的点积结果就是用户对于项目的评分。我们可以直接学习到用户和项目的向量，这基本上就是对用户-项目矩阵进行SVD分解。我们同样可以使用多层的神经网络学习这种隐特征。

在本教程中，我们将使用MXNet一步步实现这些想法。

## 准备数据

我们在这里只使用[MovieLens](http://grouplens.org/datasets/movielens/)数据集，但是这个方法对其他的数据集也是可行的。数据集中的每一行都包含了一个用户id，电影id，评分和时间戳的元组，但我们只会使用前三项的数据。我们首先要定义一个包含n个元组的批次。它为MXNet提供了数据和标签的名称、形状信息。

接下来我们定义一个数据迭代器，它每次返回一个批次的元组。

**译注：源代码不能正常工作，对其修正后现可用。**


```python
import mxnet as mx
import random

class DataIter(mx.io.DataIter):
    # mxnet中的DataIter需要提供data/label的name和shape，以及数据生成源
    def __init__(self, fname, batch_size):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data = []
        for line in open(fname, 'r').readlines():
            tks = line.strip().split('\t')
            if len(tks) != 4:
                continue
            self.data.append((int(tks[0]), int(tks[1]), float(tks[2])))
        self._provide_data = [('user', (batch_size, )), ('item', (batch_size, ))] # [(name, data_shape),]
        self._provide_label = [('score', (self.batch_size, ))]
    
    def __iter__(self):
        return self
    
    def reset(self):
        random.shuffle(self.data)
        
    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
    
    # 在__next__()中调用next()以兼容python2&3
    def __next__(self):
        return self.next()
    
    def next(self):
        for k in range(len(self.data) // self.batch_size):
            users = []
            items = []
            scores = []
            for i in range(self.batch_size):
                j = k * self.batch_size + i
                user, item, score = self.data[j]
                users.append(user)
                items.append(item)
                scores.append(score)

            data_all = [mx.nd.array(users), mx.nd.array(items)]
            label_all = [mx.nd.array(scores)]
            data_batch = mx.io.DataBatch(data_all, label_all)
            return data_batch
        
        raise StopIteration

```

现在我们可以下载数据并提供一个能够获取数据迭代器的函数：


```python
import os
import urllib
import zipfile
if not os.path.exists('ml-100k.zip'):
    urllib.urlretrieve('http://files.grouplens.org/datasets/movielens/ml-100k.zip', 'ml-100k.zip')
with zipfile.ZipFile("ml-100k.zip","r") as f:
    f.extractall("./")
def get_data(batch_size):
    return (DataIter('./ml-100k/u1.base', batch_size), DataIter('./ml-100k/u1.test', batch_size))
```

最后我们计算一下用户和项目的数目用于之后的使用。


```python
def max_id(fname):
    mu = 0
    mi = 0
    for line in open(fname, 'r').readlines():
        tks = line.strip().split('\t')
        if len(tks) != 4:
            continue
        mu = max(mu, int(tks[0]))
        mi = max(mi, int(tks[1]))
    return mu + 1, mi + 1
max_user, max_item = max_id('./ml-100k/u.data')
(max_user, max_item)
```

## 优化

我们首先实现一个经常在矩阵分解中使用的均方根误差（root-mean-square error，RMSE）测量函数。


```python
import math
def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    pred = pred.flatten()
    for i in range(len(label)):
        ret += (label[i] - pred[i]) * (label[i] - pred[i])
        n += 1.0
    return math.sqrt(ret / n)
```

然后我们定义一个通用的训练模块，它的设计借鉴了图像分类应用。


```python
ctx = mx.cpu() # 可更改为GPU进行加速训练
def train(network, batch_size, num_epoch, learning_rate):
    model = mx.model.FeedForward(
        ctx = ctx,
        symbol = network,
        num_epoch = num_epoch,
        learning_rate = learning_rate,
        wd = 0.0001,
        momentum = 0.9)

    batch_size = 64
    train, test = get_data(batch_size)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG)

    model.fit(X = train,
              eval_data = test,
              eval_metric = RMSE,
              batch_end_callback=mx.callback.Speedometer(batch_size, 20000/batch_size),)
```

## 网络
现在让我们尝试一下不同的网络。首先我们将直接学习隐向量。


```python
def plain_net(k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user feature lookup
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    # item feature lookup
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    # predict by the inner product, which is elementwise product and then sum
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

train(plain_net(64), batch_size=64, num_epoch=10, learning_rate=.05)
```

然后我们尝试使用两层的神经网络学习隐变量


```python
def get_one_layer_mlp(hidden, k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user latent features
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    user = mx.symbol.Activation(data = user, act_type="relu")
    user = mx.symbol.FullyConnected(data = user, num_hidden = hidden)
    # item latent features
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    item = mx.symbol.Activation(data = item, act_type="relu")
    item = mx.symbol.FullyConnected(data = item, num_hidden = hidden)
    # predict by the inner product
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

train(get_one_layer_mlp(64, 64), batch_size=64, num_epoch=10, learning_rate=.05)
```

添加Dropout层用于缓解过拟合。


```python
def get_one_layer_dropout_mlp(hidden, k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user latent features
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    user = mx.symbol.Activation(data = user, act_type="relu")
    user = mx.symbol.FullyConnected(data = user, num_hidden = hidden)
    user = mx.symbol.Dropout(data=user, p=0.5)
    # item latent features
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    item = mx.symbol.Activation(data = item, act_type="relu")
    item = mx.symbol.FullyConnected(data = item, num_hidden = hidden)
    item = mx.symbol.Dropout(data=item, p=0.5)
    # predict by the inner product
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred
train(get_one_layer_mlp(256, 512), batch_size=64, num_epoch=10, learning_rate=.05)
```


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

