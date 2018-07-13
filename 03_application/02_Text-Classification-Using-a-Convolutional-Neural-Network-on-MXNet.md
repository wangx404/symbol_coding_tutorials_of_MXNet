
# 在MXNet上使用卷积神经网络进行文本分类

本教程基于Yoon Kim的[文章](https://arxiv.org/abs/1408.5882)使用卷积神经网络进行文本的情感分类。本教程在python2.7和python3.6环境下的MXNet（1.0版本）上进行了测试。

在本教程中，我们将使用烂番茄上的已经情绪标注的影评语句训练一个深度卷积神经网络。最终我们将得到一个模型可以根据语句的情绪对其进行分类（得分1表示纯粹的正面情绪，得分0表示纯粹的负面情绪，而得分0.5表示中立的情绪）。

第一步，我们要获取已经标注的包含正面情绪和负面情绪句子的训练数据，然后将其处理成向量集合，最后随机将其划分成训练集和测试机。


```python
from __future__ import print_function

from collections import Counter
import itertools
import numpy as np
import re

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
    
def download_sentences(url):
    """
    从特定的URL下载语句，剥离换行符，并转码为Unicode。  
    """
    remote_file = urlopen(url)
    return [line.decode('Latin1').strip() for line in remote_file.readlines()]

def clean_str(string):
    """
    符号/字符串清洗
    原始脚本来源于https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py    
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()

def load_data_and_labels():
    """
    从文件中加载极性数据，并将其分割成单词并生成对应的标签。
    返回分割后的语句和对应的标签
    """
    # 下载
    positive_examples = download_sentences('https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos')
    negative_examples = download_sentences('https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg')
    
    # 符号化
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent).split(" ") for sent in x_text]

    # 生成标签
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    
    return x_text, y


def pad_sentences(sentences, padding_word="</s>"):
    """
    将所有语句填充到最长语句的长度，返回填充后的语句。
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
        
    return padded_sentences


def build_vocab(sentences):
    """
    根据语句建立一个从到的词汇表极其映射，返回词汇表和词汇表的逆向映射
    """
    # 建立词汇表 word:count_num
    word_counts = Counter(itertools.chain(*sentences)) # [[word, word],[...]]->[word, word]
    
    # 从索引到单词的映射，根据词频排序的word list
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    
    # 从单词到索引的映射
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    
    return vocabulary, vocabulary_inv


def build_input_data(sentences, labels, vocabulary):
    """
    根据词汇表将句子和标签映射为向量。
    """
    x = np.array([
            [vocabulary[word] for word in sentence]
            for sentence in sentences])
    y = np.array(labels)
    
    return x, y

"""
加载并预处理MR数据集的数据，返回经过输入向量，标签，词汇表和反向词汇表
"""
# 加载和预处理数据
sentences, labels = load_data_and_labels()
sentences_padded = pad_sentences(sentences)
vocabulary, vocabulary_inv = build_vocab(sentences_padded)
x, y = build_input_data(sentences_padded, labels, vocabulary)

vocab_size = len(vocabulary)

# 随机打乱数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# 训练集/验证集切割，本数据集中共有10662个样本
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

sentence_size = x_train.shape[1]

print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
print('train shape:', x_train.shape) # 训练集形状
print('dev shape:', x_dev.shape) # 测试集形状
print('vocab_size', vocab_size)　# 词汇数目
print('sentence max words', sentence_size) #　最长语句长度
```

现在我们通过加载，向量化和打乱得到了训练数据和测试数据，我们可以进一步定义我们想要使用数据进行训练的网络结构。

首先，我们需要为网络的输入和输出设置相应的占位符；然后我们将第一层设置为一个嵌入层，这样一来单词向量可以被映射到一个低维的向量空间中，在此空间中向量之间的距离大小（根据它们的情绪）对应着它们的相关性大小。


```python
import mxnet as mx
import sys, os

'''
为网络的输入和输出设置批次大小和占位符。
'''

batch_size = 50
print('batch size', batch_size)

input_x = mx.sym.Variable('data') # placeholder for input data
input_y = mx.sym.Variable('softmax_label') # placeholder for output label


'''
设置网络的第一层
'''
# 创建一个嵌入层以便像word2vec一样学习单词在低维空间中的表示
num_embed = 300 # dimensions to embed words into 嵌入的维度大小
print('embedding dimensions', num_embed)

embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')

# reshape嵌入后的数据以便进行卷积操作
conv_input = mx.sym.Reshape(data=embed_layer, shape=(batch_size, 1, sentence_size, num_embed)) #(batch, 1, 56, 300)
```

网络的下一层将使用不同大小的卷积核对句子中的有序嵌入词向量执行卷积操作。卷积核每次在3/4/5个单词上滑动操作，这就等同于每次查看句子当中的3/4/5个单词。这将使我们能够理解单词和他们周围的单词是如何影响语句的情感的。

在每个卷积层后面我们都添加一个最大池化层用来提取卷积后最重要的元素，并将它们转化为特征向量。

因为每一个卷积+池化滤波器产生的张量都具有不同的形状，因此我们需要为它们分别创建卷积池化层，然后将这些层的结果拼接成一个更大的特征向量。


```python
# 为每一个滤波操作创建一个卷积池化层
filter_list=[3, 4, 5] # the size of filters to use
print('convolution filters', filter_list)

num_filter=100
pooled_outputs = []
for filter_size in filter_list:
    convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
    relui = mx.sym.Activation(data=convi, act_type='relu')
    pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1, 1)) # 不同大小的max pooling
    pooled_outputs.append(pooli)

# 结合所有的输出
total_filters = num_filter * len(filter_list)
concat = mx.sym.Concat(*pooled_outputs, dim=1) # 拼接

# reshape for next layer
h_pool = mx.sym.Reshape(data=concat, shape=(batch_size, total_filters)) # similar to flatten
```

接下来，我们添加一层Dropout正则化，它会随机地屏蔽层中的一部分神经元（此处设置为50%的比例）来确保我们的模型不会过拟合。这可以防止神经元之间相互适应，以确保它们能够独立学习到有用的特征。

这对于我们的模型来说是非常重要的。这是因为数据集的总数据量约20k，能够用于训练的只有10k左右；所以由于数据集实在太小，而我们的模型又过于强大，我们非常容易得到一个过拟合的模型。


```python
# dropout layer
dropout = 0.5
print('dropout probability', dropout)

if dropout > 0.0:
    h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
else:
    h_drop = h_pool
```

最后，我们为模型添加非线性的全连接层；之后则使用softmax函数对输出结果进行分类，得到介于0（负面情绪）和1（正面情绪）之间的结果。


```python
# fully connected layer
# 全连接层
num_label = 2

cls_weight = mx.sym.Variable('cls_weight')
cls_bias = mx.sym.Variable('cls_bias')

fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

# softmax output
# softmax输出
sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

# set CNN pointer to the "back" of the network
cnn = sm
```

现在我们已经定义好了CNN模型结构，接下来我们将设置用于执行训练的设备以及用于训练和测试的数据集。

*运行下述的代码时，如果你的机器上有GPU，那么你可以将ctx设置为mx.gpu(0)；否则你需要将ctx设置为mx.cpu(0)，但是这样训练就会变得很慢。*


```python
from collections import namedtuple
import math
import time

# 设置CNN模型的结构（作为命名元组）
CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

#　设置用于训练/测试的设备，如果GPU可用请使用GPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

arg_names = cnn.list_arguments()

input_shapes = {}
input_shapes['data'] = (batch_size, sentence_size)

arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
args_grad = {}
for shape, name in zip(arg_shape, arg_names):
    if name in ['softmax_label', 'data']: # input, output
        continue
    args_grad[name] = mx.nd.zeros(shape, ctx)

cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

param_blocks = []
arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
initializer = mx.initializer.Uniform(0.1)
for i, name in enumerate(arg_names):
    if name in ['softmax_label', 'data']: # input, output
        continue
    initializer(mx.init.InitDesc(name), arg_dict[name])

    param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

data = cnn_exec.arg_dict['data']
label = cnn_exec.arg_dict['softmax_label']

cnn_model= CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)
```

我们现在可以对网络执行训练和测试操作，其中mxnet将会自动的执行前向传播，梯度计算以及反向传播。


```python
'''
使用反向传播训练cnn模型
'''

optimizer = 'rmsprop'
max_grad_norm = 5.0
learning_rate = 0.0005
epoch = 50

print('optimizer', optimizer)
print('maximum gradient', max_grad_norm)
print('learning rate (step size)', learning_rate)
print('epochs to train for', epoch)

# 创建优化器
opt = mx.optimizer.create(optimizer)
opt.lr = learning_rate
updater = mx.optimizer.get_updater(opt)

# For each training epoch
for iteration in range(epoch):
    tic = time.time()
    num_correct = 0
    num_total = 0

    # Over each batch of training data
    for begin in range(0, x_train.shape[0], batch_size):
        batchX = x_train[begin: begin+batch_size]
        batchY = y_train[begin: begin+batch_size]
        if batchX.shape[0] != batch_size:
            continue

        cnn_model.data[:] = batchX
        cnn_model.label[:] = batchY

        # 前向
        cnn_model.cnn_exec.forward(is_train=True)

        # 反向
        cnn_model.cnn_exec.backward()

        # 训练数据评估
        num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)

        # 参数更新
        norm = 0
        for idx, weight, grad, name in cnn_model.param_blocks:
            grad /= batch_size
            l2_norm = mx.nd.norm(grad).asscalar()
            norm += l2_norm * l2_norm

        norm = math.sqrt(norm)
        for idx, weight, grad, name in cnn_model.param_blocks:
            if norm > max_grad_norm:
                grad *= (max_grad_norm / norm)

            updater(idx, grad, weight)

            # 梯度设为0
            grad[:] = 0.0

    # 学习率衰减以防发散
    if iteration % 50 == 0 and iteration > 0:
        opt.lr *= 0.5
        print('reset learning rate to %g' % opt.lr)

    # 本轮训练结束
    toc = time.time()
    train_time = toc - tic
    train_acc = num_correct * 100 / float(num_total)

    # 保存检查点
    if (iteration + 1) % 10 == 0:
        prefix = 'cnn'
        cnn_model.symbol.save('./%s-symbol.json' % prefix)
        save_dict = {('arg:%s' % k) : v  for k, v in cnn_model.cnn_exec.arg_dict.items()}
        save_dict.update({('aux:%s' % k) : v for k, v in cnn_model.cnn_exec.aux_dict.items()})
        param_name = './%s-%04d.params' % (prefix, iteration)
        mx.nd.save(param_name, save_dict)
        print('Saved checkpoint to %s' % param_name)


    # 评估测试集
    num_correct = 0
    num_total = 0

    # For each test batch
    for begin in range(0, x_dev.shape[0], batch_size):
        batchX = x_dev[begin:begin+batch_size]
        batchY = y_dev[begin:begin+batch_size]

        if batchX.shape[0] != batch_size:
            continue

        cnn_model.data[:] = batchX
        cnn_model.cnn_exec.forward(is_train=False)

        num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)

    dev_acc = num_correct * 100 / float(num_total)
    print('Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
            --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc))
```

完成模型的训练后，我们将学习到的参数存储在本地文件中的.params文件中。我们现在可以随时加载这个文件，并通过预训练模型的前向传播来预测语句的情绪。

## 引用
- ["Implementing a CNN for Text Classification in TensorFlow" blog post](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
