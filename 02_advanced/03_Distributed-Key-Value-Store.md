
# 分布式键值存储

KVStore是一个用于数据共享的地方。你可以将其视为一个可在不同设备间（GPUs和计算机）进行数据共享的对象，每一台设备都可以向其中存入或者取出数据。

## 初始化

让我们考虑一个简单的例子：初始化存储并将（`int`, `NDArray`）存入，然后再将这个值取出：


```python
import mxnet as mx

kv = mx.kv.create('local') # create a local kv store.
shape = (2,3)
kv.init(3, mx.nd.ones(shape)*2)
a = mx.nd.zeros(shape)
kv.pull(3, out = a)
print(a.asnumpy())
```

## 存入，聚合，更新
对于任意一个已经初始化的键，你都可以将相同形状的新值存入其中。


```python
kv.push(3, mx.nd.ones(shape)*8)
kv.pull(3, out = a) # pull out the value
print(a.asnumpy())
```

将要存入的数据可以存储在任意设备上。而且，你还可以将多个值存入同一个键中；KVStore会首先对这些值进行加和，然后将其存入键中：


```python
# The numbers used below assume 4 GPUs
gpus = mx.test_utils.list_gpus()
if len(gpus) > 1:
    contexts = [mx.gpu(i) for i in gpus]
else:
    contexts = [mx.cpu(i) for i in range(4)]
b = [mx.nd.ones(shape, ctx) for ctx in contexts]
kv.push(3, b)
kv.pull(3, out = a)
print(a.asnumpy())
```

每次存入时，KVStore根据`updater`将存入的值和已经存在的值进行结合。默认使用的updater是`ASSIGN`，你可以将其替换掉从而控制数据是如何合并的。


```python
def update(key, input, stored):
    print("update on key: %d" % key)
    stored += input * 2
kv._set_updater(update)
kv.pull(3, out=a)
print(a.asnumpy())
```


```python
kv.push(3, mx.nd.ones(shape))
kv.pull(3, out=a)
print(a.asnumpy())
```

## 取出

你已经看过了如何取出一个键值对。和存入数据类似，你也可以通过一次调用将值取出到多个设备上。


```python
b = [mx.nd.ones(shape, ctx) for ctx in contexts]
kv.pull(3, out = b)
print(b[1].asnumpy())
```

## 处理键值对列表

到目前为止，所有的操作都只涉及一个键。KVStore同时还提供了接口用于处理键值对列表。

对于单设备：


```python
# 列表中的值被重复存入三个键中，因而得到的结果是3，而不是1。
keys = [5, 7, 9]
kv.init(keys, [mx.nd.ones(shape)]*len(keys))
kv.push(keys, [mx.nd.ones(shape)]*len(keys))
b = [mx.nd.zeros(shape)]*len(keys)
kv.pull(keys, out = b)
print(b[1].asnumpy())
```

对于多个设备：


```python
b = [[mx.nd.ones(shape, ctx) for ctx in contexts]] * len(keys)
kv.push(keys, b)
kv.pull(keys, out = b)
print(b[1][1].asnumpy())
```

## 多机运行

基于参数服务器，`updater`在服务器节点上运行。当分布式版本准备好后，我们将更新这一节。

<!-- ## How to Choose Between APIs -->

<!-- You can mix APIs as much as you like. Here are some guidelines -->
<!-- * Use the Symbolic API and a coarse-grained operator to create  an established structure. -->
<!-- * Use a fine-grained operator to extend parts of a more flexible symbolic graph. -->
<!-- * Do some dynamic NDArray tricks, which are even more flexible, between the calls of forward and backward executors. -->

<!-- Different approaches offer you different levels of flexibility and -->
<!-- efficiency. Normally, you do not need to be flexible in all parts of the -->
<!-- network, so use the parts optimized for speed, and compose it -->
<!-- flexibly with a fine-grained operator or a dynamic NDArray. Such a -->
<!-- mixture allows you to build the deep learning architecture both efficiently and -->
<!-- flexibly as your choice.  -->


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
