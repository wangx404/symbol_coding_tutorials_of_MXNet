
# CSRNDArray-压缩系数矩阵格式存储的NDArray

现实世界当中的很多数据集都是高纬度的稀疏特征向量。以推荐系统为例，其中类别和用户的数量是数以百万计的。但是用户的购买数据显示，每个用户只购买过其中极少数类别的一些商品，这就导致了数据集具有很高的稀疏性。（即，大部分的元素都是零。）

使用传统密集矩阵的方式来存储和处理这样的大型稀疏矩阵会造成严重的内存浪费和零操作。为了能够利用矩阵的稀疏结构，MXNet中的`CSRNDArray`将其存储成压缩系数矩阵的格式（[compressed sparse row，CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29)），并为其运算符开发了专用的算法。这种格式专门为这类2D结构的大型稀疏矩阵设计，此类矩阵包含了大量的列，而且每一行数据都是稀疏的（即，只有少数的非零值）。

## 压缩系数矩阵NDArray(CSRNDArray)的优点

对于高度稀疏的矩阵来说（例如，只有大约1%的非零元素，或者说密度大约为1%），和使用常规的`NDArray`相比，`CSRNDArray`主要有以下两个优点。
- 内存占用被显著地降低
- 特定的操作显著加快（例如矩阵-向量乘法）

你可能对[SciPy](https://www.scipy.org/)中的CSR存储格式比较熟悉，那么你应该注意到了它和MXNet的CSR实现有着诸多的相似性。然而，`CSRNDArray`从`NDArray`继承了一些额外的有竞争力的特性；例如，`CSRNDArray`所具有的非阻塞异步评估和自动并行化都是SciPy的CSR所不具有的。在[NDArray tutorial](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html#lazy-evaluation-and-automatic-parallelization)中你能够找到更多关于MXNet评估和自动并行化策略的解释。

`CSRNDArray`的引入为`NDArray`增加了`stype`这个新的属性，其代表了存储类型的信息。除了像**ndarray.shape**，**ndarray.dtype**和**ndarray.context**这些经常可调用的属性外，现在你还可以调用**ndarray.stype**。对于常规的密集NDArray，其`stype`的值为**"default"**；而对于`CSRNDArray`，`stype`的值则是**"csr"**。

## 准备工作
为了完成本教程，你需要：
- 安装MXNet。[Setup and Installation](https://mxnet.io/install/index.html)
- [Jupyter](http://jupyter.org/)在这里你可以看到如何在你的操作系统中进行MXNet的安装。
    ```
    pip install jupyter
    ```
- 对MXNet中的NDArray有基本的了解。[NDArray - Imperative tensor operations on CPU/GPU](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html)在这里你可以看到相关的基础教程。
- SciPy。本教程中的一节会用到SciPy的Python实现。如果你没有安装SciPy，那么可以忽略此节中的相关内容。
- GPUs。本教程中一节需要用到GPU。如果你的机器上没有GPU，那么请将变量`gpu_device`设置为`mx.cpu()`。

## 压缩稀疏矩阵
CSRNDArray使用三个分开的1D数组**data**，**indptr**和**indices**表示一个2D矩阵，其中：第i行的列序号以升序的顺序存储在`indices[indptr[i]:indptr[i+1]]`当中，而对应的数值则存储在`data[indptr[i]:indptr[i+1]]`当中。

- **data**: CSR格式数据数组
- **indices**: CSR格式索引数组
- **indptr**: CSR格式索引指针数组

### 矩阵压缩样例

例如，给出这样一个矩阵：
```
[[7, 0, 8, 0]
 [0, 0, 0, 0]
 [0, 9, 0, 0]]
```

我们可以使用CSR对其进行压缩，要做到这一点，我们首先需要计算一下`data`，`indices`和`indptr`。

`data`数组当中以行优先的顺序存储着矩阵的所有非零元素。换句话说，你可以创建一个data数组，其中所有的0都已经从矩阵中删除，然后一行一行按顺序将剩余数字存储起来。这样你就得到了。

    data = [7, 8, 9]

`indices`数组当中存储着`data`数组中所有元素的列序号。当你从头开始索引数组的时候，你可以看到7的列序号是0，然后8的列序号是2，而后面的9的列序号则是1。

    indices = [0, 2, 1]

`indptr`数组帮助确认数据出现在哪一行。它存储着矩阵当中每一行的第一个非零元素在`data`数组中的偏移量（即位置）。这个数字通常从0开始（原因会稍后解释），所以indptr[0]=0。数组当中的每个后续值都是直到该行时，data当中存储的非零元素的合计值。上述矩阵的第一行中有两个非零元素，因此indptr[1]=2。下一行包含了0个非零元素，因此合计值仍然是2，所以indptr[2]=2。最后一行中有一个非零元素，所以合计值为3，因此indptr[3]=3。为了重建密集矩阵，你可以为第一行使用`data[0:2]`和`indices[0:2]`，为第二行使用`data[2:2]`和`indices[2:2]`（包含了0个非零元素），而第三行使用`data[2:3]`和`indices[2:3]`。这样你得到的indptr为：

    indptr = [0, 2, 2, 3]

请注意，在MXNet中，指定行的列索引总是以升序排列，同一行的列索引也不允许重复。

## 创建数组

有几种不同的方法创建`CSRNDArray`，但是首先让我们使用刚刚计算得到的`data`， `indices`和`indptr`来创建矩阵。

你可以通过调用`csr_matrix`函数将`data`，`indices`和`indptr`创建为一个矩阵。


```python
import mxnet as mx
# Create a CSRNDArray with python lists
shape = (3, 4)
data_list = [7, 8, 9]
indices_list = [0, 2, 1]
indptr_list = [0, 2, 2, 3]
a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
# Inspect the matrix
a.asnumpy()
```


```python
import numpy as np
# Create a CSRNDArray with numpy arrays
data_np = np.array([7, 8, 9])
indptr_np = np.array([0, 2, 2, 3])
indices_np = np.array([0, 2, 1])
b = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
b.asnumpy()
```


```python
# Compare the two. They are exactly the same.
{'a':a.asnumpy(), 'b':b.asnumpy()}
```

你也可以通过使用`array`函数从`scipy.sparse.csr.csr_matrix`对象中创建一个MXNet CSRNDArray。


```python
try:
    import scipy.sparse as spsp
    # 使用scipy生成一个csr矩阵
    c = spsp.csr.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
    # 从一个scipy csr对象创建一个CSRNDArray
    d = mx.nd.sparse.array(c)
    print('d:{}'.format(d.asnumpy()))
except ImportError:
    print("scipy package is required")
```

但是如果你有一个大型的数据集，你也没有将其计算成indices和indptr的形式，你该怎么办呢？让我们从一个已经存在数组中创建一个简单的CSRNDArray，并使用内置的函数对其进行派生。我们可以使用随机数量的非零数据来模拟一个大的数据集，然后使用`tostype`函数将其压缩；在[Storage Type Conversion](#storage-type-conversion)一节中你可以看到更多的解释。


```python
big_array = mx.nd.round(mx.nd.random.uniform(low=0, high=1, shape=(1000, 100)))
print(big_array)
big_array_csr = big_array.tostype('csr')
# Access indices array
indices = big_array_csr.indices
# Access indptr array
indptr = big_array_csr.indptr
# Access data array
data = big_array_csr.data
# The total size of `data`, `indices` and `indptr` arrays is much lesser than the dense big_array!
```

你也可以通过`array`函数使用其他的CSRNDArray来创建一个新的CSRNDArray；而且，你还可以通过`dtype`来指定其中元素的类型（numpy dtype），其默认值为`float32`。


```python
# Float32 is used by default
e = mx.nd.sparse.array(a)
# Create a 16-bit float array
f = mx.nd.array(a, dtype=np.float16)
(e.dtype, f.dtype)
```

## 数组检查

有一系列的方法可以帮助你检查CSR数组，例如说：

* **.asnumpy()**
* **.data**
* **.indices**
* **.indptr**

就像你已经看到的那样，我们可以通过`asnumpy`函数将`CSRNDArray`转成`numpy.ndarray`格式，然后检查其元素内容。


```python
a.asnumpy()
```

你也可以通过访问CSRNDArray的属性，例如`indptr`，`indices`和`data`，来检查其内部存储。


```python
# Access data array
data = a.data
# Access indices array
indices = a.indices
# Access indptr array
indptr = a.indptr
{'a.stype': a.stype, 'data':data, 'indices':indices, 'indptr':indptr}
```

## 存储类型转换

你也可以使用下面的两个功能来转换存储的类型：

* **tostype**
* **cast_storage**

你可以通过``tostype``函数将NDArray转换成CSRNDArray，反之亦然。


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type from `default` to `csr`
csr = ones.tostype('csr')
# Cast the storage type from `csr` to `default`
dense = csr.tostype('default')
{'csr':csr, 'dense':dense}
```

或者你也可以使用`cast_storage`来转换存储类型：


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type to `csr`
csr = mx.nd.sparse.cast_storage(ones, 'csr')
# Cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(csr, 'default')
{'csr':csr, 'dense':dense}
```

## 复制

你可以使用`copy`方法创建一个新的数组，它是原数组的深层复制副本。你也可以使用`copyto`方法或者切片操作将数据复制到现有数组当中。


```python
# depp copy
a = mx.nd.ones((2,2)).tostype('csr')
b = a.copy()
# slice
c = mx.nd.sparse.zeros('csr', (2,2))
c[:] = a 
# copyto
d = mx.nd.sparse.zeros('csr', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```

当使用`copyto`函数或者切片操作时，如果源数组和目标数组的存储类型不一致，那么在复制后，目标数组的存储类型不会改变。


```python
e = mx.nd.sparse.zeros('csr', (2,2))
f = mx.nd.sparse.zeros('csr', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype, 'g.stype':g.stype}
```

## 索引和切片

对于CSRNDArray，你可以在0维上进行索引切片，这会复制得到一个新的CSRNDArray。


```python
a = mx.nd.array(np.arange(6).reshape(3,2)).tostype('csr')
b = a[1:2].asnumpy()
c = a[:].asnumpy()
{'a':a, 'b':b, 'c':c}
```

请注意，CSRNDArray目前还不支持多维索引或者在特定的维度上进行切片。

## 稀疏运算符和存储类型推断
我们在`mx.nd.sparse`中实现了专门为稀疏数组而设计的运算符。你可以通过查阅[mxnet.ndarray.sparse API文档](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html)来找到哪些稀疏运算符是可用的。


```python
shape = (3, 4)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
a = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape) # a csr matrix as lhs
rhs = mx.nd.ones((4, 1))      # a dense vector as rhs
out = mx.nd.sparse.dot(a, rhs)  # invoke sparse dot operator specialized for dot(csr, dense)
{'out':out}
```

对于任何稀疏运算符，输出数组的存储类型是通过输入来推断的。你可以通过查阅文档或者检查输出数组的`stype`属性的方式来确定推断出的存储类型格式。


```python
b = a * 2  # b will be a CSRNDArray since zero multiplied by 2 is still zero
c = a + mx.nd.ones(shape=(3, 4))  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```

对于那些并非为稀疏数组而设计的运算符，我们仍然可以在稀疏数组上调用它们，只是此时会有性能的损失。在MXNet中，稠密运算符要求所有的输入和输出数组都是稠密格式。

如果提供了稀疏格式的输入，MXNet会将稀疏格式的输入暂时转换为稠密格式，以便可以使用稠密运算符。

如果提供了稀疏格式的输出，MXNet会将稠密运算符生成的稠密输出转化为提供的稀疏格式。


```python
e = mx.nd.sparse.zeros('csr', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd.stype':d.stype, 'e.stype':e.stype} # stypes of a and e will be not changed
```

注意，当发生此类存储回退事件时，警告信息将被打印出来。如果你在使用jupyter notebook，这些警告信息将被打印在终端控制台中。

## 数据加载
你可以使用`mx.io.NDArrayIter`从CSRNDArray中加载批次数据：


```python
# 创建CSRNDArray数据源
data = mx.nd.array(np.arange(36).reshape((9,4))).tostype('csr')
labels = np.ones([9, 1])
batch_size = 3
dataiter = mx.io.NDArrayIter(data, labels, batch_size, last_batch_handle='discard')
# 检查数据批次
[batch.data[0] for batch in dataiter]
```

你也可以使用`mx.io.LibSVMIter`来加载以[libsvm文件格式](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)存储的数据，其数据格式为：``<label> <col_idx1>:<value1> <col_idx2>:<value2> ... <col_idxN>:<valueN>``。文件中的每一行都记录着非零条目的标签值，列索引和数据。例如，对于一个6列的矩阵来说，``1 2:1.5 4:-3.5``就意味着标签是``1``，数据是``[[0, 0, 1.5, 0, -3.5, 0]]``。更多关于`mx.io.LibSVMIter`的例子可以在[API文档](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter)中查到。


```python
# Create a sample libsvm file in current working directory
import os
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data.t')
with open(data_path, 'w') as fout:
    fout.write('1.0 0:1 2:2\n')
    fout.write('1.0 0:3 5:4\n')
    fout.write('1.0 2:5 8:6 9:7\n')
    fout.write('1.0 3:8\n')
    fout.write('-1 0:0.5 9:1.5\n')
    fout.write('-2.0\n')
    fout.write('-3.0 0:-0.6 1:2.25 2:1.25\n')
    fout.write('-3.0 1:2 2:-1.25\n')
    fout.write('4 2:-1.2\n')

# Load CSRNDArrays from the file
data_train = mx.io.LibSVMIter(data_libsvm=data_path, data_shape=(10,), label_shape=(1,), batch_size=3)
for batch in data_train:
    print(data_train.getdata())
    print(data_train.getlabel())
```

请注意，在文件中列索引需要按照行的升序进行排列，并且索引值是从0而不是从1开始的。

## 高级主题

### GPU支持

默认情况下，`CSRNDArray`操作符在CPU上执行。在MXNet中，对于`CSRNDArray`的GPU支持仍然是实验性质的，只有像`cast_storage`和`dot`少数一些操作才支持在GPU上执行。

想要在GPU上创建`CSRNDArray`，我们需要指定`context`。

**注意**在下一节中，如果GPU不可用，那么代码将会报告一个错误。若想要在CPU上执行此操作，请将`gpu_device`设置为`mx.cpu()`.


```python
import sys
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
try:
    a = mx.nd.sparse.zeros('csr', (100, 100), ctx=gpu_device)
    a
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```



<!-- INSERT SOURCE DOWNLOAD BUTTONS -->




