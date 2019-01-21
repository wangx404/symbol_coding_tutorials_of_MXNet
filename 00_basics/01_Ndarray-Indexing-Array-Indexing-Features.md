
# NDArray索引-数组索引功能

MXNet的高级数组索引功能是模仿[NumPy文档](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#combining-advanced-and-basic-indexing)实现的。在这里你将会看到许多接近Numpy索引功能的改编，因此我们直接借用了他们的文档。

可以使用标准的python语法`x[obj]`对`NDArray`进行索引，其中_x_是数组，_obj_是选择。

有两种索引的方式：
1. 基础切片
2. 高级索引

仿照Numpy `ndarray`的索引规则，MXNet支持基础和高级的索引方式。

## 基础切片和索引

基础切片将python切片的基础内容扩展到N维。来快速浏览一下吧：

```
a[start:end] # 从start到end-1
a[start:]    # 从start到数组结束
a[:end]      # 从数组开始到end-1
a[:]         # 复制整个数组
```


```python
from mxnet import nd
```

对于基础切片的示例，我们先从一些简单的例子开始。


```python
x = nd.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int32')
x[5:]
```


```python
x = nd.array([0, 1, 2, 3])
print('1D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
```


```python
x = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('multi-D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
```


```python
# 行/列赋值
print('original x, x=', x)
x[2] = 9.0
print('replaced entire row with x[2] = 9.0, x=', x)
```


```python
# 单独元素赋值
print('original x, x=', x)
x[0, 2] = 9.0
print('replaced specific element with x[0, 2] = 9.0, x=', x)
```


```python
# 区域赋值
print('original x, x=', x)
x[1:2, 1:3] = 5.0
print('replaced range of elements with x[1:2, 1:3] = 5.0, x=', x)
```

## 1.0版本中的新索引功能

### 步长

在基础的切片语法`i:j:k`中，_i_是索引的起始点，_j_是索引的终点，_k_是步长（_k_必须是非零值）。

**注意**在之前的版本中，MXNet只支持步长为1的切片操作。而从1.0版本开始，MXNet支持任意步长的索引。


```python
x = nd.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int32')
# Select elements 1 through 7, and use a step of 2
x[1:7:2]
```

## 负数索引

负数的_i_和_j_将被解释为_n + i_和_n + j_，其中_n_是对应维度中元素的数目。而负数的_k_则意味着从大到小的索引顺序。


```python
x[-2:10]
```

如果选择元组中的对象数目小于N，那么后续的维度将被默认为使用`:`索引。


```python
x = nd.array([[[1],[2],[3]],
              [[4],[5],[6]]], dtype='int32')
x[1:2]
```

你可以使用切片的方式对数组进行赋值，但是（不像列表）你永远不能扩增数组。使用`x[obj] = value`进行赋值时，value的形状必须能够被广播成`x[obj]`的形状。


```python
x = nd.arange(16, dtype='int32').reshape((4, 4))
print(x)
```


```python
print(x[1:4:2, 3:0:-1])
```


```python
x[1:4:2, 3:0:-1] = [[16], [17]]
print(x)
```

## 1.0版本的高级索引功能

当选择对象obj是一个非元组序列对象（python列表），Numpy的`ndarray`（整数类型），MXNet的`NDArray`或者具有至少一个序列对象的元组时，MXNet的高级索引功能将被触发。

高级索引将会返回一份复制的数据。

**注意**：
- 使用Python的列表进行索引时，仅支持元素为整数的列表，不支持嵌套的列表。例如，MXNet支持`x[[1, 2]]`，但不支持`x[[1], [2]]`。
- 当使用numpy `ndarray`或者MXNet `NDArray`进行索引的时候，对维度无限制。
- 当索引对象是包含python列表的元组的时候，其整数列表和嵌套列表是都支持的。例如，MXNet支持`x[1:4, [1, 2]]`和`x[1:4, [[1], [2]]`。

### 纯整数数组索引

当索引序列由一系列数量等同于数组维度的整数数组组成时，索引以不同于切片的形式直接执行。

高级索引序列是被作为一个值进行[广播](https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#ufuncs-broadcasting)和迭代的。

    result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M], ..., ind_N[i_1, ..., i_M]] 

请注意，结果的形状和索引序列的形状`ind_1, ..., ind_N`相同。

**示例**：
对于每一行都选择一个特定的元素。这里行序号是[0, 1, 2, 2]，指定了对应的行；而列序号[0, 1, 0, 1]，指明了对应的列。将两者结合后可以通过高级索引解决。


```python
x = nd.array([[1, 2],
              [3, 4],
              [5, 6]], dtype='int32')
x[[0, 1, 2, 2], [0, 1, 0, 1]]
# 相当于将[0,0],[1,1],[2,0]位置的元素检索了出来
```

为了能够实现类似于上述基础切片的操作，需要使用广播。通过示例可以很好的理解这一点。

**示例**：
想要通过高级索引从4x3的数组中挑选出位于边角的元素值，你需要选择位于`[0, 2]`列和`[0, 3]`行的元素。因此你需要通过高级索引指明这一点。通过上面解释的方法，你可以这样来写：


```python
x = nd.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype='int32')
#x = nd.arange(20).reshape((5,4))

print(x[[[0, 0],[3, 3]],[[0,2], [0,2]]])
```

但是，由于上述的索引只是重复它们自己，因为可以使用广播来完成相同的索引。


```python
x = nd.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype='int32')
x[[[0], [3]],
  [[0, 2]]]
```

### 结合基础和高级索引

在三种情况下，我们需要考虑在一个索引对象中将基础和高级索引结合起来。让我们看一些示例来理解这一点。

- 索引对象中只有一个高级索引序列。例如，`x`是一个`shape=(10, 20, 30, 40, 50)`的`NDArray`，`result=x[:, :, ind]`中包含一个`shape=(2, 3, 4)`的高级索引序列`ind`。结果的形状将会是`(10, 20, 2, 3, 4, 40, 50)`。这是因为`x`第三维的子空间被形状为`(2, 3, 4)`的子空间替代。如果我们使用_i_，_j_，_k_对形状为(2,3,4)的子空间进行索引，那么结果等同于`result[:, :, i, j, k, :, :] = x[:, :, ind[i, j, k], :, :]`。


```python
import numpy as np
shape = (10, 20, 30, 40, 50)
x = nd.arange(np.prod(shape), dtype='int32').reshape(shape)
ind = nd.arange(24).reshape((2, 3, 4))
print(x[:, :, ind].shape)
```

- 索引对象中包含了两个彼此相邻的高级索引序列。例如，`x`是一个形状为`(10, 20, 30, 40, 50)`的`NDArray`，而`result=x[:, :, ind1, ind2, :]`中拥有的两个高级索引序列将被广播成形状为`(2, 3, 4)`的索引序列。之后，`result`的形状将变成`(10, 20, 2, 3, 4, 50)`，这是因为形状为`(30, 40)`的子空间被形状为`(2, 3, 4)`的子空间替代。


```python
# 现在索引序列间进行广播，然后进行高级索引
ind1 = [0, 1, 2, 3]
ind2 = [[[0], [1], [2]], [[3], [4], [5]]]
print(x[:, :, ind1,:,:].shape)
print(x[:, :, :, ind2, :].shape)
print(x[:, :, ind2, ind1, :].shape)
```

- 索引对象中包含了两个彼此分离的高级索引序列。例如，`x`是一个形状为`(10, 20, 30, 40, 50)`的`NDArray`，而`result=x[:, :, ind1, ind2, :]`中拥有的两个高级索引序列将被广播成形状为`(2, 3, 4)`的索引序列。之后`result`的形状将变成`(2, 3, 4, 10, 20, 40)`，这是因为没有明确的指明放置索引子空间的位置，于是它被添加到了数组的头部。


```python
print(x[:, :, ind1, :, ind2].shape)
```
