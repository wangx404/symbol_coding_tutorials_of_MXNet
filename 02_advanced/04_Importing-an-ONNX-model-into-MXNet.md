
# 在MXNet中导入ONNX模型

在本教程中，我们将会：
- 学习如何在MXNet中加载一个预训练的ONNX模型
- 在MXNet中实现推断过程

## 准备工作

本示例假定已安装下列python库：

- [mxnet](http://mxnet.incubator.apache.org/install/index.html)
- [onnx](https://github.com/onnx/onnx)
- Pillow：python图像处理库，用于输入图像的预处理，你可以使用pip进行安装```pip install Pillow```
- matplotlib


```python
from PIL import Image
import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet.test_utils import download
from matplotlib.pyplot import imshow
```

### 获取所需文件


```python
img_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_res_input.jpg'
download(img_url, 'super_res_input.jpg')
model_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_resolution.onnx'
onnx_model_file = download(model_url, 'super_resolution.onnx')
```

## 在MXNet中加载模型

在MXNet中想要完整描述一个预训练的模型，我们需要两种元素：包含了模型结构定义的符号图和一个包含了模型权重的二进制文件。你可以通过使用``import_model``API加载一个ONNX模型并获取模型的符号和参数对象。其中参数对象分为argument参数和辅助参数。


```python
sym, arg, aux = onnx_mxnet.import_model(onnx_model_file)
```

现在我们可以可视化导入的模型（你需要首先安装好graphviz）


```python
mx.viz.plot_network(sym, node_attrs={"shape": "oval", "fixedsize": "false"})
```

## 输入预处理

我们将之前下载的图片转换为张量。


```python
img = Image.open('super_res_input.jpg').resize((224, 224))
img_ycbcr = img.convert("YCbCr")
img_y, img_cb, img_cr = img_ycbcr.split()
test_image = np.array(img_y)[np.newaxis, np.newaxis, :, :]
```

## 使用MXNet的Module API运行推断

我们将使用MXNet的Module API实现推断过程。为了实现这个目的，我们需要创建一个模型，将其与输入数据绑定，从argument参数和辅助参数对象中分配加载的权重。

为了获取输入数据的名称，我们运行下面的代码，它将排除argument和辅助参数，取出符号图的所有输入。


```python
data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg and graph_input not in aux]
print(data_names)
```


```python
mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[(data_names[0], test_image.shape)], label_shapes=None)
mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)
```

Module API的forward方法需要批次数据作为输入。我们将数据准备成这种格式，并将其输入给forward方法。


```python
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# forward on the provided data batch
mod.forward(Batch([mx.nd.array(test_image)])) # Batch(data=ndarray)
```

你需要使用``module.get_outputs()``方法获取之前前向计算的结果。我们将返回的``ndarray``转换为``numpy``数组，然后转成Pillow的图片格式。


```python
output = mod.get_outputs()[0][0][0]
img_out_y = Image.fromarray(np.uint8((output.asnumpy().clip(0, 255)), mode='L'))
result_img = Image.merge(
"YCbCr", [
                img_out_y,
                img_cb.resize(img_out_y.size, Image.BICUBIC),
                img_cr.resize(img_out_y.size, Image.BICUBIC)
]).convert("RGB")
result_img.save("super_res_output.jpg")
```

现在你可以对比一下输入图片和输出图片。你会注意到，这个模型能够将图片的分辨率从``256x256``提高到``672x672``。

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
