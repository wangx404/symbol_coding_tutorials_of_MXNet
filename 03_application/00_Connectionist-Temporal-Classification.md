
# 联结主义时间分类


```python
import mxnet as mx
print(mx.__version__)
```

[联结主义时间分类(Connectionist Temporal Classification，CTC)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)是一种监督学习中的成本函数，它常被用来训练递归神经网络(Recurrent Neural Networks, RNNs)用以标记未分段的输入序列数据。例如，在语音识别中，如果使用典型的交叉熵损失函数，输入信号需要被分割成单词或者子单词。但是如果使用了CTC损失函数，为输入序列提供一个标签序列就够了，网络可以同时学习到数据对齐和标签信息。百度的warp-ctc页面中包含了关于[CTC损失](https://github.com/baidu-research/warp-ctc#introduction)更详细的介绍。

## MXNet中的CTC损失
MXNet在Symbol API中提供了两种CTC损失接口：
- `mxnet.symbol.contrib.ctc_loss`作为MXNet标准包的一部分已被实现。
- `mxnet.symbol.WarpCTC`使用了百度的warp-ctc库，因此你需要同时从源代码中构建warp-ctc库和mxnet库。

## LSTM OCR示例
MXNet的示例文件夹中包含了一个[CTC的示例](https://github.com/apache/incubator-mxnet/tree/master/example/ctc)，其中使用了LSTM网络和CTC损失函数用于在验证码（CAPTCHA）图片上执行光学字符识别（Optical Character Recognition， OCR）预测。该示例演示了两种CTC损失函数的使用，以及使用网络符号和参数检查点在训练结束后进行预测推断。

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
