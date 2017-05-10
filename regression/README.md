
#回归问题

回归问题是机器学习中的一个经典问题，主要目的是构建一个函数将输入数据与输出数据关联起来。本示例中利用机器翻译中的[WMT-14](https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation#数据介绍)，构建了一个相同结构的网络，对源数据与目标数据进行编码。迭代更新源数据网络的参数，来拟合目标数据的编码，完成了一个简单的回归问题。经典的线性回归，请参考[fit_a_line](https://github.com/PaddlePaddle/book/tree/develop/01.fit_a_line).
