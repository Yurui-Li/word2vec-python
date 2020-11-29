# word2vec-python

## 摘要

这是一个练习，做的事情是根据word2vce的C源码，改写成python3形式，目的是巩固一下对word2vec的原理和代码实现的理解。

## 学习资料
首先，列一下学习word2vec的资料：

论文两篇：

《Distributed Representations of Words and Phrases》

《Efficient Estimation of Word Representations in Vector Space》

数学原理博客：

https://blog.csdn.net/itplus/article/details/37969519

源码解析：

https://blog.csdn.net/jeryjeryjery/article/details/80245924

https://cloud.tencent.com/developer/article/1066888

源码下载：

Google的源码需要一番努力才能下载到，如果没有条件的可以使用下面的百度云链接下载

链接：https://pan.baidu.com/s/16eBN5erf-HgDIscAoglU3A 
提取码：yjt8 


## 未竟之处
1.Google的源码之中有使用C版本的多线程来加快运算。
限于笔者的编程水平，未能改写成python3版本的多线程。

2.笔者仅仅只是调通，并不是完整运行，并和源码的结果进行对比。

3.源码之中还有对结果k-means聚类，也没有实现。

笔者水平有限，欢迎大佬发现问题后issue。
