### <center>Quora Question Pairs （短文本主题相似）</center>

### Siamese网络结构

<img src="https://img-blog.csdn.net/20170704193321782?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGhyaXZpbmdfZmNs/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="图片名称" align=center>

### 空间金字塔网络结构



<img src="https://img-blog.csdn.net/20180611113108398?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzMzNzQxNTQ3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="图片名称" align=center>

### 代码组织结构

<pre><code>
├── PreProcess.py: 数据预处理，
├── README.md
├── __init__.py
├── data
│   ├── csv
│   │   └── train.csv: 源数据
│   └── pkl: 暂存的中间结果
│       ├── test.pkl
│       ├── train.pkl
│       └── vocab.model
├── model
│   ├── MatchPyramid.py: 空间金字塔模型
│   ├── SiameseBiLSTM.py: 孪生语义网络，子网络是双向LSTM
│   ├── SiameseCNN.py: 孪生语义网络，子网络是CNN
│   ├── SiameseLSTM.py: 孪生语义网络，子网络是LSTM
│   ├── __init__.py
├── papers: 相关论文
├── summary
│   └── SiameseCNN: SiameseCNN的TensorBoard值
└── train.py: 模型训练
</code></pre>
	

### 代码

####PreProcess.py: 数据预处理

```

class data:
    :param data_path: 训练数据集: /data/csv/train.csv
    
    :function text_to_wordlist: 统一一部分词汇, 并返回词组
    :function get_batch: 用于批处理训练分割数据
    :function get_ont_hot: 对文本词汇进行编号
```

#### train.py 模型训练

```
class siamese_cnn_train:    
    :function __init__: 定义网络，优化方法，获取数据，定义TensorBoard
    :function train_step: 网络训练
    :function dev_step: 模型效果预测
    :function main: 神经网络训练入口
```

#### MatchPyramid.py: 空间金字塔模型

<img src="https://img-blog.csdn.net/20180611113108398?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzMzNzQxNTQ3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="图片名称" align=center>

#### Siamese*.py

<img src="https://img-blog.csdn.net/20170704193321782?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGhyaXZpbmdfZmNs/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="图片名称" align=center>

#### Contrastive Loss (博客链接)

<pre><code>
http://blog.csdn.net/autocyz/article/details/53149760
</code></pre>



### 相关参考资料和论文

[1]  Ways of Asking and Replying in Duplicate Question Detection<br>
&ensp;&ensp;&ensp;&ensp;&ensp;http://www.aclweb.org/anthology/S17-1030 <br>

[2]  英文博客<br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07<br>

[3]  中文博客<br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://www.leiphone.com/news/201802/X2NTBDXGARIUTWVs.html<br>

[4]  Quora Question Duplication <br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://web.stanford.edu/class/cs224n/reports/2761178.pdf <br>

[5]  上海交通大学报告（非常重要）<br>
&ensp;&ensp;&ensp;&ensp;&ensp;http://xiuyuliang.cn/about/kaggle_report.pdf <br>

[6]  Deep text-pair classification with Quora’s 2017 question dataset<br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://explosion.ai/blog/quora-deep-text-pair-classification <br>

[7]  NOTES FROM QUORA DUPLICATE QUESTION PAIRS FINDING KAGGLE COMPETITION <br>
&ensp;&ensp;&ensp;&ensp;&ensp;http://laknath.com/2017/09/12/notes-from-quora-duplicate-question-pairs-finding-kaggle-competition/ <br>

