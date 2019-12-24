# Project 于宏正<br>

这是最终的project的源码. <br>

本项目是第三个题目, 知乎看山杯的project. 在三个文件夹中, 分别是:<br>

* topic_embed: 是对topic序列使用AutoEncdoer进行特征提取的过程.<br>

* question_embed: 是对问题标题序列使用seq2seq进行编码的过程.<br>

* lstm-dnn-model: 是最终搭建的lstm-dnn-based 多输入神经网络模型<br>

<br>
下面给出论文摘要, 具体介绍和信息详见论文
<br><br>
本文要解决的问题是知乎的看山杯2019的比赛问题. 
比赛将提供知乎的问题信息、用户画像、用户回答记录，以及用户接受邀请的记录，要求选手预测这个用户是否会接受某个新问题的邀请. 
本文提出了一个LSTM-DNN-Based的end-to-end多输入神经网络模型. 
并且, 使用了神经网络领域中两种重要的编码方式AutoEncoder 和Seq2Seq对 话题 信息和问题信息进行编码.
 在经过AutoEncoder编码话题信息之后, 可以补全缺失的话题数据. 而对问题信息进行Seq2Seq编码之后, 可以提取其中的语义信息. 
 之后, 本文构建了一个多输入神经网络模型, 使用LSTM-Based方法来提取用户的回答历史信息, 使用DNN-Based方法提取其中的用户信息,
  构建了一个end-to-end的预测模型.
  最后本文在比赛中的得分(AUC)为0.693.