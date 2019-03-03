# GeoMAN
a easy pytorch implement of GeoMAN

[GeoMAN](http://yuxuanliang.com/assets/pdf/ijcai-18/paper.pdf) : Multi-level Attention Networks for Geo-sensory Time Series Prediction.

## Paper
Yuxuan Liang, Songyu Ke, Junbo Zhang, Xiuwen Yi, Yu Zheng, "GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction", IJCAI, 2018.

## Application Scene

<div align=center><img src="https://github.com/xchadesi/GeoMAN/blob/master/scene.png"/></div>

上图是北京市地区的空气检测站的分布图，每一个检测站被称之为一个sensor，每一个sensor在一天之内都会间隔固定的时间（一般是5min）采集空气数据，其中包括：温度、湿度、PM2.5（是本文需要预测的指标---目标属性）、NO、NO2、以及各个方向的风力大小等19个维度的属性特征。这种场景的数据特点就是，每一个sensor地理位置是不会变化的，它们之间的相对位置也是不变的，但是由于是每隔一段时间就会采集一次数据，因此每一个sensor都会产生一系列的时序数据。假设sensor的数量是Ng个，每一个sensor采集的属性特征数为Nl。分析任务就是，给一个时间间隔T范围内的所有sensor数据，来预测某一个sensori在接下来的T+τ时间段内的某一维属性特征序列值。从问题描述我们可以知道，对于其中的一个sensor X，它在一段时间T内产生的数据可以用矩阵来表示，这个矩阵的维度是Nl∗T。

## Model Framework

从问题分类上看，这是一个时间序列的回归问题，损失函数也就是简单的均方损失函数。由于是序列生成序列，故本文采用的是经典的seq2seq架构（在NLP中常用于机器翻译和摘要生成等任务），因此一定会使用encoder-decoder架构，同时由于该任务的特殊性，本文在decoder生成阶段又巧妙的引入了其他的外部信息（比方说sensor对应的poi信息，天气信息以及sensor ID信息），显著地提升了模型的性能。整个模型如下所示： 
<div align=center><img src="https://github.com/xchadesi/GeoMAN/blob/master/model.png"/></div>

可以看出在encoder和decoder该模型都使用了LSTM，在LSTM的输入部分可以看到一个被称之为“Spatial Attn”的结构，从右边放大的框图可以看出这个结构由“Local”和“Global”拼接而成即[local_x; global_x]，这个“Local”指的是当前被预测的sensori的信息编码，“Global”指的则是其他sensor的信息编码，这两个编码过程中都是用了Attention机制。
<div align=center><img src="https://github.com/xchadesi/GeoMAN/blob/master/local_global_attention.PNG"/></div>
<div align=center><img src="https://github.com/xchadesi/GeoMAN/blob/master/temporal_attention.PNG"/></div>


## Model Input
The model has the following inputs:<br>
- local_inputs: the input of local spatial attention, shape->[batch_size, n_steps_encoder, n_input_encoder]<br>
- global_inputs: the input of global spatial attention, shape->[batch_size, n_steps_encoder, n_sensors]<br>
- external_inputs: the input of external factors, shape->[batch_size, n_steps_decoder, n_external_input]<br>
- local_attn_states: shape->[batch_size, n_input_encoder, n_steps_encoder]<br>
- global_attn_states: shape->[batch_size, n_sensors, n_input_encoder, n_steps_encoder]<br>
- labels: ground truths, shape->[batch_size, n_steps_decoder, n_output_decoder]<br>

### How to understand the input data?

（1）每个传感器可以记录多个维度的数据（文中是19个），其中一个是目标属性，其它是相关属性

（2）一个空间里面，可以部署多个传感器（文中是35个）

（3）所以，总的数据格式是：T_data = [500条（每一条数据在时间上是顺移的关系）数据：[35个传感器[每个传感器包含19个其它维度[每个维度包含12个时间步]]]]

（4）在没有确定时间步的原始数据应该是：global_data = [N个时刻：[35个传感器[每个传感器包含19个维度]]]

所有数据都由global_data产生，先由global_data根据时间步产生T_data，再产生如下的数据：

- local_inputs.npy: (100, 12, 19) 是研究目标，也就是选定一个传感器，作为研究对象 (所以传感器维度消失)<br>
- global_input_indics.npy: (100,) 选定一个传感器的情况下，选取100条数据<br>
- global_attn_state_indics.npy:(100,) 选定一个传感器的情况下，选取100条数据<br>
- external_inputs.npy: (100, 6, 83) 选定一个传感器的情况下，选取100条该传感器其它的特征数据（83维，按6个时间步）<br>
- decoder_gts.npy: (100, 6) 选定一个传感器的情况下，对应选取100条目标属性的真实值<br>
- global_inputs.npy: (500, 35) 所有的传感器目标属性数据<br>
- global_attn_state.npy: (500, 35, 19, 12) 所有的传感器数据<br>

### How to understand the Local atttion and Global attention

从500条数据中，选取一条数据包含：data = [35个传感器[每个传感器包含19个其它维度[每个维度包含12个时间步]]] <br>

 对于第i个传感器（每个传感器都有一个目标属性）（i∈（1,35））：<br>
 
 计算局部注意力：local_data = data[每个传感器包含19个其它维度[每个维度包含12个时间步]]<br>
 
              local_x = attention(local_data)
              
 计算全局注意力：global_data = data[35个传感器[每个传感器包含12个时间步，目标属性的维度]]  <br>
 
              global_x = attention(global_data)
              
 对每个i传感器的目标属性：x = [local_x, global_x]

## Reference

[1]  Tensorflow implements: [GeoMAN](https://github.com/yoshall/GeoMAN)<br>

[2]  https://blog.csdn.net/guoyuhaoaaa/article/details/80564356<br>
