# 人工智能实践：探究不同解码策略对机器翻译模型翻译结果的影响

**介绍：**

> 人工智能实践项目，使用transformer构建的机器翻译系统，探究不同解码策略对机器翻译模型翻译结果的影响
>
> 数据集：iwslt’14 de-en
>
> 训练模型及预处理后训练集下载地址：https://www.aliyundrive.com/s/jEqtJPU7RgL

**运行环境:**

> pytorch_gpu:1.10
>
> nltk

**目录结构：**

```sh
├── data 
│   ├── de2en_5k #bpe大小为5k的数据集
│   └── raw_data #原始的数据集
├── model
│   └── Transformer.py #Transformer模型文件
├── checkpoints-ori #训练过程和训练结果保存目录
├── tool
│   ├── DataTool.py #数据载入工具类
│   ├── Global.py # Transformer模型参数及 其它参数（本实践的主要参数配置）
│   └── TrainTool.py # Warm-up函数
├── preprocess  #预处理数据
├── test.py #解码测试
├── decoding_strategy.py #解码策略（本实践需要完成的部分）
└── train.py #模型训练
```

**待实践部分：**解码策略.py

## 0、数据预处理（可跳过）

按顺序执行以下两个脚本

```sh
python preprocess/S1_GenerateBpe.py
python preprocess/S2_ApplyBpe.py
```

## 1、训练

### 1.1 从头训练

1. 配置相关参数

   ```python
   # vim Globay.py
   
   root="/headless/Desktop/人工智能实习3/" # 项目路径
   
   max_tokens = 16000  # 每个batch所包含的单词数量,根据显卡显存调整(12G:8000,24G:16000)
   epochs = 20  # 训练迭代轮次。
   init_model_number=0 # 0表示从头训练；>=1时，表示载入chekpoint，继续训练
   pre_batch_num_per_epoch=570 # 继续训练时，需配置该参数，570为之前训练时，一个epoch所包含的batch的数量
   
   #解码策略：sampling、beam、greedy、topK、topP
   decode_method="sampling"
   ```

2. 开始训练

   ```sh
   python train.py
   ```

### 1.2 载入保存点，继续训练

1. 配置模型参数，确保与训练时的模型参数一致

   ```sh
   # vim Globay.py
   
   root="/headless/Desktop/人工智能实习3/" # 项目路径
   
   max_tokens = 16000  # 每个batch所包含的单词数量,根据显卡显存调整(12G:8000,24G:16000)
   epochs = 20  # 训练迭代轮次。
   init_model_number=0 # 0表示从头训练；>=1时，表示载入chekpoint，继续训练
   pre_batch_num_per_epoch=570 # 继续训练时，需配置该参数，570为之前训练时，一个epoch所包含的batch的数量
   
   #解码策略：sampling、beam、greedy、topK、topP
   decode_method="sampling"
   ```

2. 开始训练

   ```sh
   python train.py
   ```

### 1.3 训练过程

此处训练参数设置为3层，4头，256维度，bpe大小为2k

```sh
gpu模式
载入已有字典
encoder_chars: 2004
decoder_chars: 2004
max_enc_seq_length: 363
max_dec_seq_length: 278
数据集大小:172530
数据集大小:872
batch num:1120
Epoch:0001  prog:100.0017% batch:1120/1120 batch_size:180 mean_loss=5.996997 mean_accu=11.41% lr=0.000098
进行测试中...batch num:44
valid accuracy:22.38 % test loss:4.4169
......
......
batch num:1120
Epoch:0020  prog:100.0017% batch:1120/1120 batch_size:200 mean_loss=2.138918 mean_accu=68.88% lr=0.000418
进行测试中...batch num:44
valid accuracy:70.41 % test loss:1.4048
```



### 1.4 查看训练过程

```sh
cat checkpoints-ori/de2en_2k.txt

n_layers3  n_heads:4  d_model:256  d_ff:1024  batch_size:20  encoder_len:363  decoder_len:278
Epoch:0001  batch:1120  loss=5.996997  train_accu=11.412931  valid_accu=22.383728  lr=0.000098
Epoch:0002  batch:1120  loss=4.755715  train_accu=24.517178  valid_accu=36.618428  lr=0.000196
Epoch:0003  batch:1120  loss=3.974497  train_accu=35.328851  valid_accu=45.392770  lr=0.000293
Epoch:0004  batch:1120  loss=3.470239  train_accu=43.350113  valid_accu=52.148458  lr=0.000391
Epoch:0005  batch:1120  loss=3.135638  train_accu=49.315405  valid_accu=56.918749  lr=0.000489
Epoch:0006  batch:1120  loss=2.912801  train_accu=53.593547  valid_accu=59.448717  lr=0.000587
Epoch:0007  batch:1120  loss=2.763752  train_accu=56.515822  valid_accu=62.279953  lr=0.000685
Epoch:0008  batch:1120  loss=2.642074  train_accu=58.967584  valid_accu=63.859193  lr=0.000660
Epoch:0009  batch:1120  loss=2.534271  train_accu=61.111251  valid_accu=65.178845  lr=0.000623
Epoch:0010  batch:1120  loss=2.453718  train_accu=62.700495  valid_accu=66.578826  lr=0.000591
Epoch:0011  batch:1120  loss=2.393994  train_accu=63.887431  valid_accu=67.546647  lr=0.000563
Epoch:0012  batch:1120  loss=2.344394  train_accu=64.858695  valid_accu=67.881313  lr=0.000539
Epoch:0013  batch:1120  loss=2.303971  train_accu=65.646550  valid_accu=68.559155  lr=0.000518
Epoch:0014  batch:1120  loss=2.270668  train_accu=66.321382  valid_accu=68.881203  lr=0.000499
Epoch:0015  batch:1120  loss=2.240943  train_accu=66.875999  valid_accu=69.428752  lr=0.000482
Epoch:0016  batch:1120  loss=2.216345  train_accu=67.376881  valid_accu=69.645483  lr=0.000467
Epoch:0017  batch:1120  loss=2.193870  train_accu=67.818207  valid_accu=69.951239  lr=0.000453
Epoch:0018  batch:1120  loss=2.173374  train_accu=68.200069  valid_accu=70.102368  lr=0.000440
Epoch:0019  batch:1120  loss=2.155479  train_accu=68.574355  valid_accu=70.356942  lr=0.000428
Epoch:0020  batch:1120  loss=2.138918  train_accu=68.877916  valid_accu=70.409328  lr=0.000418
```

## 2、测试不同解码策略，并计算BLEU得分

1. 配置模型参数，确保与训练时的模型参数一致

   ```sh
   # vim Globay.py
   
   #解码策略：sampling、beam、greedy、topK、topP
   decode_method="sampling"
   ```

2. 修改“测试.py”

   ```python
   model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
   # 载入第20轮的模型进行测试
   m_state_dict = torch.load(modelPath(20))
   model.load_state_dict(m_state_dict)
   ```
   
   
   
3. 运行测试

   ```sh
   python test.py
   
   原文：nun , das bringt ihnen keinen doktortitel in harvard , aber es ist viel interessanter als stamina zu zählen . 
   参考译文：now , that &apos;s not going to get you a ph.d. at harvard , but it &apos;s a lot more interesting than counting stamens . 
   解码结果：log概率(-70.937)	now , this doesn &apos;t causes you undermost topeless doctoratics -- but it &apos;s much more interesting than doing to count cartoon . 
   
   原文：der punkt ist hier , dass wir dinge mit der sozialen umgebung anstellen können . es werden jetzt daten von allen gesammelt -- von dem gesamten kollektiven speicher davon , wie , visuell die erde aussieht -- und alles wird miteinander verbunden . 
   参考译文：what the point here really is is that we can do things with the social environment . this is now taking data from everybody -- from the entire collective memory of , visually , of what the earth looks like -- and link all of that together . 
   解码结果：log概率(-104.006)	the point here is we four sort of doing things <?>it &apos;s collecting , and it &apos;s now going to data worldwide that &apos;s data from all the collective memory of what , visual side the earth looks like challenges -- everything is connected . 
   
   原文：wir konzentrieren uns nun auf treibstoffe der vierten generation . 
   参考译文：we &apos;re focusing on now fourth-generation fuels . 
   解码结果：log概率(-8.858)	we &apos;re now focused on the four generation fuels . 
   
   ......
   ......
   
    
   bleu：0.5828,0.3548,0.2298,0.1537
   mean bleu：0.3303
   ```
   

