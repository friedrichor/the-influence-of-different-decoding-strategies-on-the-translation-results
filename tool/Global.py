# *_*coding:utf-8 *_*

# root="/headless/Desktop/人工智能实习3/"  # 项目路径
# root="/content/drive/MyDrive/trans/"  # 项目路径
root = "E:/pythonProjects/course_AIpractice2/code-13/"  # 项目路径
weights_path = "checkpoints-ori/"  # 模型保存路径

max_tokens = 8000  # 每个batch所包含的单词数量,根据显卡显存调整(12G:8000,24G:16000)
epochs = 20  # 训练迭代轮次。
init_model_number = 0  # 0表示从头训练；>=1时，表示载入chekpoint，继续训练
pre_batch_num_per_epoch = 570  # 继续训练时，需配置该参数，570为之前训练时，一个epoch所包含的batch的数量

#解码策略：sampling、beam、greedy、topK、topP
decode_method = "greedy"

# 是否使用GPU
use_gpu = True


# encoder和decoder的层数
num_layers = 6
# 多头注意力中的头数
num_heads = 8  # base:8, big:16
# 字嵌入和位置嵌入的维度
d_model = 512  # base:512, big:1024
embedding_dim = d_model
# 全连接
d_ff = d_model * 4


# 特殊字符
char_space = ' '
char_start = '<start>'
char_end = '<end>'
char_unknown = '<?>'
word_end = '<e>'  # bpe标志


# 数据集
corpus_de_path = root+'data/raw_data/de.txt'
corpus_en_path = root+'data/raw_data/en.txt'

# bpe字典
encoder_bpe_dic_path = root+'data/de2en_5k/de_5000.txt'
decoder_bpe_dic_path = root+'data/de2en_5k/en_5000.txt'

# bpe切分后的数据集
combined_vocab_path = root+'data/de2en_5k/de2en_5k_vocab.txt'

# 重新切分并预处理好的数据集
train_file_path = root+"data/de2en_5k/train_5k.txt"
valid_file_path = root+"data/de2en_5k/valid_5k.txt"
test_file_path = root+"data/de2en_5k/test_5k.txt"


# 训练集字典信息
data_path_vocab_desc = root+'data/de2en_5k/corps_5k_desc.txt'



# 保存模型名称
modelName = "de2en_5k"


# 模型存储路径
def modelPath(epoch):
    return root + weights_path + modelName + '_%04d' % (epoch) + '.pt'


# 打印训练进度
def printProgress(epoch, prog, batch_no, batch_all, batch_size, loss, accu, lr):
    print('\rEpoch:%04d  prog:%.4f%% batch:%d/%d batch_size:%d mean_loss=%.6f mean_accu=%.2f%% lr=%.6f' % (
        epoch, prog, batch_no, batch_all, batch_size, loss, accu, lr), end="")


# 输出参数到文件
def writeParametersToFile(n_layers, n_heads, d_model, d_ff, encoder_len, decoder_len):
    progress = 'n_layers' + '%d' % (n_layers) + \
               '  n_heads:%d' % (n_heads) + \
               '  d_model:%d' % (d_model) + \
               '  d_ff:%d' % (d_ff) + \
               '  encoder_len:%d' % (encoder_len) + \
               '  decoder_len:%d' % (decoder_len) + "\n"
    with open(root + weights_path + modelName + '.txt', 'a') as f:
        f.write('\n')
        f.write(progress)


# 输出训练进度到文件
def writeProgreeToFile(epoch, batch_all, loss, train_accu, valid_accu, lr):
    progress = 'Epoch:' + '%04d' % (epoch) + \
               '  batch:%d' % (batch_all) + \
               '  loss=' + '{:.6f}'.format(loss) + \
               '  train_accu=' + '{:.6f}'.format(train_accu) + \
               '  valid_accu=' + '{:.6f}'.format(valid_accu) + \
               '  lr=' + '{:.6f}\n'.format(lr)
    with open(root + weights_path + modelName + '.txt', 'a') as f:
        f.write(progress)


# 输出解码结果到文件
def writeGenerateToFile(content):
    # 解码结果保存文件
    decode_path = root + weights_path + modelName + "_generate_{}.txt".format(decode_method)
    with open(decode_path, 'a',encoding='utf-8') as f:
        print(content)
        content="{}\n".format(content)
        f.write(content)