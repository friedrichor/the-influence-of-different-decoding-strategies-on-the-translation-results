import random

import numpy as np
import torch

from tool.Global import *
import torch.nn.functional as F
import math


# 随机搜索:在下一个token生成的时候，按照概率值，进行多项式采样，选择一个token
def sampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence):
    source_sentence = source_sentence.split(char_space)

    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(source_sentence) * 1.5

    # encoder为德语的编码
    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

    final_result.append([dec_vocab2id[char_start]])
    final_scores.append(0)

    enc_output = None

    while True:
        # 将之前解码出来的序列，放入编码器，继续解码
        dec_input = final_result[0]

        # 对该序列进行搜索
        if enc_output is None:
            enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                       enc_output)
        else:
            _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                              enc_output)

        # print('output = ', output)
        # prob为搜索到的单词在词表中的概率分布
        # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
        prob = F.softmax(output[-1], dim=-1)

        # 多项式采样，按照概率值，随机抽取一个词，放入final_result中，例如上述prob中，最大概率抽中1
        random_char_idx = torch.multinomial(prob, 1).data[0].item()
        # print('prob = ', prob)
        # print('random_char_idx =', random_char_idx)
        # print('prob[random_char_idx] =', prob[random_char_idx])

        final_result[0].append(random_char_idx)

        # 计算该单词的log概率，并将其累加
        final_scores[0] += math.log(prob[random_char_idx].item())

        # 判断该序列是否有必要继续搜索
        sentence_len = len(dec_input)
        last_word_id = dec_input[len(dec_input) - 1]
        last_word_vocab = dec_id2vocab[last_word_id]

        if last_word_vocab == char_end or sentence_len >= dec_max_len:
            # 解码到char_end，或者已经抵达最大长度，停止解码
            break

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]


# 贪心搜索:在下一个token生成的时候，按照概率值，选择概率值最大的token
def greedySearch(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence):
    source_sentence = source_sentence.split(char_space)

    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(source_sentence) * 1.5

    # encoder为德语的编码
    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

    final_result.append([dec_vocab2id[char_start]])
    final_scores.append(0)

    enc_output = None

    while True:
        # 将之前解码出来的序列，放入编码器，继续解码
        dec_input = final_result[0]

        # 对该序列进行搜索
        if enc_output is None:
            enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                       enc_output)
        else:
            _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                              enc_output)

        # print('output = ', output)
        # prob为搜索到的单词在词表中的概率分布
        # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
        prob = F.softmax(output[-1], dim=-1)

        random_char_idx = torch.argmax(prob).item()

        final_result[0].append(random_char_idx)

        # 计算该单词的log概率，并将其累加
        final_scores[0] += math.log(prob[random_char_idx].item())

        # 判断该序列是否有必要继续搜索
        sentence_len = len(dec_input)
        last_word_id = dec_input[len(dec_input) - 1]
        last_word_vocab = dec_id2vocab[last_word_id]

        if last_word_vocab == char_end or sentence_len >= dec_max_len:
            # 解码到char_end，或者已经抵达最大长度，停止解码
            break

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]


# top-k sampling
def topKSampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k):
    source_sentence = source_sentence.split(char_space)

    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(source_sentence) * 1.5

    # encoder为德语的编码
    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

    final_result.append([dec_vocab2id[char_start]])
    final_scores.append(0)

    enc_output = None

    while True:
        # 将之前解码出来的序列，放入编码器，继续解码
        dec_input = final_result[0]

        # 对该序列进行搜索
        if enc_output is None:
            enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                       enc_output)
        else:
            _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                              enc_output)

        # prob为搜索到的单词在词表中的概率分布
        # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
        prob = F.softmax(output[-1], dim=-1)

        if k > 0:
            indices_to_remove = prob < torch.topk(prob, k)[0][..., -1, None]
            prob[indices_to_remove] = 0

            # 多项式采样，按照概率值，随机抽取一个词，放入final_result中，例如上述prob中，最大概率抽中1
            random_char_idx = torch.multinomial(prob, 1).data[0].item()

            final_result[0].append(random_char_idx)

            # 计算该单词的log概率，并将其累加
            final_scores[0] += math.log(prob[random_char_idx].item())

            # 判断该序列是否有必要继续搜索
            sentence_len = len(dec_input)
            last_word_id = dec_input[len(dec_input) - 1]
            last_word_vocab = dec_id2vocab[last_word_id]

            if last_word_vocab == char_end or sentence_len >= dec_max_len:
                # 解码到char_end，或者已经抵达最大长度，停止解码
                break

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]


# top-p sampling
def topPSampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, p):
    source_sentence = source_sentence.split(char_space)
    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(source_sentence) * 1.5

    # encoder为德语的编码
    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

    final_result.append([dec_vocab2id[char_start]])
    final_scores.append(0)

    enc_output = None

    while True:
        # 将之前解码出来的序列，放入编码器，继续解码
        dec_input = final_result[0]

        # 对该序列进行搜索
        if enc_output is None:
            enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                       enc_output)
        else:
            _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                              enc_output)

        # prob为搜索到的单词在词表中的概率分布
        # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
        prob = F.softmax(output[-1], dim=-1)
        # 将prob倒序排序
        prob_sort_values, prob_sort_indices = torch.sort(prob, descending=True)
        # 计算cumsum累加和    如[4,3,2,1]经cumsum后为[4,7,9,10]
        prob_sort_cumsum = torch.cumsum(prob_sort_values, dim=-1)
        # 累加和>阈值的部分为topp的mask,也就是后续要删掉的    如tensor([True, True, False, False])这样的值
        mask = prob_sort_cumsum > p
        # 第0位必是False(如0.9>0.8时，按上式第0位为True，出错)
        mask[0] = False
        # 向后移一位(如[0.7,0.18,0.12],p=0.8,对应mask=[False,True,True],而我们先想要的是[False,False,True])
        mask[1:] = mask[:-1].clone()

        # 找到要删掉的索引(概率值低的元素所对应的索引)
        remove = prob_sort_indices[mask]
        # 置0,后续torch.multinomial无法取到概率为0的元素(即概率低的)
        prob[remove] = 0

        # 多项式采样，按照概率值，随机抽取一个词，放入final_result中
        random_char_idx = torch.multinomial(prob, 1).data[0].item()  # 得到的是选取元素的索引值

        final_result[0].append(random_char_idx)
        # print('random_char_idx =', random_char_idx)
        # print('prob[random_char_idx] =', prob[random_char_idx])

        # 计算该单词的log概率，并将其累加
        final_scores[0] += math.log(prob[random_char_idx].item())

        # 判断该序列是否有必要继续搜索
        sentence_len = len(dec_input)
        last_word_id = dec_input[len(dec_input) - 1]
        last_word_vocab = dec_id2vocab[last_word_id]

        if last_word_vocab == char_end or sentence_len >= dec_max_len:
            # 解码到char_end，或者已经抵达最大长度，停止解码
            break

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    # exit()
    return final_scores[0], final_result[0]


# 束搜索
# k:束宽
def beamSearch(device,model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k: int):
    source_sentence = source_sentence.split(char_space)

    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(source_sentence) * 1.5

    # encoder为德语的编码
    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

    final_result.append([dec_vocab2id[char_start]])
    final_scores.append(0)

    enc_output = None
    first = True
    while True:
        if not first :#第一次和之后区分开
            for n in range(0, k):
                # if final_result[n][-1] == 2:
                #     continue
                dec_input = final_result[n]  # 将之前解码出来的序列，放入编码器，继续解码
                # print(final_result[0])
                # 对该序列进行搜索
                if enc_output is None:
                    enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                               enc_output)
                else:
                    _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                      enc_output)

                # prob为搜索到的单词在词表中的概率分布
                # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
                prob = F.softmax(output[-1], dim=-1)

                prob_sort_values, prob_sort_indices = torch.sort(prob, descending=True)

                prob_topk_values = prob_sort_values[:k]  # [0.7,0.2,0.1]
                prob_topk_indices = prob_sort_indices[:k]  # [2,0,1]

                # print(dict3)#在每一个中，最极端的是选出k个，所以dict3也选出来k个，对这k进行计算，存储到quanju中，后面的同理，
                # 然后quanju排序取前k
                # print(n)
                # 设置变量，方便回溯
                frs = final_scores[n]
                for j in range(0, k):
                    if j == 0:
                        # print("dnnm",final_result[n])
                        final_result[n].append(prob_topk_indices[j].item())
                        # print("1235",final_result[n])
                        if prob_topk_indices[j].item() != 2:
                            final_scores[n] += math.log(prob_topk_values[j].item())  # [0, -0.314934056936339, -1.7277977651412721]
                    else:
                        fi_re = []
                        # print(final_result[n][:-1])
                        fi_re = final_result[n][:-1]  # 去掉最后一位4，这样才能加上112

                        # print()
                        fi_re.append(prob_topk_indices[j].item())
                        # print("1235", fi_re)
                        final_result.append(fi_re)
                        # print(final_result)#[[1, 53, 4], [1, 112, 4], [1, 53, 11], [1, 112, 11]],接下来求final_score
                        # 根据final_score的结果进行排序，排序的结果用于消去不要的，保留剩下的k个
                        if prob_topk_indices[j].item() != 2:
                            frs += math.log(prob_topk_values[j].item())
                        final_scores.append(frs)
                # print(final_result)
                # print(final_scores)[[1, 53, 4], [1, 112, 4], [1, 53, 11], [1, 112, 11]]
            # [-0.7097622096513343, -2.122625917856267, -2.145524124344232, -3.558387832549165]
            final_scores_punish = []
            for i in range(len(final_result)):
                final_scores_punish.append(final_scores[i] / math.pow(len(final_result[i]), 0.4))
            # 进行排序:
            key = []
            for i in range(0, len(final_result)):
                key.append(i)
            zongtidict = dict(zip(key, final_scores_punish))
            # print(zongtidict)#{0: -0.7097622096513343, 1: -2.122625917856267, 2: -2.145524124344232, 3: -3.558387832549165}
            # 进行对value的排序，只保留前K个。
            zongtidict2 = sorted(zongtidict.items(), reverse=True, key=lambda x: x[1])
            # print(zongtidict2)
            zongtidict3 = []

            for i in range(0, k):
                zongtidict3.append(zongtidict2[i])

            # print(zongtidict3)#[(0, -0.7097622096513343), (1, -2.122625917856267)]得到以下结果，
            # 找到它对应的队列final_result
            for i in range(0, len(final_result)):  # 把不是的剔除0
                flag = 0
                for pb in zongtidict3:
                    if i == pb[0]:
                        # print("heihei")
                        flag = 1

                if flag == 0:
                    final_result[i] = []  # 直接删除会导致位置混乱
                    final_scores[i] = 0

            # print(final_result)#[[1, 53, 4], [1, 112, 4], [], []]
            # 这个时候再删除：

            final_result = list(filter(None, final_result))
            listscore = []
            for i in range(0, len(final_scores)):
                if final_scores[i] != 0:
                    # final_scores[i]=(1/(math.pow(long,3/4)))*final_scores[i]
                    listscore.append(final_scores[i])
            # print(1)
            # print(listscore)
            final_scores = listscore
            # print(1)
            # print( final_scores)

            # mini=min(final_scores)
            # result=final_result[final_scores.index(mini)]
            sentence_len = len(dec_input)
            if sentence_len >= dec_max_len:
                # 已经抵达最大长度，停止解码
                break

        else:#第一次的时候
            first = False
            # 将之前解码出来的序列，放入编码器，继续解码
            dec_input = final_result[0]
            #print(final_result[0])
            # 对该序列进行搜索
            if enc_output is None:
                enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                           enc_output)

            else:
                _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                  enc_output)

            # prob为搜索到的单词在词表中的概率分布
            # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
            prob = F.softmax(output[-1], dim=-1)

            prob_sort_values, prob_sort_indices = torch.sort(prob, descending=True)

            prob_topk_values = prob_sort_values[:k]  # [0.7,0.2,0.1]
            prob_topk_indices = prob_sort_indices[:k]  # [2,0,1]

            #建立两个列表，分别存储下标和概率
            final_xiabiao = [[] for i in range(k)]
            final_gailv = [[] for i in range(k)]

            for i in range(0, k):
                final_xiabiao[i].append([dec_vocab2id[char_start]][0])
                final_xiabiao[i].append(prob_topk_indices[i].item())  # key(下标）
                final_gailv[i].append(0)
                final_gailv[i].append(prob_topk_values[i].item())  # value

                final_result.append(final_xiabiao[i])  # [[1],[53],[112]

                # 计算该单词的log概率，并将其累加
                final_sc = 0
                final_sc += math.log(final_gailv[i][1])#[0, -0.314934056936339, -1.7277977651412721]

                final_scores.append( final_sc)

            final_result = final_result[1:]
            final_scores = final_scores[1:]

    for i in range(len(final_result)):
        for j in range(len(final_result[i])):
            if final_result[i][j] == 2:
                final_result[i] = final_result[i][:j] # [1,8,11,5,2,6,7,2,2,8,10]
                break
    for i in range(len(final_result)):
        final_scores[i] = final_scores[i] / math.pow(len(final_result[i]), 0.4)
        print('len(final_result_pro[i]) = ', len(final_result[i]), end=' ')

    mini = max(final_scores)

    result = final_result[final_scores.index(mini)]
    print(mini, result)

    return mini, result

# Greedy Search 并行
def greedySearch_parallelization(device, model, enc_input, batch_size, dec_id2vocab, dec_vocab2id):
    dec_max_len = len(enc_input[0]) * 1.5

    final_result = [[dec_vocab2id[char_start]] for i in range(batch_size)]
    final_scores = [0 for i in range(batch_size)]

    enc_output = None

    while True:
        dec_input = [result for result in final_result]

        if enc_output is None:
            enc_output, output = model(enc_input.to(device), torch.tensor(dec_input).to(device),
                                       enc_output)
        else:
            _, output = model(enc_input.to(device), torch.tensor(dec_input).to(device),
                              enc_output)

        # print('output.size=',output.size())
        output = output.view((batch_size, output.size(0) // batch_size, output.size(1)))
        prob = F.softmax(output[:, -1, :], dim=-1)

        for i in range(batch_size):
            random_char_idx = torch.argmax(prob[i]).item()
            final_result[i].append(random_char_idx)
            final_scores[i] += math.log(prob[i][random_char_idx].item())

        sentence_len = len(dec_input[0])
        if sentence_len >= dec_max_len:
            break

    return final_scores, final_result

# top-k sampling 并行
def topKSampling_parallelization(device, model, enc_input, batch_size, dec_id2vocab, dec_vocab2id, k):
    dec_max_len = len(enc_input[0]) * 1.5

    final_result = [[dec_vocab2id[char_start]] for i in range(batch_size)]
    final_scores = [0 for i in range(batch_size)]

    enc_output = None

    while True:
        dec_input = [result for result in final_result]

        if enc_output is None:
            enc_output, output = model(enc_input.to(device), torch.tensor(dec_input).to(device),
                                       enc_output)
        else:
            _, output = model(enc_input.to(device), torch.tensor(dec_input).to(device),
                              enc_output)

        output = output.view((batch_size, output.size(0) // batch_size, output.size(1)))
        prob = F.softmax(output[:, -1, :], dim=-1)

        for i in range(batch_size):
            indices_to_remove = prob[i] < torch.topk(prob[i], k)[0][..., -1, None]
            prob[i][indices_to_remove] = 0

            random_char_idx = torch.multinomial(prob[i], 1).data[0].item()
            final_result[i].append(random_char_idx)
            final_scores[i] += math.log(prob[i][random_char_idx].item())

        sentence_len = len(dec_input[0])

        if sentence_len >= dec_max_len:
            break

    return final_scores, final_result


# top-p sampling 并行
def topPSampling_parallelization(device, model, enc_input, batch_size, dec_id2vocab, dec_vocab2id, p):
    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(enc_input[0]) * 1.5

    # 搜索结果
    final_result = [[dec_vocab2id[char_start]] for i in range(batch_size)]  # [[dec_vocab2id[char_start]]] * batch_size
    # 搜索结果的得分
    final_scores = [0 for i in range(batch_size)]  # [0] * batch_size

    enc_output = None

    # print(enc_input)
    # print(torch.tensor([enc_input]))
    while True:
        # 将之前解码出来的序列，放入编码器，继续解码
        dec_input = [result for result in final_result]

        # 对该序列进行搜索
        enc_output, output = model(enc_input.to(device), torch.tensor(dec_input).to(device), enc_output)

        output = output.view((batch_size, output.size(0) // batch_size, output.size(1)))

        # prob为搜索到的单词在词表中的概率分布
        # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
        prob = F.softmax(output[:, -1, :], dim=-1)

        for i in range(batch_size):
            # 将prob倒序排序
            prob_sort_values, prob_sort_indices = torch.sort(prob[i], descending=True)
            # 计算cumsum累加和    如[4,3,2,1]经cumsum后为[4,7,9,10]
            prob_sort_cumsum = torch.cumsum(prob_sort_values, dim=-1)
            # 累加和>阈值的部分为topp的mask,也就是后续要删掉的    如tensor([True, True, False, False])这样的值
            mask = prob_sort_cumsum > p
            # 第0位必是False(如0.9>0.8时，按上式第0位为True，出错)
            mask[0] = False
            # 向后移一位(如[0.7,0.18,0.12],p=0.8,对应mask=[False,True,True],而我们先想要的是[False,False,True])
            mask[1:] = mask[:-1].clone()

            # 找到要删掉的索引(概率值低的元素所对应的索引)
            remove = prob_sort_indices[mask]
            # 置0,后续torch.multinomial无法取到概率为0的元素(即概率低的)
            prob[i][remove] = 0

            # 多项式采样，按照概率值，随机抽取一个词，放入final_result中
            random_char_idx = torch.multinomial(prob[i], 1).data[0].item()  # 得到的是选取元素的索引值

            final_result[i].append(random_char_idx)

            # 计算该单词的log概率，并将其累加
            final_scores[i] += math.log(prob[i][random_char_idx].item())

        # # 判断该序列是否有必要继续搜索
        sentence_len = len(dec_input[0])

        if sentence_len >= dec_max_len:
            # 已经抵达最大长度，停止解码
            break

    return final_scores, final_result