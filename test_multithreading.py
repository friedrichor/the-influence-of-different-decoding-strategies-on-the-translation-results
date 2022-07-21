# *_*coding:utf-8 *_*
from nltk.translate.bleu_score import sentence_bleu
import math
import time

from tool.DataTool import *
import torch.nn.functional as F
from model.Transformer import Transformer
import multiprocessing
from multiprocessing import Pool
from functools import partial
import warnings

from decoding_strategy import *

warnings.filterwarnings("ignore")
# multiprocessing.set_start_method('spawn', force=True)
multiprocessing.set_start_method('forkserver', force=True)


def single_sentence(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, sentence):
    line = sentence

    enc_input = line.split('\t')[0]
    true_output = line.split('\t')[1].strip()
    enc_pre_1 = enc_input.replace(" ", "")
    enc_pre_1 = enc_pre_1.replace("<e>", " ")

    target_sentence = line.split("\t")[1].strip()
    target_sentence = target_sentence.replace(" ", "")
    target_sentence = target_sentence.replace("<e>", " ")

    writeGenerateToFile("原文：{}".format(enc_pre_1))
    writeGenerateToFile("参考译文：{}".format(target_sentence))

    enc_input = char_start + char_space + enc_input + char_space + char_end

    if decode_method == "sampling":
        decode_score, decode_result = sampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, enc_input)
    elif decode_method == "greedy":
        decode_score, decode_result = greedySearch(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, enc_input)
    elif decode_method == "topK":
        decode_score, decode_result = topKSampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, enc_input,
                                                   5)
    elif decode_method == "topP":
        decode_score, decode_result = topPSampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, enc_input,
                                                   0.5)
    elif decode_method == "beam":
        decode_score, decode_result = beamSearch(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id,
                                                 enc_input, 5)
    else:
        print("请使用有效的解码策略！")
        exit()

    # 将下标转化成句子
    sent = ''
    for w in decode_result:
        sent += dec_id2vocab[w] + ' '

    # 处理特俗符号
    sent = sent.replace(char_space, "")
    sent = sent.replace(word_end, " ")
    sent = sent.replace(char_start, "")
    sent = sent.replace(char_end, "")

    # 计算该条解码句子的bleu得分
    bleu_score_1 = sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                      weights=(1, 0, 0, 0))
    bleu_score_2 = sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                      weights=(0, 1, 0, 0))
    bleu_score_3 = sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                      weights=(0, 0, 1, 0))
    bleu_score_4 = sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                      weights=(0, 0, 0, 1))

    writeGenerateToFile('解码结果：log概率({:.3f})\t{}\n'.format(decode_score, sent))

    return bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4


if __name__ == '__main__':
    time_start = time.time()  # 开始计时

    if use_gpu:
        device = torch.device("cuda")
        print("gpu模式")
    else:
        device = torch.device("cpu")
        print("cpu模式")

    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    print('encoder_chars:', len(encoder_chars))
    print('decoder_chars:', len(decoder_chars))
    print('max_enc_seq_length:', max_enc_seq_length)
    print('max_dec_seq_length:', max_dec_seq_length)
    print("\n")

    enc_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    enc_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    dec_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    dec_id2vocab = {i: word for i, word in enumerate(decoder_chars)}

    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    m_state_dict = torch.load(modelPath(20))
    model.load_state_dict(m_state_dict)
    model = model.to(device)

    model.eval()
    all_bleu_score_1 = 0
    all_bleu_score_2 = 0
    all_bleu_score_3 = 0
    all_bleu_score_4 = 0
    with torch.no_grad():
        test_lines = open(test_file_path, 'r', encoding='utf-8').readlines()
        test_size = len(test_lines)

        pool = Pool(4)
        par_single_sentence = partial(single_sentence, device, model,
                                      enc_vocab2id, dec_id2vocab, dec_vocab2id)
        results = pool.map_async(par_single_sentence, test_lines)
        for res in results.get():
            all_bleu_score_1 += res[0]
            all_bleu_score_2 += res[1]
            all_bleu_score_3 += res[2]
            all_bleu_score_4 += res[3]

        pool.close()
        pool.join()
        # 计算所有句子的平均bleu得分
        avg_bleu_score_1 = all_bleu_score_1 / test_size
        avg_bleu_score_2 = all_bleu_score_2 / test_size
        avg_bleu_score_3 = all_bleu_score_3 / test_size
        avg_bleu_score_4 = all_bleu_score_4 / test_size
        writeGenerateToFile("bleu：{:.4f},{:.4f},{:.4f},{:.4f}".format(avg_bleu_score_1,
                                                                      avg_bleu_score_2,
                                                                      avg_bleu_score_3,
                                                                      avg_bleu_score_4))

        writeGenerateToFile("mean bleu：{:.4f}".format((avg_bleu_score_1 + avg_bleu_score_2 + avg_bleu_score_3 + avg_bleu_score_4) / 4))

    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')