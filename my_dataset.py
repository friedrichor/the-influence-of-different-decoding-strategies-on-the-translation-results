import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tool.Global import *


class MyDataSet(Dataset):
    def __init__(self, inputs, targets, enc_vocab2id, dec_vocab2id):
        self.inputs = inputs
        self.targets = targets
        self.enc_vocab2id = enc_vocab2id
        self.dec_vocab2id = dec_vocab2id

    def __getitem__(self, item):
        def enc2id(enc):
            try:
                return self.enc_vocab2id[enc]
            except:
                return 3  # <?>

        input = self.inputs[item].split(char_space)
        target = self.targets[item].split(char_space)

        input_vocab2id = []
        for w in input:
            input_vocab2id.append(enc2id(w))
        target_vocab2id = []
        for w in target:
            target_vocab2id.append(self.dec_vocab2id[w])

        return input_vocab2id, target_vocab2id

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def collate_fn(batch):
        # for example:
        # [([0,1],[2,3]),([4,5],[6,7])] -> (([0, 1], [4, 5]), ([2, 3], [6, 7]))
        inputs, targets = tuple(zip(*batch))

        inputs_list = [torch.tensor(input) for input in inputs]
        targets_list = [torch.tensor(target) for target in targets]
        inputs_pad = pad_sequence(inputs_list, batch_first=True)
        targets_pad = pad_sequence(targets_list, batch_first=True)

        return torch.tensor(inputs_pad), torch.tensor(targets_pad)


