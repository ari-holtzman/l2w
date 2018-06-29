import sys, os, pickle
import numpy as np
import torch
from torch.autograd import Variable
sys.path.insert(0, '../utils/') # for loading dictionary pickle

class Corpus(object):
    def __init__(self, path, dic_path):
        with open(dic_path, 'rb') as dic_file:
            self.dictionary = pickle.load(dic_file)
        self.train_path = os.path.join(path, 'train.txt')
        self.valid_path = os.path.join(path, 'valid.txt')
        self.test_path = os.path.join(path, 'test.txt')
        self.splits = {
                          'train' : self.train_path,
                          'valid' : self.valid_path,
                          'test'  : self.test_path
                      }

    def iter(self, split, bsz, seq_len, use_cuda=True, evaluation=False):
        """Tokenizes a text file."""
        # Tokenize file content
        if split in self.splits:
            path = self.splits[split]
        else:
            raise LookupError
        assert os.path.exists(path)
        tokens = []
        with open(path, 'r') as f:
            token = 0
            for line in f:
                words = line.split() + ['</s>']
                tokens += map(self.dictionary.__getitem__, words)
        strip_len = len(tokens) // bsz
        usable = strip_len * bsz
        data = np.asarray(tokens[:usable]).reshape(bsz, strip_len).transpose()

        for b in range(strip_len // seq_len):
            source = torch.LongTensor(data[(b*seq_len):((b+1)*seq_len), :]) 
            target = torch.LongTensor(data[(b*seq_len)+1:((b+1)*seq_len)+1, :]) 
            if use_cuda:
                source = source.cuda()
                target = target.cuda()
            source = Variable(source, volatile=evaluation)
            target = Variable(target)
            yield (source, target)
