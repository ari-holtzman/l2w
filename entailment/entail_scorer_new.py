import code
import sys
import pickle
import itertools

import numpy as np

import torch
import torch.autograd as autograd

class EntailmentScorer():
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f).cuda()
        self.model.eval()
        self.normalize = torch.nn.LogSoftmax()
        self.delimiter = 1

    def int_to_tensor(self, tokens):
        return autograd.Variable(torch.LongTensor(tokens)).cuda().view(-1, 1)

    def pad(self, tensor, length):
            return autograd.Variable(torch.cat([tensor, tensor.new( length - tensor.size(0), *tensor.size()[1:]).zero_()])).cuda().view(-1, 1)

    def sent_split(self, tokens, terminals):
        return [list(y) for x, y in itertools.groupby(tokens,
                lambda z: z in terminals) if not x]

    def __call__(self, init_tokens, conts, current_scores, terminals,
            normalize_scores, paragraph_level_score):
        self.model.eval()
        sentence_level_score = not paragraph_level_score

        # first identity and sentence split conts

        # score latest complete sentence against sequence of all previous sentences

        latest_cands = []
        latest_indexes = {} # only for complete sentences
        contexts = [] # array of arrays of concatenated contexts

        for i, cont in enumerate(conts):
            if len(cont) > 1 and cont[-1] in terminals:
                if self.delimiter in cont[:-2]:
                    delim_index = len(cont) - 2 - cont[:-1][::-1].index(self.delimiter)
                    #delim_index = cont[:-1].rindex(self.delimiter) + 1
                    cand = cont[delim_index:]  
                    context = init_tokens + cont[:delim_index] # same init with different conts
                else:
                    cand = cont
                    context = init_tokens
                latest_indexes[i] = len(latest_cands)
                latest_cands.append(cand)
                contexts.append(context)

        if not latest_cands:
            return current_scores

        cont_contexts_indexes = [] # start index of each example
        if sentence_level_score:
            cont_contexts_cand = [] # array of sentences
            cont_contexts_latest = []
            for i, cont in enumerate(contexts):
                cont_cands = self.sent_split(cont, terminals)
                cont_contexts_indexes.append(len(cont_contexts_cand))
                for sent in cont_cands:
                    cont_contexts_cand.append(sent)
                    cont_contexts_latest.append(latest_cands[i])
            contexts = cont_contexts_cand
            latest_cands = cont_contexts_latest

        # batch
        max_len = max([len(cont) for cont in latest_cands])
        latest_ind = [self.pad(torch.LongTensor(cont), max_len)
                      for cont in latest_cands]
        #latest_ind = [self.int_to_tensor(cont) for cont in latest_cands]
        latest_ts = torch.cat(latest_ind, 1)   
        latest_lens = self.int_to_tensor([len(cont) for cont in latest_cands]) # not actually used

        cont_max_len = max([len(cont) for cont in contexts])
        conts_ind = [self.pad(torch.LongTensor(cont), cont_max_len)
                      for cont in contexts]
        #conts_ind = [self.int_to_tensor(cont) for cont in contexts]
        conts_ts = torch.cat(conts_ind, 1)   

        # apply model
        score_tensor = self.model(conts_ts, (latest_ts, latest_lens))

        if normalize_scores:
            score_tensor = self.normalize(score_tensor)[:,0]
        else:
            score_tensor = score_tensor[:,0]
        scores = score_tensor.data.cpu().numpy()

        # reconstruct new scores
        new_scores = []
        for i, _ in enumerate(conts):
            if i in latest_indexes:
                if sentence_level_score:
                  start = cont_contexts_indexes[latest_indexes[i]]
                  if latest_indexes[i] + 1 < len(cont_contexts_indexes):
                      end = cont_contexts_indexes[latest_indexes[i]+1]
                  else:
                      end = len(scores)
                  score = min(scores[start:end])
                else:    
                    score = scores[latest_indexes[i]]
            else:
                score = current_scores[i]
            new_scores.append(score)

        return new_scores

