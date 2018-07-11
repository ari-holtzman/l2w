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
        self.normalize = torch.nn.functional.log_softmax
        self.delimiter = 1

    def int_to_tensor(self, tokens):
        return autograd.Variable(torch.LongTensor(tokens)).cuda().view(-1, 1)

    def pad(self, tensor, length):
            return autograd.Variable(torch.cat([tensor, tensor.new( length - tensor.size(0), *tensor.size()[1:]).zero_()])).cuda().view(-1, 1)

    def sent_split(self, tokens):
        return [list(y) for x, y in itertools.groupby(tokens,
                lambda z: z == self.delimiter) if not x]

    def __call__(self, init_tokens, conts, current_scores, terminals,
            normalize_scores, paragraph_level_score):
        self.model.eval()
        # first identity and sentence split conts
        latest_cands = []
        latest_indexes = {}
        cont_contexts_cand = [] # array of arrays
        cont_contexts_latest = []
        cont_contexts_indexes = {}
        #TODO we should only be doing the computation for items where the
        # continuation ends in a terminal.
        for i, cont in enumerate(conts):
            cont_cands = self.sent_split(cont)
            if (len(cont_cands) > 0):
                latest_indexes[i] = len(latest_cands)
                latest_cands.append(cont_cands[-1])
                if len(cont_cands) > 1:
                    cont_contexts_indexes[i] = len(cont_contexts_cand)
                    cont_contexts_cand.append(cont_cands[:-1])
                    cont_contexts_latest.append(cont_cands[-1])

        init_sentences = self.sent_split(init_tokens[1:])
        if not (latest_cands and init_sentences):
            return current_scores

        # batched latests
        max_len = max([len(cont) for cont in latest_cands])
        conts_ind = [self.pad(torch.LongTensor(cont), max_len)
                     for cont in latest_cands]
        #conts_ind = [self.int_to_tensor(cont) for cont in latest_cands]
        #print(conts_ind)
        conts_ts = torch.cat(conts_ind, 1)
        cont_lens = self.int_to_tensor([len(cont) for cont in conts])

        # each init vs all latest cont
        init_scores = []
        for init_sent in init_sentences:
            init_ind = self.int_to_tensor(init_sent)
            score_tensor = self.model(init_ind, (conts_ts, cont_lens))

            if normalize_scores:
                score_tensor = self.normalize(score_tensor, dim=score_tensor.dim()-1)[:,0]
            else:
                score_tensor = score_tensor[:,0]
            score = score_tensor.data.cpu().numpy()
            init_scores.append(score)

        init_scores = np.min(init_scores, 0) # np.stack(init_scores, 0)

        # all prev cont vs each latest cont
        prev_cont_scores = []
        for i, latest_cont in enumerate(cont_contexts_latest):
            latest_ind = self.int_to_tensor(latest_cont)
            latest_len = self.int_to_tensor([len(latest_cont)])
            # batch prev cont
            max_len = max([len(cont) for cont in cont_contexts_cand[i]])
            conts_ind = [self.pad(torch.LongTensor(cont), max_len)
                         for cont in cont_contexts_cand[i]]
            conts_ts = torch.cat(conts_ind, 1)
            score_tensor = self.model(conts_ts, (latest_ind, latest_len))

            if normalize_scores:
                score_tensor = self.normalize(score_tensor, dim=score_tensor.dim()-1)[:,0]
            else:
                score_tensor = score_tensor[:,0]
            score = score_tensor.data.cpu().numpy()
            min_score = np.min(score)
            prev_cont_scores.append(min_score)

        new_scores = []
        for i in range(len(conts)):
            if i in latest_indexes:
                score = init_scores[latest_indexes[i]]
                if i in cont_contexts_indexes:
                    score = min(score, prev_cont_scores[cont_contexts_indexes[i]])
            else:
                score = current_scores[i]
            new_scores.append(score)

        return new_scores

