import numpy as np
import itertools
import operator
import functools
import torch
from math import log

from torch import nn, optim
import torch.nn.functional as F

from .candidate import Candidate
from .StaticCoefficientModel import StaticCoefficientModel

class BeamSearchDecoder():
    def __init__(self, predictor, beam_size=32, term=1, 
                temperature=None, verbosity=0):
        self.predictor = predictor
        self.beam_size = beam_size
        self.term = term
        self.temperature = temperature
        self.verbosity = verbosity

    def decode(self, tokens):
        beam = [ Candidate(tokens, []) ]
        beam = self.predictor.top_k_next(beam, self.beam_size, self.term)
        beam = beam[0]
        step = 0
        while best.score < beam[0].score:
            if self.verbosity > 0:
                print(beam[0].convert())
                if self.verbosity > 2:
                    input()
            conts, best_end_cont = self.predictor.top_k_next(
                                       beam,
                                       self.beam_size,
                                       self.term,
                                       temperature=self.temperature)
            beam = sorted([candidate for candidates in conts for candidate in candidates],
                          key=lambda c: c.score,
                          reverse=True)[:self.beam_size]
            if best_end_cont.score > best.score:
                best = best_end_cont
            step += 1
        return best.tokens

class BeamRerankDecoder():
    def __init__(self, predictor, scorers, coefs,
                learn=False, lr=0.01, rescale_scores=True,
                ranking_loss=False,
                paragraph_level_score=False,
                beam_size=32, terms=[1], temperature=None,
                verbosity=0, sep=None, stop=None, dictionary=None,
                max_len=150, forbidden=[]):
        self.predictor = predictor
        self.scorers = scorers
        self.coefs = np.asarray(coefs)
        self.beam_size = beam_size
        self.rescale_scores = rescale_scores
        self.terms = set(terms)
        self.sep = sep
        self.temperature = temperature
        self.verbosity = verbosity
        self.learn = learn
        self.stop = stop
        self.dictionary = dictionary
        self.max_len = max_len
        self.forbidden = set(forbidden)
        self.use_ranking_loss = ranking_loss
        self.paragraph_level_score = paragraph_level_score

        if self.learn:
            self.model = model = StaticCoefficientModel(len(scorers))
            model.coefs.weight.data = torch.FloatTensor(np.ones((1, len(scorers))))*0
            if ranking_loss:
                self.loss = nn.BCEWithLogitsLoss()
            self.loss = nn.MSELoss()
            self.optimizer = optim.SGD(model.parameters(), lr=lr)


    def decode(self, init_tokens, cont_tokens=None, itos=None, sample=True): 
        assert((not self.learn) or cont_tokens)
        dictionary = self.dictionary
        if self.learn:
            self.coefs = self.model.coefs.weight.data.cpu().squeeze().numpy()
        beam = [ Candidate(init_tokens, []) ]
        beam = self.predictor.top_k_next(beam, self.beam_size, temperature=self.temperature)[0]
        beam = list(filter(lambda c: c.cont_tokens[-1] != dictionary['<unk>'], beam))
        gold_cont_raw_scores = None
        best = None
        step = 2
        cont_latest_scores = log(0.34)
        while (((best is None) or (best.adjusted_score < max(map(lambda c: c.score, beam)))) and (step < self.max_len)):
            if self.verbosity > 0:
                for c in beam:
                    print(' '.join([itos[i] for i in c.cont_tokens]) + ' %f' % c.score)
                print('-'*30)
            conts = self.predictor.top_k_next(
                           beam,
                           self.beam_size,
                           temperature=self.temperature)
            if self.verbosity > 0:
                for cs in conts:
                    for c in cs:
                        print(' '.join([itos[i] for i in c.cont_tokens]) + ' %f' % c.score)
                print('*'*50)
                if self.verbosity > 2:
                    input()
            candidates, cand_cont_tokens, cand_latest_scores = [], [], []
            for cands in conts:
                for candidate in cands:
                    candidates.append(candidate)
                    cand_cont_tokens.append(candidate.cont_tokens)
                    cand_latest_scores.append(candidate.latest_score)
            if self.learn and step < len(cont_tokens):
                cand_cont_tokens.append(cont_tokens[:step])
                cand_latest_scores.append(cont_latest_scores)

            score_adjustment = np.zeros(len(candidates))
            if len(self.scorers) > 0:
                all_raw_scores = []
                for coef, scorer in zip(self.coefs, self.scorers):
                    raw_scores = np.asarray(scorer(init_tokens, cand_cont_tokens,
                        cand_latest_scores, self.terms, self.rescale_scores,
                        self.paragraph_level_score))
                    all_raw_scores.append(raw_scores)
                    score_adjustment += raw_scores[:len(candidates)] * coef
                last_raw_scores = all_raw_scores[-1]
                all_raw_scores = np.stack(all_raw_scores, axis=-1)
                if self.learn and step < len(cont_tokens):
                    gold_cont_raw_scores = all_raw_scores[-1]
                    cont_latest_scores = gold_cont_raw_scores[-1]
                    
            for i, candidate in enumerate(candidates):
                candidate.adjusted_score = candidate.score + score_adjustment[i]
                if len(self.scorers) > 0:
                    candidate.latest_score = last_raw_scores[i]
                    candidate.raw_scores = all_raw_scores[i]

            candidates = sorted(candidates, key=lambda c: c.adjusted_score, reverse=True)
            filtered_candidates = list(filter(lambda c: c.cont_tokens[-1] not in (self.forbidden | set([dictionary['<end>']])), candidates))
            if sample and len(filtered_candidates) > self.beam_size:
                p = np.asarray(list(map(lambda c: c.adjusted_score, filtered_candidates)))
                p = np.exp(p / 1.8)
                p /= p.sum()
                beam = np.random.choice(filtered_candidates, size=self.beam_size, replace=True, p=p)
            else:
                beam = [cand for cand in itertools.islice(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, candidates), self.beam_size)]

            for candidate in filter(lambda c: c.cont_tokens.count(1) == 5 and c.cont_tokens[-1] in self.terms, candidates):
                if best is None or candidate.adjusted_score > best.adjusted_score:
                    best = candidate
            step += 1
        best = best or beam[0]

        if self.learn:
            self.model.zero_grad()
            truth_lm_scores = self.predictor.logprobs([init_tokens + cont_tokens]).squeeze().cpu().data.numpy()
            truth_lm_score = sum([truth_lm_scores[i+len(init_tokens)-1, cont_tokens[i]] for i in range(len(cont_tokens))])
            lm_scores = torch.Tensor([truth_lm_score, beam[0].score])
            training_pair = [gold_cont_raw_scores, beam[0].raw_scores]
            training_pair = torch.Tensor(np.stack(training_pair))
            pair_scores = self.model(training_pair).squeeze()
            pair_scores = pair_scores + lm_scores
            pred = pair_scores[0] - pair_scores[1]
            diff = pred.data[0]
            if self.use_ranking_loss:
              loss = self.loss(pred, torch.FloatTensor([1]))
            else:
              loss = self.loss(pred, torch.FloatTensor([0]))
            loss.backward()
            self.optimizer.step()
            self.model.coefs.weight.data = self.model.coefs.weight.data.clamp(min=0)

        return best.tokens if not self.learn else diff
