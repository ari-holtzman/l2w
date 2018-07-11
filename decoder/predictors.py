import torch
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from .candidate import Candidate

class RNNPredictor():
    def __init__(self, model, vocab_size, use_cuda=True, asm=False):
        if use_cuda:
            model = model.cuda()
        self.model = model
        self.vocab_size = vocab_size
        self.asm = asm

    def top_k_next(self, beam, k, temperature=None, mask=None, vc=None):
        n = len(beam)
        assert(n > 0)
        if beam[0].hidden is None:
            hidden = self.model.init_hidden(len(beam))
            tokens = [ cand.tokens for cand in beam ]
        else:
            hidden = torch.cat([ cand.hidden for cand in beam ], dim=1)
            tokens = [ [ cand.next_token ] for cand in beam ]
        source = Variable(torch.LongTensor(tokens).t().cuda())
        if self.asm:
            self.model.full = True
        output, hidden = self.model(source, hidden)
        if self.asm:
            output = output.view(source.size(0), source.size(1), self.vocab_size)
        else:
            output = output.data

        temp =  output[-1,:,:]
        ps = F.log_softmax(temp, dim=temp.dim()-1).data
        if temperature is None:
            _, idxs = ps.topk(k)
        else:
            idxs = ps.div(temperature).exp().multinomial(k)
        idxs_np = idxs.cpu().numpy()
        beam_cands, best_ender = [], None
        part = 0
        for i in range(n):
            ith_cands = []
            base_score = beam[i].score
            cur_hidden = hidden[:, i, :].unsqueeze(1).clone()
            for j in range(k):
                next_word = int(idxs_np[i, j])
                nu_score = base_score + ps[i, next_word]
                nu_cand  = Candidate(beam[i].tokens + [ next_word ],
                                     beam[i].cont_tokens + [ next_word ],
                                     next_word,
                                     score=nu_score,
                                     latest_score = beam[i].latest_score,
                                     hidden = cur_hidden)
                ith_cands.append(nu_cand)
            ith_cands.append(nu_cand)
            beam_cands.append(ith_cands)

        return beam_cands

    def logprobs(self, seqs):
        hidden = self.model.init_hidden(len(seqs))
        source = Variable(torch.LongTensor(seqs).t().cuda())
        if self.asm:
            self.model.full = True
        output, hidden = self.model(source, hidden)
        if self.asm:
            output = output.view(source.size(0), source.size(1), self.vocab_size)
        temp = output.transpose(0, 2)
        output = (F.log_softmax(temp, dim=temp.dim()-1)).transpose(0, 2)
        return output

    def top_next(self, cand, term):
        if cand.hidden is None:
            hidden = self.model.init_hidden(1)
            tokens = [ cand.tokens ]
        else:
            hidden = cand.hidden
            tokens = [ [ cand.next_token ] ]
        source = Variable(torch.LongTensor(tokens).t().cuda())
        if self.asm:
            self.model.full = True
        output, hidden = self.model(source, hidden)
        if self.asm:
            output.view(source.size(0), source.size(1), self.vocab_size)
        else:
            output = output.data

        ps = output[-1,:,:].exp()
        ps = ps.div(ps.sum(1).repeat(self.vocab_size, 1).t())
        vals, idxs = ps.max(dim=1)
        next_word = int(idxs[0])
        nu_cand = Candidate(cand.tokens + [ next_word ],
                  cand.cont_tokens + [ next_word ],
                  next_word,
                  score = cand.score + float(vals[0]),
                  latest_score = cand.latest_score,
                  hidden = hidden)

        return nu_cand, next_word == term

    def top_k_next_spec(self, beam, k, term, r=0, temp=None):
        n = len(beam)
        assert(n > 0)
        if beam[0].hidden is None:
            hidden = self.model.init_hidden(len(beam))
            tokens = [ cand.tokens for cand in beam ]
        else:
            hidden = torch.cat([ cand.hidden for cand in beam ], dim=1)
            tokens = [ [ cand.next_token ] for cand in beam ]
        source = Variable(torch.LongTensor(tokens).t().cuda())
        if self.asm:
            self.model.full = True
        output, hidden = self.model(source, hidden)
        if self.asm:
            output = output.view(source.size(0), source.size(1), self.vocab_size)
        else:
            output = output.data

        temp = output[-1,:,:] 
        ps = F.log_softmax(temp, dim=temp.dim()-1).data
        for i, cand in enumerate(beam):
            for w in cand.unused:
                ps[i,w] += r
        if temp is None:
            _, idxs = ps.topk(k+1)
        else:
            idxs = ps.div(temp).exp().multinomial(k+1)
        idxs_np = idxs.cpu().numpy()
        beam_cands, best_ender = [], None
        part = 0
        for i in range(n):
            ith_cands = []
            base_score = beam[i].score
            cur_hidden = hidden[:, i, :].unsqueeze(1).clone()
            for j in range(k):
                next_word = int(idxs_np[i, j])
                if next_word in [0, 33]:
                    continue
                nu_score = base_score + ps[i, next_word]
                nu_unused = set(beam[i].unused)
                if next_word in nu_unused:
                    nu_unused.remove(next_word)
                nu_cand  = Candidate(beam[i].tokens + [ next_word ],
                                     beam[i].cont_tokens + [ next_word ],
                                     next_word,
                                     score=nu_score,
                                     latest_score = beam[i].latest_score,
                                     hidden = cur_hidden,
                                     unused = nu_unused)
                ith_cands.append(nu_cand)
            next_word = term
            nu_score = base_score + ps[i, next_word]
            nu_cand  = Candidate(beam[i].tokens + [ next_word ],
                                 beam[i].cont_tokens + [ next_word ],
                                 next_word,
                                 score=nu_score,
                                 latest_score = beam[i].latest_score,
                                 hidden = cur_hidden)
            ith_cands.append(nu_cand)
            beam_cands.append(ith_cands)

        return beam_cands


