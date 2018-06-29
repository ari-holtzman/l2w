import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class RepRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, embed_mat):
        super(RepRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds.weight.data = embed_mat
        self.word_embeds.weight.requires_grad=False
        self.hidden_dim = hidden_dim

        self.rnn1 = nn.GRU(1, hidden_dim, num_layers=1, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, context, endings, itos=None):
        # context not used.
        ends = endings[0]
        ends_ls = endings[1]

        seq_len = ends.size()[0]
        end = ends.view(seq_len, -1)
        batch_size = end.size()[1]

        if itos:
            ttexts = end.data.cpu().numpy()
            for j in range(batch_size):
                for i in range(seq_len):
                    print(itos[ttexts[i,j]], end=' ')
                print()
            input()

        text_embeds = self.word_embeds(end.view(-1))
        text_embeds = text_embeds.view(seq_len, batch_size, self.embedding_dim)
        proc_texts = []
        proc_texts.append(autograd.Variable(torch.zeros(batch_size).cuda()))
        for i in range(1, seq_len):
            a = text_embeds[i].unsqueeze(0).expand(i, batch_size,
                    self.embedding_dim).contiguous().view(-1, self.embedding_dim)
            b = text_embeds[:i,:,:].view(-1, self.embedding_dim)
            sims = F.cosine_similarity(a, b).view(i, batch_size)
            sims[sims != sims] = -1
            sims, _ = sims.max(0)
            proc_texts.append(sims)
        proc_texts = torch.stack(proc_texts).unsqueeze(2)
        init_hidden = autograd.Variable(
                          torch.FloatTensor(
                                               1, 
                                               batch_size, 
                                               self.hidden_dim
                                            ).zero_().cuda()
                                        )
        out, _ = self.rnn1(proc_texts, init_hidden)
        #input_to_fc1  = torch.stack([out[l-1, i] for i, l in enumerate(ends_ls)])
        input_to_fc1  = out[-1]
        final = self.fc1(input_to_fc1).view(-1)
        return final

