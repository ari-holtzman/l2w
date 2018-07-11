import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import weight_norm

# This implementation is partly based on https://github.com/libowen2121/SNLI-decomposable-attention/blob/master/models/baseline_snli.py

class DecomposableAttentionClassifier(nn.Module):

    def __init__(self, vocab_size, label_size, embedding_dim, hidden_dim, 
                 dropout_rate, init_normal_var=0.01,
                 embed_mat=None, fix_embeddings=False):
        super(DecomposableAttentionClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.label_size = label_size
        self.fix_embeddings = fix_embeddings

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if embed_mat is not None:
            self.word_embeds.weight.data = embed_mat
            if fix_embeddings:
                self.word_embeds.weight.requires_grad=False
                self.word_embeds = weight_norm(self.word_embeds, dim=1)

        # linear layer on input
        self.input_linear = nn.Linear(
            self.embedding_dim, self.hidden_dim, bias=False)  # linear transformation

        self.mlp_f = self._mlp_layers(self.hidden_dim, self.hidden_dim)
        self.mlp_g = self._mlp_layers(2 * self.hidden_dim, self.hidden_dim)
        self.mlp_h = self._mlp_layers(2 * self.hidden_dim, self.hidden_dim)

        self.final_linear = nn.Linear(
            self.hidden_dim, self.label_size, bias=True)

        #self.drop = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, init_normal_var)
                if m.bias is not None:
                    m.bias.data.normal_(0, init_normal_var)


    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=self.dropout_rate))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=self.dropout_rate))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())        
        return nn.Sequential(*mlp_layers)   # * used to unpack list


    def embed_seq(self, vec):
        vec1 = self.word_embeds(vec.transpose(0, 1).contiguous())
        #if self.fix_embeddings: #TODO test
        #  vec1 = F.normalize(vec1, p=2, dim=1)
        batch_size = vec1.size(0)
        proj_vec = self.input_linear(vec1).view(batch_size, -1, self.hidden_dim)
        return proj_vec # dim [batch_size, length, hidden_dim]


    # Input dimensions: 
    #   context: Tensor dim [seq_len, batch_size].
    #   endings: tuple of Tensors - 
    #            (dim [end_seq_len*, batch_size or num_endings] - endings, 
    #             dim [batch_size or num_endings] - batch lengths).
    #   Training: num_endings = 1; decoding: batch_size = 1.
    def forward(self, context, endings, itos=None):
        ends = endings[0]
        ends_ls = endings[1]
        seq_len1 = context.size()[0]
        context = context.view(seq_len1, -1)  
        batch_size = context.size()[1]

        seq_len2 = ends.size()[0]
        end = ends.view(seq_len2, -1)  
        end_batch_size = end.size()[1]
        decode_mode = (batch_size == 1 and end_batch_size > 1)
        multi_decode_mode = (batch_size > 1 and end_batch_size == 1)
        assert decode_mode or multi_decode_mode or batch_size == end_batch_size

        # Embed
        sent1_embed = self.embed_seq(context) # 1 x len1 x hidden
        sent2_embed = self.embed_seq(end) # batch x len2 x hidden

        # Attend 
        if decode_mode:
            f1 = self.mlp_f(sent1_embed.view(-1, self.hidden_dim)).view(-1,
                    self.hidden_dim) # length x hidden_dim
            f2 = self.mlp_f(sent2_embed.view(-1, self.hidden_dim)).view(-1,
                    self.hidden_dim) # (batch_size x len2) x hidden_size

            score1 = torch.mm(f1, torch.transpose(f2, 0, 1)).view(seq_len1, -1, seq_len2)
            # e_{ij} len1 x (batch_size x len2)

            score1 = torch.transpose(score1.contiguous(), 0, 1).contiguous()
            temp = score1.view(-1, seq_len2)
            prob1 = F.softmax(temp, dim=temp.dim()-1).view(-1, seq_len1, seq_len2) # v0.2

            score2 = torch.transpose(score1, 1, 2) # batch x len2 x len1
            score2 = score2.contiguous()
            temp = score2.view(-1, seq_len1)
            prob2 = F.softmax(temp, dim=temp.dim()-1).view(-1, seq_len2, seq_len1)

            sent1_attended = torch.bmm(prob1, sent2_embed)
            sent2_attended = torch.mm(
                    prob2.view(-1, seq_len1),
                    sent1_embed.view(seq_len1, -1)).view(
                            -1, seq_len2, self.hidden_dim)

        else:
            if multi_decode_mode:   
                sent2_embed = sent2_embed.expand(batch_size, seq_len2,
                        self.hidden_dim).contiguous()

            f1 = self.mlp_f(sent1_embed.view(-1, self.hidden_dim)).view(-1,
                    seq_len1, self.hidden_dim)
            f2 = self.mlp_f(sent2_embed.view(-1, self.hidden_dim)).view(-1,
                    seq_len2, self.hidden_dim) # batch_size x len2 x hidden_size

            score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
            # e_{ij} batch_size x len1 x len2

            temp = score1.view(-1, seq_len2)
            prob1 = F.softmax(temp, dim=temp.dim()-1).view(-1, seq_len1, seq_len2) # v0.2

            score2 = torch.transpose(score1.contiguous(), 1, 2)
            score2 = score2.contiguous() # e_{ji} batch_size x len2 x len1
            temp = score2.view(-1, seq_len1)
            prob2 = F.softmax(temp, temp.dim()-1).view(-1, seq_len2, seq_len1)

            sent1_attended = torch.bmm(prob1, sent2_embed)
            sent2_attended = torch.bmm(prob2, sent1_embed)

        # Compare
        if decode_mode:
          sent1_embed = sent1_embed.expand(end_batch_size, seq_len1,
              self.hidden_dim)

        sent1_combine = torch.cat((sent1_embed, sent1_attended), 2).view(-1,
                2*self.hidden_dim)
        sent2_combine = torch.cat((sent2_embed, sent2_attended), 2).view(-1,
                2*self.hidden_dim)

        g1 = self.mlp_g(sent1_combine).view(-1, seq_len1, self.hidden_dim)
        g2 = self.mlp_g(sent2_combine).view(-1, seq_len2, self.hidden_dim)

        # Aggregate
        sent1_output = torch.sum(g1, 1).squeeze(1) 
        sent2_output = torch.sum(g2, 1).squeeze(1) # batch_size x hidden_size

        hidden = self.mlp_h(torch.cat((sent1_output, sent2_output), 1)) # batch_size * hidden_size
        
        # Predict
        final = self.final_linear(hidden).view(-1, self.label_size)
        return final

