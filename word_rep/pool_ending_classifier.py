import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class PoolEndingClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 embed_mat=None, fix_embeddings=False):
        super(PoolEndingClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if embed_mat is not None:
            self.word_embeds.weight.data = embed_mat
            if fix_embeddings:
                self.word_embeds.weight.requires_grad=False
        
        self.fc = nn.Linear(self.embedding_dim, 1)

    def embed_seq(self, vec):
        vec1 = self.word_embeds(vec.transpose(0, 1).contiguous())
        vec_tr = vec1.transpose(1, 2).contiguous()	
        return vec_tr # dim [batch_size, embed_dim, length]

    # Input dimensions: 
    #   context: Tensor dim [seq_len, batch_size].
    #   endings: tuple of Tensors - 
    #            (dim [end_seq_len*, batch_size or num_endings] - endings, 
    #             dim [batch_size or num_endings] - batch lengths).
    #   Training: num_endings = 1; decoding: batch_size = 1.
    def forward(self, context, endings, itos=None):
        # context not used.
        ends = endings[0]
        ends_ls = endings[1]

        end_seq_len = ends.size()[0]
        end = ends.view(end_seq_len, -1)
        end_batch_size = end.size()[1]
        maxpool_end = nn.MaxPool1d(end_seq_len)

        end_embed = self.embed_seq(end)
        end_pooled = maxpool_end(end_embed).view(end_batch_size, self.embedding_dim)
        #end_pooled = torch.sum(end_conv, 2)/end_seq_len

        final = self.fc(end_pooled).view(-1)
        return final

