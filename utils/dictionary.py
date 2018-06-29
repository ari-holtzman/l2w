class Dictionary(object):
    def __init__(self):
        self.unk_tok = '<unk>'
        self.sent_tok = '</s>'
        self.word2idx = { self.unk_tok : 0, self.sent_tok : 1 }
        self.idx2word = [ self.unk_tok, self.sent_tok ]
        self.unk_idx = self.word2idx[self.unk_tok]
        self.sent_idx = self.word2idx[self.sent_tok]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def __getitem__(self, key):
        if type(key) == str:
            return self.word2idx.get(key, self.unk_idx)
        elif type(key) == int:
            return self.idx2word[key]
        else:
            raise KeyError
