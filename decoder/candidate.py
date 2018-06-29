from math import log
class Candidate():
    def __init__(self, tokens, cont_tokens,
            next_token=None, score=0, latest_score=log(0.34), hidden=None, unused=None):
        self.tokens = tokens
        self.cont_tokens = cont_tokens
        self.next_token = next_token
        self.score = score
        self.latest_score = latest_score
        self.hidden = hidden
        self.unused = unused or set(tokens)

    def convert(self):
        return ' '.join(self.tokens)
