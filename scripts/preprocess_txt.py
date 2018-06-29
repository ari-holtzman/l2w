
import code
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import sys
from tqdm import tqdm

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

output = []
for line in tqdm(lines):
    sents = [word_tokenize(s) for s in sent_tokenize(line.strip())]

    passage = ['<beg>']
    for sent in sents:
        for word in sent:
            passage.append(word.lower())
        passage.append('</s>')
    output.append(' '.join(passage))

with open(sys.argv[2], 'w') as f:
    for line in output:
        f.write(line + '\n')
