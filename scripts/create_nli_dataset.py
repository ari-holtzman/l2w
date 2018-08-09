import sys
import numpy as np
import spacy

def read_txt(fname):
    return open(fname).read().split('\n')[:-1]

assert len(sys.argv) > 2, "Arguments required."

data_path = sys.argv[1]
out_path = sys.argv[2]

nlp = spacy.load('en')

lines = read_txt(data_path)

label_strs = [line.strip().split('\t')[0] for line in lines]
labels = []
for label in label_strs:
    if label == 'neutral':
        labels.append(0)
    elif label == 'contradiction':
        labels.append(1)
    else:
        labels.append(2)

sent1 = [line.strip().split('\t')[5] for line in lines]
sent1_tok = [' '.join([tok.text.lower() for tok in nlp.tokenizer(sent)]) for sent in sent1]
sent2 = [line.strip().split('\t')[6] for line in lines]
sent2_tok = [' '.join([tok.text.lower() for tok in nlp.tokenizer(sent)]) for sent in sent2]

with open(out_path, 'w') as out_file:
    for i in range(len(sent1)):
        out_file.write(sent1_tok[i] + '\t' + sent2_tok[i] + '\t' + str(labels[i]) + '\n')
        
