import argparse, os, pickle
from dictionary import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('data', help='text file to make dictionary from')
parser.add_argument('out', help='path to write dictionary pickle to')
parser.add_argument('--max_vocab', type=int, default=100000,
                    help='max_words in dictionary')
args = parser.parse_args()

assert(os.path.exists(args.data))
dic = Dictionary()
freq = {}
with open(args.data, 'r') as f:
    for line in f:
        for word in line.split():
            freq[word] = freq.get(word, 0) + 1
for _, word in sorted([(-f, w) for w, f in freq.items()])[:args.max_vocab-len(dic.idx2word)]:
    dic.add_word(word)

with open(args.out, 'wb') as out_file:
    pickle.dump(dic, out_file)
