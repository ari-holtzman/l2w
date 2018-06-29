import argparse, os, random
import numpy as np

parser = argparse.ArgumentParser('Split text data into context and continuation')
parser.add_argument('data_dir', type=str,
                    help='directory with data splits in it')
parser.add_argument('--len_context', type=int, default=5,
                    help='number of sentences in context')
parser.add_argument('--len_continuation', type=int, default=5,
                    help='number of sentences in continuation')
parser.add_argument('--doc_level', action='store_true',
                    help='use this flag if each line in the dataset is a document')
parser.add_argument('--sent_sym', type=str, default='</s>',
                    help = 'the sentence delimeter to use')
args = parser.parse_args()

def no_rep_shuffle(l):
    l = list(zip(l, range(len(l))))
    nu_l = l[:]
    while True:
        random.shuffle(nu_l)
        for x, y in zip(l, nu_l):
            if x == y:
                break
        else:
            break
    return next(zip(*nu_l))

filenames = ['disc_train.txt', 'valid.txt', 'test.txt']
for filename in filenames:
    print(filename)
    contexts, continuations = [], []
    with open(os.path.join(args.data_dir, filename), 'r') as lines:
        if args.doc_level:
            assert(args.sent_sym is not None)
            for line in lines:
                sents = line.split(args.sent_sym)
                context = (' %s ' % args.sent_sym).join(sents[:args.len_context])
                continuation = (' %s ' % args.sent_sym).join(sents[args.len_context:(args.len_context+args.len_continuation)])
                contexts.append(context)
                continuations.append(continuation)
        else:
            context, continuation = [], []
            for line in lines:
                if (len(context) == 0) or ((len(context) % args.len_context) != 0):
                    context.append(line.strip())
                elif (len(continuation) == 0) or ((len(continuation) % args.len_continuation) != 0):
                    continuation.append(line.strip())
                else:
                    context = (' %s ' % args.sent_sym).join(context)
                    continuation = (' %s ' % args.sent_sym).join(continuation)
                    contexts.append(context)
                    continuations.append(continuation)
                    context, continuation = [], []

    with open(os.path.join(args.data_dir, filename + '.context'), 'w') as out:
        out.write('\n'.join(contexts))
    with open(os.path.join(args.data_dir, filename + '.true_continuation'), 'w') as out:
        out.write('\n'.join(continuations))
    with open(os.path.join(args.data_dir, filename + '.shuffled_continuation'), 'w') as out:
        out.write('\n'.join(no_rep_shuffle(continuations)))
