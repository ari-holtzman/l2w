import argparse
import sys, time, os
import math
import torch
import torch.nn as nn

path = os.path.realpath(__file__)
path = path[:path.rindex('/')+1]
sys.path.insert(0, os.path.join(path, '../utils/'))
from doing import doing

import corpus
import model

from adaptive_softmax import AdaptiveLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--dic', type=str,
                    help='path to dictionary pickle')
parser.add_argument('--old', type=str, default=None,
                    help='old model to keep training')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, GRU)')
parser.add_argument('--emsize', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--cutoffs', nargs='+', type=int,
                    help='cutoffs for buckets in adaptive softmax')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--ar', type=float, default=0.9,
                    help='learning rate annealing rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1024, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
# Hardware
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int,  default=0,
                    help='gpu to use')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)

with doing('Loading data'):
    corpus = corpus.Corpus(args.data, args.dic)
    ntokens = len(corpus.dictionary.idx2word)
    cutoffs = args.cutoffs + [ntokens]

with doing('Constructing model'):
    if args.old is None:
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, cutoffs, args.dropout, args.tied)
    else:
        with open(args.old, 'rb') as model_file:
            model = torch.load(model_file)
    if args.cuda:
        model.cuda()
    
    criterion = AdaptiveLoss(cutoffs)

###############################################################################
# Training code
###############################################################################

# Loop over epochs.
global lr, best_val_loss
lr = args.lr
best_val_loss = None

def repackage_hidden(h):
    """Detaches hidden states from their history"""
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(split):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss, nbatches = 0, 0
    ntokens = len(corpus.dictionary.idx2word)
    hidden = model.init_hidden(args.eval_batch_size)
    for source, target in corpus.iter(split, args.eval_batch_size, args.bptt, use_cuda=args.cuda):
        model.softmax.set_target(target.data.view(-1))
        output, hidden = model(source, hidden)
        total_loss += criterion(output, target.view(-1)).data.sum()
        hidden = repackage_hidden(hidden)
        nbatches += 1
    return total_loss / nbatches


def train():
    global lr, best_val_loss
    # Turn on training mode which enables dropout.
    model.train()
    total_loss, nbatches = 0, 0
    start_time = time.time()
    ntokens = len(corpus.dictionary.idx2word)
    hidden = model.init_hidden(args.batch_size)
    for b, batch in enumerate(corpus.iter('train', args.batch_size, args.bptt, use_cuda=args.cuda)):
        model.train()
        source, target = batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        model.softmax.set_target(target.data.view(-1))
        output, hidden = model(source, hidden)
        loss = criterion(output, target.view(-1))
        loss.backward()


        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)

        total_loss += loss.data.cpu()

        if b % args.log_interval == 0 and b > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            val_loss = evaluate('valid')
            print('| epoch {:3d} | batch {:5d} | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                epoch, b, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss),
                val_loss, math.exp(val_loss)))

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr *= args.ar

            total_loss = 0
            start_time = time.time()



# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate('valid')
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate('test')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
