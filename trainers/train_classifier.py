import argparse, pickle, os, sys
import numpy as np
from collections import Counter
from collections import defaultdict
from itertools import chain

from scipy.special import expit
import torch
from torchtext import data
from torchtext import vocab
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

path = os.path.realpath(__file__)
path = path[:path.rindex('/')+1]
sys.path.insert(0, os.path.join(path, '../utils/'))
sys.path.insert(0, os.path.join(path, '../word_rep/'))

from cnn_context_classifier import CNNContextClassifier
from pool_ending_classifier import PoolEndingClassifier
from reprnn import RepRNN

parser = argparse.ArgumentParser()
# Data
parser.add_argument('data_dir', type=str, help='path to data directory')
parser.add_argument('--save_to', type=str, default='', help='path to save model to')
parser.add_argument('--load_model', type=str, default='', help='existing model file to load')
parser.add_argument('--dic', type=str, default='dic.pickle',
                    help='lm dic to use as vocabulary')
# Model
parser.add_argument('--decider_type', type=str, default='cnncontext',
                    help='Decider classifier type [cnncontext, poolending, reprnn]')
# Run Parameters
parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help='number of examples to process in parallel')
parser.add_argument('--num_epochs',
                    type=int,
                    default=5,
                    help='number of times to run through training set')
parser.add_argument('--stop_threshold',
                    type=float,
                    default=0.99,
                    help='Early stopping threshold on validation accuracy')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='learning rate for optimizer')
parser.add_argument('--adam',
                    action='store_true',
                    help='train with adam optimizer')
parser.add_argument('--train_prefixes',
                    action='store_true',
                    help='train on all ending prefixes')
parser.add_argument('--valid_only',
                    action='store_true',
                    help='use only validation set')
parser.add_argument('--ranking_loss',
                    action='store_true',
                    help='train based on ranking loss')
parser.add_argument('--margin_ranking_loss',
                    action='store_true',
                    help='train based on margin ranking loss')
# Model Parameters
parser.add_argument('--embedding_dim',
                    type=int,
                    default=300,
                    help='length of word embedding vectors')
parser.add_argument('--hidden_dim',
                    type=int,
                    default=300,
                    help='length of hidden state vectors')
parser.add_argument('--filter_size',
                    type=int,
                    default=3,
                    help='convolutional filter size')
parser.add_argument('--dropout_rate',
                    type=float,
                    default=0.5,
                    help='dropout rate')
parser.add_argument('--fix_embeddings',
                    action='store_true',
                    help='fix word embeddings')
# Output Parameters
parser.add_argument('--valid_every',
                    type=int,
                    default=128,
                    help='batch interval for running validation')
parser.add_argument('-p',
                    action='store_true',
                    help='use this flag to print samples of the data')
args = parser.parse_args()

TEXT = data.Field(sequential=True, lower=True, include_lengths=True)

LABEL = data.Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor, postprocessing=data.Pipeline(lambda x, y: float(x)))

if args.valid_only:
    train_name = 'valid.tsv'
else:
    train_name = 'disc_train.tsv'

print('Reading the data')
train, valid = data.TabularDataset.splits(
    path=args.data_dir,
    train=train_name, validation='valid.tsv',
    format='tsv',
    fields=[
        ('context', TEXT),
        ('gold', TEXT),
        ('generated', TEXT),
        ])

#TEXT.build_vocab(train)
# Read in the LM dictionary.
print('Building the dictionary')
with open(args.dic, 'rb') as dic_file:
    dictionary = pickle.load(dic_file)

# Reconstruct the dictionary in torchtext.
counter = Counter({'<unk>': 0, '</s>':0})
TEXT.vocab = vocab.Vocab(counter, specials=['<unk>', '</s>'])
TEXT.vocab.itos = dictionary.idx2word
TEXT.vocab.stoi = defaultdict(vocab._default_unk_index, dictionary.word2idx)

TEXT.vocab.load_vectors('glove.6B.%dd' % args.embedding_dim)
itos = TEXT.vocab.itos if args.p else None
print('Vocab size %d' % len(TEXT.vocab))

train_iter = data.Iterator(dataset=train, batch_size=args.batch_size,
        sort_key=lambda x: len(x.context), sort=True, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=args.batch_size, sort_key=lambda x: len(x.context), sort=True, repeat=False)

print('Initializing the model')

if args.load_model != '':
    with open(args.load_model, 'rb') as f:
        model = torch.load(f).cuda()
elif args.decider_type == 'cnncontext':
    model = CNNContextClassifier(len(TEXT.vocab), args.embedding_dim,
            args.hidden_dim, args.filter_size, args.dropout_rate,
            embed_mat=TEXT.vocab.vectors,
            fix_embeddings=args.fix_embeddings).cuda()
elif args.decider_type == 'poolending':
    model = PoolEndingClassifier(len(TEXT.vocab), args.embedding_dim,
            args.hidden_dim,
            embed_mat=TEXT.vocab.vectors,
            fix_embeddings=args.fix_embeddings).cuda()
elif args.decider_type == 'reprnn':
    model = RepRNN(len(TEXT.vocab), args.embedding_dim,
            args.hidden_dim,
            embed_mat=TEXT.vocab.vectors).cuda()
else:
  assert False, 'Invalid model type.'

loss_function = nn.BCEWithLogitsLoss()
margin_loss_function = nn.MarginRankingLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters())
if args.adam:
    optimizer = optim.Adam(parameters, lr=args.lr)
else:
    optimizer = optim.SGD(parameters, lr=args.lr)

print('Evaluating model')
if args.load_model != '':
    model.eval()
    valid_iter.init_epoch()
    v_correct, v_total = 0, 0
    ones = 0
    for k, batch in enumerate(valid_iter):
        #if k % 100 == 0:
        #    print(k)
        batch_size = batch.context[0].size()[1]

        decision_negative = model(batch.context[0],
            batch.generated, itos=itos)
        decision_positive = model(batch.context[0],
            batch.gold, itos=itos)

        if args.ranking_loss or args.margin_ranking_loss:
            decision = decision_positive - decision_negative
        else:
            # Evaluate predictions on gold
            decision = decision_positive

        decis = decision.data.cpu().numpy()
        predicts = np.round(expit(decis))
        v_correct += np.sum(np.equal(predicts, np.ones(batch_size)))
        v_total += batch_size
        ones += np.sum(predicts)
    print('Valid: %f' % (v_correct / v_total))
    print('%d ones %d zeros' % (ones, v_total - ones))


early_stop = False
for epoch in range(args.num_epochs):
    if early_stop:
        break
    print('Starting epoch %d' % epoch)
    train_iter.init_epoch()
    correct, total = 0, 0
    total_loss = 0
    for b, batch in enumerate(train_iter):
        model.train()
        model.zero_grad()
        batch_size = batch.context[0].size()[1]

        def compute_loss(context, generated, gold):
            decision_negative = model(context, generated, itos=itos)
            decision_positive = model(context, gold, itos=itos)
            if args.ranking_loss or args.margin_ranking_loss:
                decision = decision_positive - decision_negative
            else:
                decision = decision_positive

            x_loss = None
            if args.ranking_loss:
                x_loss = loss_function(
                  decision_positive - decision_negative,
                  autograd.Variable(torch.ones(batch_size)).cuda())
            elif args.margin_ranking_loss:
                # 1: positive ranked higher than negative
                x_loss = margin_loss_function(
                  decision_positive, decision_negative,
                  autograd.Variable(torch.ones(batch_size)).cuda())
            else:
                x_loss = loss_function(decision_positive,
                  autograd.Variable(torch.ones(batch_size)).cuda())
                x_loss += loss_function(decision_negative,
                  autograd.Variable(torch.zeros(batch_size)).cuda())

            return x_loss, decision

        loss = None
        if args.train_prefixes:
            end_seq_len = max(batch.generated[0].size()[0],
                              batch.gold[0].size()[0])
            loss = 0
            #length_range = chain(range(min(10, end_seq_len)), 
            #                     range(10, end_seq_len, 5))
            length_range = chain(range(0, min(10, end_seq_len-1), 2), 
                                 range(10, min(end_seq_len, 30), 5),
                                 iter([end_seq_len-1]))

            for i in length_range:
                gen_len = min(i+1, batch.generated[0].size()[0])
                gold_len = min(i+1, batch.gold[0].size()[0])
                prefix_loss, decision = compute_loss(batch.context[0],
                    (batch.generated[0][:gen_len,:].view(gen_len, -1),
                    autograd.Variable(torch.ones(batch_size)*i).cuda()),
                    (batch.gold[0][:gold_len,:].view(gold_len, -1),
                    autograd.Variable(torch.ones(batch_size)*i).cuda()))
                loss += prefix_loss
        else:
            loss, decision = compute_loss(batch.context[0],
                    batch.generated, batch.gold)

        loss.backward()
        total_loss += loss.data.item()
        optimizer.step()

        correct += np.sum(np.equal(
            np.round(expit(decision.data.cpu().numpy())),
            np.ones(batch_size)))
        total += batch_size

        if b % args.valid_every == 0:
            model.eval()
            valid_iter.init_epoch()
            v_correct, v_total = 0, 0
            ones = 0
            for k, batch in enumerate(valid_iter):
                #if k % 100 == 0:
                #    print(k)
                batch_size = batch.context[0].size()[1]

                decision_negative = model(batch.context[0],
                    batch.generated, itos=itos)
                decision_positive = model(batch.context[0],
                    batch.gold, itos=itos)

                if args.ranking_loss or args.margin_ranking_loss:
                    decision = decision_positive - decision_negative
                else:
                    # Evaluate predictions on gold
                    decision = decision_positive

                decis = decision.data.cpu().numpy()
                predicts = np.round(expit(decis))
                v_correct += np.sum(np.equal(predicts, np.ones(batch_size)))
                v_total += batch_size
                ones += np.sum(predicts)
            print('Valid: %f' % (v_correct / v_total))
            print('%d ones %d zeros' % (ones, v_total - ones))
            if v_correct / v_total > args.stop_threshold: # early stopping
                early_stop = True
                break

    print('Train: %f' % (correct / total))
    print('Loss: %f' % (total_loss / total))
    print('Saving model')
    with open(args.save_to, 'wb') as f:
        torch.save(model, f)

