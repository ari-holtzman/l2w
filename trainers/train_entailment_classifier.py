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
sys.path.insert(0, os.path.join(path, '../entailment/'))
sys.path.insert(0, os.path.join(path, '../utils/'))

#from entailment.cnn_entailment_classifier import CNNEntailmentClassifier
from decomposable_attention_classifier import DecomposableAttentionClassifier


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='path to data directory')
parser.add_argument('--save_to', type=str, default='', help='path to save model')
parser.add_argument('--dic', type=str, default='dic.pickle',
                    help='lm dic to use as vocabulary')
# Run Parameters
parser.add_argument('--batch_size',
                    type=int, 
                    default=32,
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
                    default=0.05,
                    help='learning rate for optimizer')
parser.add_argument('--adam',
                    action='store_true',
                    help='train with adam optimizer')
parser.add_argument('--adagrad',
                    action='store_true',
                    help='train with adagrad optimizer')
parser.add_argument('--adagrad_init',
                    type=float, 
                    default=0.1,
                    help='Adagrad accumulator initializer')
#parser.add_argument('--weight_decay',
#                    type=float, 
#                    default=5e-5,
#                    help='Adagrad accumulator initializer')
parser.add_argument('--train_prefixes',
                    action='store_true',
                    help='train on all ending prefixes')
# Model Parameters
parser.add_argument('--embedding_dim',
                    type=int, 
                    default=300,
                    help='length of word embedding vectors')
parser.add_argument('--hidden_dim',
                    type=int, 
                    default=200,
                    help='length of hidden state vectors')
parser.add_argument('--filter_size',
                    type=int, 
                    default=3,
                    help='convolutional filter size')
parser.add_argument('--dropout_rate',
                    type=float, 
                    default=0.2,
                    help='dropout rate')
parser.add_argument('--fix_embeddings',
                    action='store_true',
                    help='fix word embeddings')
parser.add_argument('--conv_layer',
                    action='store_true',
                    help='conv layer')
parser.add_argument('--max_pool',
                    action='store_true',
                    help='max pooling')
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

LABEL = data.Field(sequential=False, use_vocab=False,
        tensor_type=torch.LongTensor, postprocessing=data.Pipeline(lambda x, y: int(x)))

print('Reading the data')
train, valid = data.TabularDataset.splits(
    path=args.data_dir, 
    train = 'train.txt', validation='valid.txt',
    format='tsv',
    fields=[
        ('context', TEXT),
        ('candidate', TEXT),
        ('label', LABEL),
        ])

# Read in the LM dictionary.
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

train_iter = data.Iterator(dataset=train, batch_size=args.batch_size, sort_key=lambda x: len(x.context), sort=True, repeat=False)
valid_iter = data.Iterator(dataset=valid, batch_size=args.batch_size, sort_key=lambda x: len(x.context), sort=True, repeat=False)

print('Initializing the model')

model = DecomposableAttentionClassifier(len(TEXT.vocab), 3, args.embedding_dim,
          args.hidden_dim, args.dropout_rate,
          embed_mat=TEXT.vocab.vectors, fix_embeddings=args.fix_embeddings).cuda()

loss_function = nn.CrossEntropyLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters())
if args.adam:
    optimizer = optim.Adam(parameters, lr=args.lr)
elif args.adagrad:
    optimizer = optim.Adagrad(parameters, lr=args.lr)
            #weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(parameters, lr=args.lr)

early_stop = False
best_valid = 0.0
last_valid = 0.0

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

        # initialize the optimizer
        if epoch == 0 and args.adagrad:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    state['sum'] += args.adagrad_init

        def compute_loss(context, candidate, label):
            decision = model(context, candidate, itos=itos) 
            x_loss = loss_function(decision, label)
            return x_loss, decision

        loss = None
        if args.train_prefixes:
            end_seq_len = batch.candidate[0].size()[0] 
            loss = 0
            length_range = chain(range(min(10, end_seq_len)), 
                                 range(10, end_seq_len, 10))
            for i in length_range:
                cand_len = min(i+1, batch.candidate[0].size()[0])
                prefix_loss, decision = compute_loss(batch.context[0], 
                    (batch.candidate[0][:cand_len,:].view(cand_len, -1),
                    autograd.Variable(torch.ones(batch_size)*i).cuda()),
                    batch.label)
                loss += prefix_loss
        else:
            loss, decision = compute_loss(batch.context[0], 
                    batch.candidate, batch.label)

        loss.backward()
        total_loss += loss.data.item()
        optimizer.step()

        correct += np.sum(np.equal(np.argmax(decision.data.cpu().numpy(), 1),
            np.round(batch.label.data.cpu().numpy())))
        
        total += batch_size

        if b % args.valid_every == 0:
            model.eval()
            valid_iter.init_epoch()
            v_correct, v_total = 0, 0
            ones = 0
            for batch in valid_iter:
                decision = model(batch.context[0], batch.candidate) 
                predictions = np.argmax(decision.data.cpu().numpy(), 1)
                labels = np.round(batch.label.data.cpu().numpy())
                v_correct += np.sum(np.equal(predictions, labels))
                v_total += batch.label.size(0)
                ones += np.sum(predictions)
            last_valid = v_correct / v_total
            print('Valid: %f' % (v_correct / v_total))
            if v_correct / v_total > args.stop_threshold: # early stopping
                early_stop = True
                break  

    print('Train: %f' % (correct / total))
    print('Loss: %f' % (total_loss / total))
    if last_valid > best_valid:
        best_valid = last_valid
        print('Saving model')
        with open(args.save_to, 'wb') as f:
            torch.save(model, f)

