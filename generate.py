import sys, argparse, pickle, os, random
from importlib import import_module
import torch
import numpy as np

from decoder import predictors, decoders
#import adaptive_softmax.model as asmodel

path = os.path.realpath(__file__)
path = path[:path.rindex('/')+1]
sys.path.insert(0, os.path.join(path, 'lm/'))
sys.path.insert(0, os.path.join(path, 'utils/'))
sys.path.insert(0, os.path.join(path, 'entailment/'))
sys.path.insert(0, os.path.join(path, 'context/'))
sys.path.insert(0, os.path.join(path, 'word_level/'))
sys.path.insert(0, os.path.join(path, 'diction/'))
sys.path.insert(0, os.path.join(path, 'reprnn/'))
sys.path.insert(0, os.path.join(path, 'style/'))
sys.path.insert(0, os.path.join(path, 'adaptive_softmax/'))

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='input.txt',
                    help='text file containing initial strings to continue')
parser.add_argument('--skip', type=int, default=0,
                    help='number of lines to skip in data file before beginning')
parser.add_argument('--out', type=str, default='output.txt',
                    help='text file to write generations to')
parser.add_argument('--lm', type=str, default='lm.pt',
                    help='lm to use for decoding')
parser.add_argument('--dic', type=str, default='dic.pickle',
                    help='dic to use for lm')
parser.add_argument('--print', action='store_true',
                    help='whether to print output to stdout (in addition to writing it to a file)')
parser.add_argument('--both', action='store_true',
                    help='also include pure LM output')
parser.add_argument('--gen_disc_data', action='store_true',
                    help='generator discriminator data from LM')
parser.add_argument('--epochs', type=int, default=1,
                    help='how many times to go through the input file')
parser.add_argument('--verbosity', type=int, default=0,
                    help='how verbose to be during decoding')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--trip', action='store_true',
                    help='random seed')
## Learning
parser.add_argument('--learn', action='store_true')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--save_every', type=int, default=1)
## Decoding Stuff
parser.add_argument('--beam_size', type=int, default=10,
                    help='number of candidates in beam at every step')
parser.add_argument('--term', type=str, default='<end>',
                    help='what string to use as the end token')
parser.add_argument('--sep', type=str, default='</s>',
                    help='what string to use as the sentence seperator token')
parser.add_argument('--temp', type=float, default=None,
                    help='temperature, if using stochastic decoding')
parser.add_argument('--ranking_loss', action='store_true',
                    help='metaweight learning ranking loss')
parser.add_argument('--paragraph_level_score', action='store_true',
                    help='paragraph level score')
# Arbitrary Scorers
parser.add_argument('--scorers', type=str, default=None,
                    help='tsv with scorer information')
args = parser.parse_args()

np.random.seed(args.seed)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print("Load model")
with open(args.lm, 'rb') as model_file:
    model = torch.load(model_file)

model.eval()
with open(args.dic, 'rb') as dic_file:
    dictionary = pickle.load(dic_file)
predictor = predictors.RNNPredictor(model, len(dictionary), asm=True)

print("Creating scorers and decoder")
scorer_config, scorers, coefs = [], [], []
if args.scorers:
    with open(args.scorers) as scorer_file:
        for line in scorer_file:
            fields = line.strip().split('\t')
            scorer_config.append(fields)
            weight, path, classname  = fields[:3]
            weight = float(weight)
            arg_scripts = fields[3] 
            module = import_module(path)
            constructor = getattr(module, classname) 
            scorer =  constructor(arg_scripts)
            scorers.append(scorer)
            coefs.append(weight)
if not args.trip:
    decoder = decoders.BeamRerankDecoder(predictor,
                                         scorers,
                                         coefs,
                                         learn=args.learn,
                                         lr=args.lr,
                                         ranking_loss=args.ranking_loss,
                                         paragraph_level_score=args.paragraph_level_score,
                                         beam_size=args.beam_size,
                                         temperature=args.temp,
                                         terms=[dictionary['</s>']],
                                         forbidden=[dictionary['<unk>']],
                                         sep=dictionary[args.sep],
                                         verbosity=args.verbosity,
                                         dictionary=dictionary)
    lm_decoder = decoders.BeamRerankDecoder(predictor,
                                         [],
                                         [],
                                         [],
                                         [],
                                         beam_size=args.beam_size,
                                         temperature=args.temp,
                                         terms=[dictionary['</s>']],
                                         forbidden=[dictionary['<unk>']],
                                         sep=dictionary[args.sep],
                                         verbosity=args.verbosity,
                                         dictionary=dictionary)
else:
    decoder = decoders.BeamRerankDecoder(predictor,
                                         scorers,
                                         coefs,
                                         learn=args.learn,
                                         lr=args.lr,
                                         ranking_loss=args.ranking_loss,
                                         beam_size=args.beam_size,
                                         temperature=args.temp,
                                         terms=[dictionary['</s>'], dictionary['<end>']],
                                         forbidden=[dictionary['<unk>'], dictionary['<beg>']],
                                         sep=dictionary[args.sep],
                                         verbosity=args.verbosity,
                                         dictionary=dictionary)
    lm_decoder = decoders.BeamRerankDecoder(predictor,
                                         [],
                                         [],
                                         [],
                                         [],
                                         beam_size=args.beam_size,
                                         temperature=args.temp,
                                         terms=[dictionary['</s>'], dictionary['<end>']],
                                         forbidden=[dictionary['<unk>'], dictionary['<beg>']],
                                         sep=dictionary[args.sep],
                                         verbosity=args.verbosity,
                                         dictionary=dictionary)

print("Start decoding")
avg, a_n = None, 0
for i in range(args.epochs):
    with open(args.data) as data_file, open(args.out, 'w') as out_file:
        for i, line in enumerate(data_file):
            if i < args.skip:
                continue
            if args.gen_disc_data:
                init_tokens = line.strip().lower().split()
                init_tokens_ints = [dictionary[token] for token in init_tokens]

            else:
                initial, continuation = line.split('\t')[:2]
                init_tokens = initial.strip().lower().split()
                true_cont_tokens = continuation.strip().lower().split()
                true_cont_ints = [dictionary[token] for token in true_cont_tokens]
                init_tokens_ints = [dictionary[token] for token in init_tokens]
    
            if args.learn:
                diff = decoder.decode(init_tokens_ints,
                                      true_cont_ints)
                out_str = '%f\n' % diff
            elif args.gen_disc_data:
                init = ' '.join(init_tokens)
                lm_pred_tokens_ints = lm_decoder.decode(init_tokens_ints, itos=dictionary)
                lm_pred_cont_tokens = [dictionary[token]
                                        for token in lm_pred_tokens_ints[len(init_tokens):]]
                lm_cont = ' '.join(lm_pred_cont_tokens)
                out_str = '%s\n' % lm_cont
            else:
                pred_tokens_ints = decoder.decode(init_tokens_ints, itos=dictionary)
                pred_cont_tokens = [dictionary[token]
                                for token in pred_tokens_ints[len(init_tokens):]]
                init = ' '.join(init_tokens)
                cont = ' '.join(pred_cont_tokens)
                if args.both:
                    lm_pred_tokens_ints = lm_decoder.decode(init_tokens_ints, itos=dictionary)
                    lm_pred_cont_tokens = [dictionary[token]
                                            for token in lm_pred_tokens_ints[len(init_tokens):]]
                    lm_cont = ' '.join(lm_pred_cont_tokens)
                    out_str = '%s***\t%s***\t%s\n' % (init, cont, lm_cont)
                else:
                    out_str = '%s\t%s\n' % (init, cont)
    
            out_file.write(out_str)
            out_file.flush()
            if args.print:
                print(out_str, end='')
    
            # Save coeffecients if learning them
            if args.learn and (i+1) % args.save_every == 0:
                with open(args.scorers, 'w') as out:
                    if avg is None:
                        avg = decoder.model.coefs.weight.data.cpu().squeeze().clone()
                    else:
                        avg += decoder.model.coefs.weight.data.cpu().squeeze()
                    a_n += 1
                    for s, coef in enumerate(avg.numpy() / a_n):
                        scorer_config[s][0] = str(coef)
                        out.write('%s\n' % '\t'.join(scorer_config[s]))
                    print(avg / a_n)
