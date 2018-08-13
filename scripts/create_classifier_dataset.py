import sys, argparse, random, os

parser = argparse.ArgumentParser('Split text data into context and continuation')
parser.add_argument('data_dir', type=str,
                    help='directory with data splits in it')
parser.add_argument('out_dir', type=str,
                    help='directory to output data to')
parser.add_argument('--comp', type=str, required=True,
                    help='what adversarial example to compare to [lm, random, none]')
args = parser.parse_args()

def read_txt(fname):
    return open(fname).read().split('\n')

assert len(sys.argv) > 2, "Arguments required."

filenames = ['disc_train.txt', 'valid.txt', 'test.txt']
for filename in filenames:
    context = read_txt(os.path.join(args.data_dir, filename) + '.context')
    if args.comp == 'lm':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.generated_continuation')
    elif args.comp == 'random':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.shuffled_continuation')
    elif args.comp == 'none':
        comp_end = read_txt(os.path.join(args.data_dir, filename) + '.true_continuation')
    else:
        assert(False)
    true_end = read_txt(os.path.join(args.data_dir, filename) + '.true_continuation')
    
    tsv_lines = []
    randomize = False
    
    for cont, comp, true in zip(context, comp_end, true_end):
        tsv_line = cont.strip() + '\t' 
        if randomize:
            if random.random() < 0.5:
                tsv_line += comp.strip() + '\t' + true.strip() + '\t' + '1'
            else: 
                tsv_line += true.strip() + '\t' + comp.strip() + '\t' + '0'
        else:
            if args.comp == 'none':
                tsv_line += true.strip()
            else:
                tsv_line += comp.strip() + '\t' + true.strip() + '\t' + '1'
        tsv_lines.append(tsv_line)
    
    with open(os.path.join(args.out_dir, filename[:-4] + '.tsv'), 'w') as out:
        out.write('\n'.join(tsv_lines))
    
