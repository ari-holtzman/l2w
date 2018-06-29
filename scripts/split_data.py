import argparse, os
import numpy as np

parser = argparse.ArgumentParser('Split text data into train, valid, test')
parser.add_argument('data', help='text file to split')
parser.add_argument('out_dir', help='directory to write output to')
parser.add_argument('--valid_frac', type=float, default=0.05,
                    help='what fraction of the data to use for validation')
parser.add_argument('--test_frac', type=float, default=0.05,
                    help='what fraction of the data to use for testing')
parser.add_argument('--disc_train_frac', type=float, default=0.2,
                    help='what fraction of the data to use for discriminator training')
parser.add_argument('--no_disc_train', action='store_true',
                    help="don't include a disc train section")
args = parser.parse_args()

n_lines = 0
with open(args.data, 'r') as lines:
    for line in lines:
        n_lines += 1
    lines.seek(0)
    valid_limit, test_limit, disc_train_limit = np.ceil(n_lines * args.valid_frac), np.ceil(n_lines * args.test_frac), np.ceil(n_lines * args.disc_train_frac)
    valid_done, test_done, disc_train_done = False, False, False
    if args.no_disc_train:
        disc_train_done = True
    line_buff = []
    for line in lines:
        line_buff.append(line.strip())
        if not valid_done:
            if len(line_buff) == valid_limit:
                with open(os.path.join(args.out_dir, 'valid.txt'), 'w') as valid_file:
                    valid_file.write('\n'.join(line_buff))
                line_buff = []
                valid_done = True
        elif not test_done:
            if len(line_buff) == test_limit:
                with open(os.path.join(args.out_dir, 'test.txt'), 'w') as test_file:
                    test_file.write('\n'.join(line_buff))
                line_buff = []
                test_done = True
        elif not disc_train_done:
            if len(line_buff) == disc_train_limit:
                with open(os.path.join(args.out_dir, 'disc_train.txt'), 'w') as test_file:
                    test_file.write('\n'.join(line_buff))
                line_buff = []
                disc_train_done = True
with open(os.path.join(args.out_dir, 'train.txt'), 'w') as train_file:
    train_file.write('\n'.join(line_buff))
