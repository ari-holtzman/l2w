import argparse, os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='path to tsv file containing data')
parser.add_argument('out_dir', type=str, help='directory where train, valid, and test files will go')
parser.add_argument('-v', type=int, help='Size of validation set')
parser.add_argument('-t', type=int, help='Size of test set')
args = parser.parse_args()

with open(args.data_path) as data_file:
    data = pd.read_csv(data_file, sep='\t', index_col=0)
start_test, start_train = args.v, args.v+args.t
valid, test, train = data[:start_test], data[start_test:start_train], data[start_train:]
valid_path = os.path.join(args.out_dir, 'valid.tsv')
test_path = os.path.join(args.out_dir, 'test.tsv')
train_path = os.path.join(args.out_dir, 'train.tsv')

splits = zip([valid_path, test_path, train_path], [valid, test, train])

for path, split in splits:
    with open(path, 'w') as out_file:
        split.to_csv(out_file, sep='\t')
