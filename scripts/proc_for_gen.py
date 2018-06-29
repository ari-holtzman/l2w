import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='tsv file with data')
parser.add_argument('out_path', type=str, help='txt file to write to')
args = parser.parse_args()

df = pd.read_csv(args.data_path, sep='\t')
with open(args.out_path, 'w') as out_file:
    for partial_rev in df['first']:
        out_file.write('%s\n' % partial_rev.replace('<sep>', '</s>'))
