"""
Cleans up output for the purposes of automatic eval (not human eval). Just
removes special tokens.
"""


import argparse


def main():
    # settings
    remove = ['<beg>', '</s>', '<end>']

    # setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'in_path',
        type=str,
        help='path to input text file (should be output of generation system)')
    parser.add_argument(
        'out_path',
        type=str,
        help='path to output text file (where cleaned text will be written)')
    args = parser.parse_args()

    # read
    with open(args.in_path, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines()]
    print('INFO: Read from "{}"'.format(args.in_path))

    # clean
    clean_lines = []
    for line in raw_lines:
        tkns = [tkn for tkn in line.split() if tkn not in remove]
        clean_lines.append(' '.join(tkns))

    # write
    with open(args.out_path, 'w') as f:
        for line in clean_lines:
            f.write('{}\n'.format(line))
    print('INFO: Wrote to "{}"'.format(args.out_path))


if __name__ == '__main__':
    main()
