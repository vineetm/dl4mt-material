import argparse, logging, codecs
import cPickle as pkl

UNK='UNK%d'


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file with unk')
    parser.add_argument('src', help='nounk file')
    parser.add_argument('vocab', help='Vocab size', type=int)
    parser.add_argument('--dict', help='Dictionary', default='src.txt.pkl')
    args = parser.parse_args()
    return args


def build_unk_map(line, vocab_size, src_dict):
    unk_map = {}
    tokens = line.split()

    unk_num = 1
    for token in tokens:
        if token in src_dict and src_dict[token] > vocab_size:
            if token not in unk_map:
                unk_map[UNK%unk_num] = token
                unk_num += 1
    return unk_map, unk_num


def get_replaced_tokens(unk_map, line):
    tokens = line.split()

    final_tokens = []
    for token in tokens:
        if token in unk_map:
            final_tokens.append(unk_map[token])
        else:
            final_tokens.append(token)
    return final_tokens


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    input_lines = codecs.open(args.input, 'r', 'utf-8').readlines()
    src_lines = codecs.open(args.src, 'r', 'utf-8').readlines()

    fw_nounk = codecs.open(args.input + '.nounk', 'w', 'utf-8')
    src_dict = pkl.load(open(args.dict, 'r'))

    for input_line, src_line in zip(input_lines, src_lines):
        unk_map, num = build_unk_map(src_line, args.vocab, src_dict)
        tokens = get_replaced_tokens(unk_map, input_line)
        fw_nounk.write(' '.join(tokens) + '\n')

    fw_nounk.close()


if __name__ == '__main__':
    main()
