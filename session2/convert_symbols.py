import argparse, logging, codecs


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('coded', help='file with coded sentences')
    parser.add_argument('input', help='Input file to decode')
    args = parser.parse_args()
    return args


def find_code_map(coded_line, decoded_line):
    codes_map = {}
    coded_tokens = coded_line.split()
    decoded_tokens = decoded_line.split()

    assert (len(coded_tokens) == len(decoded_tokens))

    for coded_token, decoded_token in zip(coded_tokens, decoded_tokens):
        if coded_token == decoded_token:
            continue
        if coded_token not in codes_map:
            codes_map[coded_token] = decoded_token
    return codes_map


def replace_codes(codes_map, orig_line):
    final_tokens = []
    tokens = orig_line.split()

    for token in tokens:
        if token in codes_map:
            replace_token = codes_map[token]
            replace_tokens = replace_token.split('_')
            if len(replace_tokens) == 1:
                final_tokens.append(replace_token)
            else:
                final_tokens.extend(replace_tokens)
        else:
            final_tokens.append(token)

    return ' '.join(final_tokens)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    coded_fr = codecs.open(args.coded, 'r', 'utf-8')
    decoded_fr = codecs.open(args.coded + '.nounk', 'r', 'utf-8')
    input_fr = codecs.open(args.input, 'r', 'utf-8')
    output_fw = codecs.open(args.input + '.nounk', 'w', 'utf-8')

    for coded_line, decoded_line, input_line in zip(coded_fr, decoded_fr, input_fr):
        codes_map = find_code_map(coded_line, decoded_line)
        output_fw.write(replace_codes(codes_map, input_line) + '\n')

    output_fw.close()

if __name__ == '__main__':
    main()
