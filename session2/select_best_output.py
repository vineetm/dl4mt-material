import argparse, logging, codecs
from nltk.translate.bleu_score import sentence_bleu as bleu


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out1', help = 'Output 1')
    parser.add_argument('out2', help = 'Output 2')
    parser.add_argument('input', help = 'Input')
    parser.add_argument('output', help='Selected Output')
    args = parser.parse_args()
    return args


def get_score(candidate, input):
    return bleu([input().split()], candidate, weights=(1.0,))


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    out1_lines = codecs.open(args.out1, 'r', 'utf-8').readlines()
    out2_lines = codecs.open(args.out2, 'r', 'utf-8').readlines()

    picked_num1 = 0
    picked_num2 = 0

    input_lines = codecs.open(args.input, 'r').readlines()

    fw = codecs.open(args.output, 'w', 'utf-8')
    for index, (out1, out2, input) in enumerate(zip(out1_lines, out2_lines, input_lines)):
        q2 = input.split('END')[2]
        score_1 = get_score(q2, out1)
        score_2 = get_score(q2, out2)
        logging.info('Index:%d Bleu1: %f Bleu2: %f'% (index, score_1, score_2))

        if score_1 > score_2:
            picked_num1 += 1
            fw.write(out1.strip() + '\n')
        else:
            picked_num2 += 1
            fw.write(out2.strip() + '\n')

    logging.info('Picked1: %d Picked:%d'% (picked_num1, picked_num2))

if __name__ == '__main__':
    main()
