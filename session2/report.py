import argparse, codecs, logging
import unicodecsv as csv
from nltk.align.bleu_score import bleu
import numpy as np

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', 'Source file')
    parser.add_argument('target', 'Translated data')
    parser.add_argument('gold', 'Gold output file')
    parser.add_argument('model', 'Model Name')
    args = parser.parse_args()
    return args

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    f = codecs.open('%s-%s.csv'% (args.out, args.suffix), 'w')
    csv_f = csv.writer(f, delimiter=',', encoding='utf-8')

    src_lines = codecs.open(args.src, 'r', 'utf-8').readlines()
    src_lines_nounk = codecs.open(args.src + '.nounk', 'r', 'utf-8').readlines()

    target_lines = codecs.open(args.target, 'r', 'utf-8').readlines()
    target_lines_nounk = codecs.open(args.target + '.nounk', 'r', 'utf-8').readlines()

    gold_lines = codecs.open(args.gold, 'r', 'utf-8').readlines()
    gold_lines_nounk = codecs.open(args.gold + '.nounk', 'r', 'utf-8').readlines()

    data = ['Src', 'Src_UNK', 'Target_UNK', 'Target', 'Gold_UNK', 'Gold', 'BLEU']

    num_lines = len(gold_lines)
    logging.info('Num Lines: %d'% num_lines)


    bleu_scores = []
    for index in range(num_lines):
        data = []
        data.append(src_lines_nounk[index].strip())
        data.append(src_lines[index].strip())

        data.append(target_lines[index].strip())
        data.append(target_lines_nounk[index].strip())

        data.append(gold_lines[index].strip())
        data.append(gold_lines_nounk[index].strip())

        bleu_score = bleu(target_lines[index].split(), [gold_lines[index].split()], [1])
        bleu_scores.append(bleu_score)
        data.append(str(bleu_score))
        csv_f.writerow(data)

    logging.info('Average BLEU Score: %f'% np.mean(bleu_scores))

if __name__ == '__main__':
    main()