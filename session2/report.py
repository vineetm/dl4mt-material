import argparse, codecs, logging, nltk
import unicodecsv as csv
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='Source file')
    parser.add_argument('target', help='Translated data')
    parser.add_argument('gold', help='Gold output file')
    parser.add_argument('model', help='Model Name')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    f = codecs.open('report-%s.csv'% args.model, 'w')
    csv_f = csv.writer(f, delimiter=',', encoding='utf-8')

    src_lines = codecs.open(args.src, 'r', 'utf-8').readlines()
    src_lines_nounk = codecs.open(args.src + '.nounk', 'r', 'utf-8').readlines()

    target_lines = codecs.open(args.target, 'r', 'utf-8').readlines()
    target_lines_nounk = codecs.open(args.target + '.nounk', 'r', 'utf-8').readlines()

    gold_lines = codecs.open(args.gold, 'r', 'utf-8').readlines()
    gold_lines_nounk = codecs.open(args.gold + '.nounk', 'r', 'utf-8').readlines()

    data = ['Src', 'Src_UNK', 'Target_UNK', 'Target', 'Gold_UNK', 'Gold', 'BLEU', 'BLUE2', 'BLEU4']
    csv_f.writerow(data)

    num_lines = len(gold_lines)
    logging.info('Num Lines: %d'% num_lines)


    references = []
    hypotheses = []
    for index in range(num_lines):
        data = []
        data.append(src_lines_nounk[index].strip())
        data.append(src_lines[index].strip())

        data.append(target_lines[index].strip())
        data.append(target_lines_nounk[index].strip())

        data.append(gold_lines[index].strip())
        data.append(gold_lines_nounk[index].strip())

        gold = gold_lines[index].strip().split()
        output = target_lines[index].strip().split()

        references.append([gold])
        hypotheses.append(output)

        if len(output) < 4:
            bleu_score = 0.0
            bleu_score_bigram = 0.0
            bleu_score_fourgram = 0.0
        else:
            bleu_score = sentence_bleu([gold], output, weights=(1.0,))
            bleu_score_bigram = sentence_bleu([gold], output, weights=(0.5, 0.5))
            bleu_score_fourgram = sentence_bleu([gold], output)

        logging.info('sentence:%d bleu:%f'%(index, bleu_score))
        data.append(str(bleu_score))
        data.append(str(bleu_score_bigram))
        data.append(str(bleu_score_fourgram))
        csv_f.writerow(data)

    final_bleu = corpus_bleu(references, hypotheses)
    logging.info('Final BLEU: %f'% final_bleu)


if __name__ == '__main__':
    main()
