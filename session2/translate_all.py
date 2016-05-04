import argparse, logging, codecs
from translation_model import TranslationModel
from nltk.translate.bleu_score import sentence_bleu as bleu

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='trained model')
    parser.add_argument('input', help='input sentences')
    parser.add_argument('out', help='translated sentences')
    parser.add_argument('--all', dest='all', action='store_true', help='Check all translations')
    args = parser.parse_args()
    return args


def find_best_translation(input_line, results):
    best_bleu_score = 0.0
    best_index = 0

    for index, result in enumerate(results):
        if len(result[1].split()) == 0:
            continue
        bleu_score = bleu([input_line.split()], result[1].split(), weights=(1.0,))
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            best_index = index

    return best_index


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    tm = TranslationModel(args.model)
    fw_out = codecs.open(args.out, 'w', 'utf-8')

    line_num = 0
    for input_line in codecs.open(args.input, 'r', 'utf-8'):
        results = tm.translate(input_line.strip(), k = 20)
        if args.all:
            index = find_best_translation(input_line, results)
        else:
            index = 0

        logging.info('Line:%d best_index:%d'% (line_num, index))
        fw_out.write(results[index][1] + '\n')
        line_num += 1
    fw_out.close()

if __name__ == '__main__':
    main()