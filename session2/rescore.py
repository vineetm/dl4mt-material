import argparse, logging, codecs
from translation_model import TranslationModel

SVM_RANK_MODEL_SUFFIX='.svm_rank.model'


'''
1) Translate Source lines as per model
2) Replace UNK symbols as per .nounk file
3) Re-order translations based on BLEU score with gold
4) Write train data for svm-rank, train svm rank
5) In test only model, make predictions
'''

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Source Data')
    parser.add_argument('gold', help='Gold Translations')
    parser.add_argument('model', help='Trained Model')
    parser.add_argument('--suffix', help='Original file without any symbols')
    parser.add_argument('--test', dest='test', action='store_true', help='Only test')
    parser.add_argument('--num', default=20, type=int)
    args = parser.parse_args()
    return args

'''
Source: Source with UNK Symbols
Target: Target with no UNK Symbols
'''
def main():
    args = setup_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(args)

    src_lines = codecs.open(args.source, 'r', 'utf-8').readlines()
    src_lines_nounk = codecs.open(args.source + args.suffix, 'r', 'utf-8').readlines()
    gold_lines = codecs.open(args.gold + args.suffix, 'r', 'utf-8').readlines()

    tm = TranslationModel(args.model)
    for src_line, src_line_nounk, gold_line in zip(src_lines, src_lines_nounk, gold_lines):
        translations = tm.translate(src_line, k=args.num)
        logging.info('Source_line UNK: %s'% src_line)
        logging.info('Gold_line UNK: %s' % gold_line)
        for idx, translation in enumerate(translations):
            logging.info('Tr:%d :%s Score: %f Text:%s'%(idx, translation[0], translation[1]))


if __name__ == '__main__':
    main()