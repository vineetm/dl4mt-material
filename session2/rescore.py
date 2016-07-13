import argparse, logging, codecs
from translation_model import TranslationModel
from collections import OrderedDict
import commands


SVM_RANK_MODEL_SUFFIX='.svm_rank.model'


'''
1) Translate Source lines as per model
2) Replace UNK symbols as per .nounk file
3) Re-order translations based on BLEU score with gold
4) Write train data for svm-rank, train svm rank
5) In test only model, make predictions
'''

def get_bleu_score(gold, prediction):
    fw_gold = codecs.open('temp.gold.txt', 'w', 'utf-8')
    fw_gold.write(gold)

    fw_hyp = codecs.open('temp.hyp.txt', 'w', 'utf-8')
    fw_hyp.write(prediction)

    cmd_bleu = 'multi-bleu.perl temp.gold.txt < temp.hyp.txt'

    logging.info('Executing cmd:%s'% cmd_bleu)
    (status, output) = commands.getstatusoutput(cmd_bleu)
    logging.info(output)






def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Source Data')
    parser.add_argument('gold', help='Gold Translations')
    parser.add_argument('model', help='Trained Model')
    parser.add_argument('--suffix', help='Original file without any symbols', default='.nounk')
    parser.add_argument('--test', dest='test', action='store_true', help='Only test')
    parser.add_argument('--num', default=20, type=int)
    args = parser.parse_args()
    return args


def build_unk_map(source, source_nounk):
    tokens = source.split()
    tokens_nounk = source_nounk.split()
    assert len(tokens) == len(tokens_nounk)

    unk_map = OrderedDict()
    for token, token_nounk in zip(tokens, tokens_nounk):
        if token != token_nounk:
            unk_map[token] = token_nounk

    return unk_map

def replace_symbols(line, unk_map):
    tokens = line.split()
    new_tokens = []

    for token in tokens:
        if token in unk_map:
            new_tokens.append(unk_map[token])
        else:
            new_tokens.append(token)
    return ' '.join(new_tokens)

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
        logging.info('Source_line: %s'% src_line_nounk)
        logging.info('Gold_line: %s' % gold_line)

        unk_map = build_unk_map(src_line, src_line_nounk)
        logging.info('UNK_map: %s'% str(unk_map))

        for idx, translation in enumerate(translations):
            translation_nounk = replace_symbols(translation[1], unk_map)
            bleu_nounk = bleu([gold_line.split()], translation_nounk.split(), (1,1,1,1))
            logging.info('Tr:%d ::%s BLEU:%f'%(idx, translation_nounk, bleu_nounk))


if __name__ == '__main__':
    main()