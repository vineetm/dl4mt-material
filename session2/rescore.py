import argparse, logging, codecs
from translation_model import TranslationModel
from collections import OrderedDict
import commands
from nltk.corpus import stopwords

stopw = set(stopwords.words('english'))

SVM_RANK_DATA='.svm_rank.data'

Q1 = 0
A1 = 1
Q2 = 2

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
    fw_gold.close()

    fw_hyp = codecs.open('temp.hyp.txt', 'w', 'utf-8')
    fw_hyp.write(prediction)
    fw_hyp.close()

    cmd_bleu = "./multi-bleu.perl temp.gold.txt < temp.hyp.txt | tail -1 | cut -d ',' -f 1 | cut -d '=' -f 2"
    logging.info('Executing cmd:%s'% cmd_bleu)
    (status, output) = commands.getstatusoutput(cmd_bleu)
    logging.info('Status: %s Output:%s'%(status, output))

    try:
      bleu = float(output)
    except ValueError:
      bleu = 0.0
    return bleu


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


def get_fraction_keywords_match(gold, hyp):
    rem_kw = gold - hyp
    return len(gold) - len(rem_kw)


def generate_features(src_line, translation):
    features = []
    translation_kw = set(translation.split()) - stopw

    parts = src_line.split('END')
    q2_bleu = get_bleu_score(parts[Q2], translation)
    features.append(q2_bleu)

    q2_keywords = set(parts[Q2].split()) - stopw
    features.append(get_fraction_keywords_match(q2_keywords, translation_kw))

    return features

def write_train_data(fw, orig_id, train_id, translations, scores, scores_index, src_line):
    fw.write('# Original ID: %d Source:%s\n'% (orig_id, src_line.strip()))

    for num, id in enumerate(scores_index):
        features = generate_features(src_line, translations[id])
        features_kv = ['%d:%d'%(index+1, feature) for index,feature in enumerate(features)]
        rank = len(scores_index) - num
        fw.write('#Translation: %s BLEU:%f\n'% (translations[id].strip(), scores[scores_index]))
        fw.write('%d qid:%d %s\n'%(rank, train_id, ' '.join(features_kv)))


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

    fw = codecs.open(args.model + SVM_RANK_DATA, 'w', 'utf-8')

    tm = TranslationModel(args.model)
    num_all_zeros = 0

    train_id = 0
    for sentence_idx, (src_line, src_line_nounk, gold_line) in enumerate(zip(src_lines, src_lines_nounk, gold_lines)):
        translations = tm.translate(src_line, k=args.num)
        logging.info('Source_line: %s'% src_line_nounk)
        logging.info('Gold_line: %s' % gold_line)

        unk_map = build_unk_map(src_line, src_line_nounk)
        logging.info('UNK_map: %s'% str(unk_map))

        scores = []
        translations_nounk = []
        for idx, translation in enumerate(translations):
            translation_nounk = replace_symbols(translation[1], unk_map)
            translations_nounk.append(translation_nounk)
            bleu_nounk = get_bleu_score(gold_line, translation_nounk)
            scores.append(bleu_nounk)
            #logging.info('Tr:%d ::%s BLEU:%s'%(idx, translation_nounk, bleu_nounk))

        if sum(scores) == 0.0:
            num_all_zeros += 1
            continue

        scores_index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        write_train_data(fw, sentence_idx, train_id, translations_nounk, scores, scores_index, src_line_nounk)
        train_id += 0

        for index in scores_index:
            logging.info('Tr: %d Text:%s Pr:%f BLEU:%f'%(index, translations[index][1],
                                                              translations[index][0], scores[index]))
    logging.info('Num all zeros: %d'%num_all_zeros)

if __name__ == '__main__':
    main()
