import logging, argparse, codecs
from translation_model import TranslationModel
from collections import OrderedDict
from nltk.translate.bleu_score import sentence_bleu as bleu


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Trained Model')
    parser.add_argument('k', help='# of Translations', type=int)
    parser.add_argument('input', help='Input sequence')
    parser.add_argument('gold', help='Gold output')
    args = parser.parse_args()
    return args


def build_unk_map(sentence_symbols, sentence_orig):
    unk_map = OrderedDict()

    for token, token_symbol in zip(sentence_orig.split(), sentence_symbols.split()):
        if token != token_symbol:
            unk_map[token_symbol] = token

    return unk_map


def replace_symbols(sentence, unk_map):
    tokens = sentence.split()
    final_tokens = []

    for token in tokens:
        if token in unk_map:
            final_tokens.append(unk_map[token])
        else:
            final_tokens.append(token)

    return ' '.join(final_tokens)


def find_match(gold, translations):
    for index, translation in enumerate(translations):
        if gold == translation:
            return index
    return -1


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    tm = TranslationModel(args.model)
    input_lines_symbols = codecs.open(args.input , 'r', 'utf-8')
    input_lines = codecs.open(args.input + '.nounk',  'r', 'utf-8')

    gold_lines = codecs.open(args.gold + '.nounk', 'r', 'utf-8')

    index = 0
    found = 0
    for input_line, input_line_symbols, gold_line in zip(input_lines, input_lines_symbols, gold_lines):
        unk_map = build_unk_map(input_line_symbols, input_line)
        translations_with_scores = tm.translate(input_line_symbols, k=args.k)
        translations = [data[1] for data in translations_with_scores]

        translations_replaced = [replace_symbols(translation, unk_map) for translation in  translations]
        match_index = find_match(gold_line, translations_replaced)
        logging.info('Index: %d Match: %d'%(index, match_index))

        if match_index != -1:
            found += 1

        index += 1

    recall_k = 0.0
    recall_k += found
    recall_k /= index
    logging.info('Recall@%d: %f (%d/%d)'% (args.k, recall_k, found, index))


if __name__ == '__main__':
    main()