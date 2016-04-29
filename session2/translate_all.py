import argparse, logging, codecs
from translation_model import TranslationModel


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='trained model')
    parser.add_argument('input', help='input sentences')
    parser.add_argument('out', help='translated sentences')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    tm = TranslationModel(args.model)
    fw_out = codecs.open(args.out, 'w', 'utf-8')
    for input_line in codecs.open(args.input, 'r', 'utf-8'):
        results = tm.translate(input_line.strip())
        fw_out.write(results[0][1] + '\n')

    fw_out.close()

if __name__ == '__main__':
    main()