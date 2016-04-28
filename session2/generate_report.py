import argparse, logging, csv
from translation_model import TranslationModel


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='trained model')
    parser.add_argument('input', help='input text file')
    parser.add_argument('gold', help='gold standard for input file')
    parser.add_argument('--out', help='report', default='report')
    parser.add_argument('--suffix', help='suffix to add to report', default='test')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    tm = TranslationModel(args.model)
    f = open('%s-%s.csv'% (args.out, args.suffix), 'w')
    csv_f = csv.writer(f, delimiter=',')

    data = ['Src', 'Target', 'Gold Standard']
    csv_f.writerow(data)
    input_lines = open(args.input, 'r').readlines()
    gold_lines = open(args.gold, 'r').readlines()
    for input_line, gold_line in zip(input_lines, gold_lines):
        data = []
        data.append(input_line.strip())
        results = tm.translate(input_line.strip())
        data.append(results[0][1])
        data.append(gold_line.strip())
        csv_f.writerow(data)

if __name__ == '__main__':
    main()