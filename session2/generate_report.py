import argparse, logging, csv
from translation_model import TranslationModel


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='trained model')
    parser.add_argument('srcd', help='src dictionary')
    parser.add_argument('tgtd', help='target dictionary')
    parser.add_argument('input', help='input text file')
    parser.add_argument('--out', help='report', default='report')
    parser.add_argument('--suffix', help='suffix to add to report', default='test')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    tm = TranslationModel(args.model, args.srcd, args.tgtd)
    f = open('%s-%s.csv'% (args.out, args.suffix), 'w')
    csv_f = csv.writer(f, delimiter=',')

    data = ['Text', 'Question']
    csv_f.writerow(data)
    for line in open(args.input, 'r'):
        data = []
        data.append(line.strip())
        results = tm.translate(line.strip())
        data.append(results[0][1])
        csv_f.writerow(data)

if __name__ == '__main__':
    main()