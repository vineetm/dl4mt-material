import argparse, logging, codecs
from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk.corpus import stopwords

stopw = set(stopwords.words('english'))


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out1', help = 'Output 1')
    parser.add_argument('out2', help = 'Output 2')
    parser.add_argument('input', help = 'Input')
    parser.add_argument('output', help='Selected Output')
    args = parser.parse_args()
    return args


def get_scores(candidate1, candidate2, input):
    #score_1 = bleu([input.split()], candidate1.split(), weights=(1.0,))
    #score_2 = bleu([input.split()], candidate2.split(), weights=(1.0,))

    input_kw = set(input.split()) - stopw
    kw1 = set(candidate1.split()) - stopw
    kw2 = set(candidate2.split()) - stopw
    match1 = kw1.intersection(input_kw)
    match2 = kw2.intersection(input_kw)
    return len(match1), len(match2)

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    out1_lines = codecs.open(args.out1, 'r', 'utf-8').readlines()
    out2_lines = codecs.open(args.out2, 'r', 'utf-8').readlines()

    picked_num1 = 0
    picked_num2 = 0

    input_lines = codecs.open(args.input, 'r').readlines()

    fw = codecs.open(args.output, 'w', 'utf-8')
    for index, (out1, out2, input) in enumerate(zip(out1_lines, out2_lines, input_lines)):
        q2 = input.split('END')[2]
        score_1, score_2 = get_scores(out1, out2, q2)
        logging.info('Index:%d score1: %f score2: %f'% (index, score_1, score_2))

        if score_1 >= score_2:
            picked_out = out1
            picked_num1 += 1
        else:
            picked_out = out2
            picked_num2 +=1
        fw.write(picked_out.strip() + '\n')

    logging.info('Picked1: %d Picked:%d'% (picked_num1, picked_num2))

if __name__ == '__main__':
    main()
