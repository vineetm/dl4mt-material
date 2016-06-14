import argparse, logging, commands, sys

EXP_PATTERN='ellipsis-exp%d%d*.out'
BLEU_PATTERN='bleu-ellipsis-exp%d%d.out'
REPORT_PATTERN='report-ellipsis-exp%d%d.out'


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', help='Base index such as 0', type=int)
    args = parser.parse_args()
    return args

def get_output(cmd):
    (status, output) = commands.getstatusoutput(cmd)
    if status:
        logging.error(output)
        sys.exit(1)
    return output

def get_best_ve(file):
    iters_cmd = "cat %s | grep Valid: | cut -d ':' -f 6 | cut -d ' ' -f 1"%file
    output = get_output(iters_cmd)
    iters = output.splitlines()

    valid_cmd = "cat %s | grep Valid: | cut -d ':' -f 7 "%file
    output = get_output(valid_cmd)
    ve_scores = output.splitlines()

    assert len(iters) == len(ve_scores)

    min_index = 0
    min_score = float(ve_scores[0])

    for index, ve_score in enumerate(ve_scores):
        ve_score = float(ve_score)
        if ve_score < min_score:
            min_score = ve_score
            min_index = index

    return min_score, int(iters[min_index])


def get_unigram_bleu(file):
    uni_bleu_cmd = " cat %s | grep Uni | cut -d ':' -f 7"%file
    output = get_output(uni_bleu_cmd)
    return float(output)


def get_bleu4(file):
    cmd = "tail -1 %s | cut -d ',' -f 1 | cut -d '=' -f 2"% file
    score = float(get_output(cmd))
    return score


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    for file_num in range(10):
        exp_file = EXP_PATTERN%(args.base, file_num)
        report_file = REPORT_PATTERN%(args.base, file_num)
        bleu_4_file = BLEU_PATTERN%(args.base, file_num)

        ve, iter = get_best_ve(exp_file)
        uni_bleu = get_unigram_bleu(report_file)
        bleu_4 = get_bleu4(bleu_4_file)
        if bleu_4 != 0.00:
            bleu_4 = bleu_4 / 100

        logging.info('Exp %d%d:: BLEU_4:%f BLEU_1:%f VE:%f Iter:%d'%(args.base, file_num, bleu_4, uni_bleu, ve, iter))

if __name__ == '__main__':
    main()
