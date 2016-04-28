import numpy
import os
import logging, argparse
from nmt import train


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', help='base directory')
    parser.add_argument('wordvec', help='word2vec trained file')
    parser.add_argument('model', help='model name')
    parser.add_argument('--srcwords', default=60000, type=int, help='src vocabulary size')
    parser.add_argument('--targetwords', default=30000, type=int, help='target vocabulary size')
    parser.add_argument('--dimword', default=300, type=int, help='word embedding dimension')
    parser.add_argument('--dimhidden', default=300, type=int, help='hidden layer dimension')
    parser.add_argument('--batch', default=128, type=int, help='Batch size')
    parser.add_argument('--dispfreq', default=500, type=int, help='Display Frequency')
    parser.add_argument('--savefreq', default=500, type=int, help='Save Frequency')
    parser.add_argument('--validfreq', default=500, type=int, help='Validation Frequency')
    parser.add_argument('--samplefreq', default=500, type=int, help='Sample Frequency')
    parser.add_argument('--maxlen', default=50, type=int, help='Maximum length of sequence')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
    parser.add_argument('--clipc', default=1.0, type=float, help='Gradient Clip threshold')
    parser.add_argument('--alphac', default=0.0, type=float, help='Alpha')
    parser.add_argument('--decay', default=0.0, type=float, help='Decay rate')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    validerr = train(saveto=args.model + '.npz',
                     reload_=False,
                     dim=args.dimhidden,
                     dim_word=args.dimword,
                     n_words=args.targetwords,
                     n_words_src=args.srcwords,
                     decay_c=args.decay,
                     clip_c=args.clipc,
                     alpha_c=args.alphac,
                     lrate=args.lr,
                     optimizer='adam',
                     patience=1000,
                     maxlen=args.maxlen,
                     batch_size=args.batch,
                     valid_batch_size=args.batch,
                     validFreq=args.validfreq,
                     dispFreq=args.dispfreq,
                     saveFreq=args.savefreq,
                     sampleFreq=args.samplefreq,
                     baseDir=args.basedir,
                     word2vecFile=args.wordvec,
                     datasets=['train_src.txt', 'train_target.txt'],
                     valid_datasets=['valid_src.txt', 'valid_target.txt'],
                     dictionaries=['src.txt.pkl', 'target.txt.pkl'],
                     use_dropout=False,
                     overwrite=True)

    logging.info('FINAL Validation error: '+ str(validerr))


if __name__ == '__main__':
    main()
