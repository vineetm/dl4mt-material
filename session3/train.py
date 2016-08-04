import argparse, logging
from nmt import train


'''
base: Base Directory, assumes: train_src.txt, train_target.txt, valid_src.txt valid_target.txt, all.txt.pkl
dimword: Word embeddding dimension
dim: Hidden layer dimension
numwords: Vocabulary size
decay: Regularization decay
clip: Clip rate
lr: Learning rate
'''

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', help='Base Directory')
    parser.add_argument('wordvec', help='Word Vectors File')

    #Model Size
    parser.add_argument('-dimword', help='Word embedding dimension', type=int, default=300)
    parser.add_argument('-dim', help='Hidden Layer dimension', type=int, default=300)
    parser.add_argument('-numwords', help='Vocabulary size', type=int, default=30000)

    #Model Regularization parameters
    parser.add_argument('-decay', help='Decay c', type=float, default=0.)
    parser.add_argument('-alpha', help='Alpha', type=float, default=0.)

    #Model learning rate, clip rate, dropout
    parser.add_argument('-clip', help='Clip', default=1., type=float)
    parser.add_argument('-lr', help='Learning rate', default=0.0001, type=float)
    parser.add_argument('-dropout', dest='dropout', help='Use Dropout', default=False, action='store_true')

    #Optimizer
    parser.add_argument('-optimizer', help='Optimizer', default='adadelta')

    #Maximum length of sequence to consider
    parser.add_argument('-maxlen', help='Max len of sequence', default=50, type=int)

    #Mini-batch size
    parser.add_argument('-batchsize', help='Batch size', default=32, type=int)

    #Logging parameters
    parser.add_argument('-samplefreq', help='Sample Frequency', default=500, type=int)
    parser.add_argument('-validfreq', help='Validation Frequency', default=500, type=int)
    parser.add_argument('-dispfreq', help='Display Frequency', default=500, type=int)

    #Model Saving parameters
    parser.add_argument('-savefreq', help='Save Frequency', default=500, type=int)
    parser.add_argument('-model', help='Model save location', default='model.npz')

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)

    train(dim_word=args.dimword, dim=args.dim, dispFreq=args.dispfreq, decay_c=args.decay, alpha_c=args.alpha,
          clip_c=args.clip, lrate=args.lr, n_words_src=args.numwords, n_words=args.numwords, maxlen=args.maxlen,
          optimizer=args.optimizer, batch_size=args.batchsize, saveto=args.model, validFreq=args.validfreq,
          saveFreq=args.savefreq, sampleFreq=args.samplefreq,
          datasets=['%s/train_src.txt'%args.base, '%s/train_target.txt'%args.base],
          valid_datasets=['%s/valid_src.txt'%args.base, '%s/valid_target.txt'%args.base],
          dictionaries=['%s/all.txt.pkl'%args.base, '%s/all.txt.pkl'%args.base],
          use_dropout=args.dropout)

if __name__ == '__main__':
    main()