import numpy
import os
import logging
from nmt import train

def main(job_id, params):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words-src'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0], 
                     patience=1000,
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=100,
                     sampleFreq=100,
		     baseDir='/u/vineeku6/data/yahoo-answers',
                     datasets=['train_src.txt.cleaned', 'train_target.txt.cleaned'],
                     valid_datasets=['valid_src.txt.cleaned', 'valid_target.txt.cleaned'],
                     dictionaries=['src.txt.cleaned.cat.pkl', 'target.txt.cleaned.pkl'],
                     use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['test.npz'],
        'dim_word': [300],
        'dim': [100],
        'n-words': [30000], 
        'n-words-src': [50000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [1.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
