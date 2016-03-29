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
                     batch_size=128,
                     valid_batch_size=128,
                     validFreq=500,
                     dispFreq=500,
                     saveFreq=500,
                     sampleFreq=500,
		     baseDir='/u/vineeku6/storage/train-data/yans',
		     word2vecFile='/u/vineeku6/storage/data/yahoo-answers/word2vec/yans.vectors',
                     datasets=['train_src.txt', 'train_target.txt'],
                     valid_datasets=['valid_src.txt', 'valid_target.txt'],
                     dictionaries=['src.txt.pkl', 'target.txt.pkl'],
                     use_dropout=params['use-dropout'][0],
		     overwrite=True)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['test.npz'],
        'dim_word': [300],
        'dim': [300],
        'n-words': [30000], 
        'n-words-src': [80000], 
        'optimizer': ['adam'],
        'decay-c': [0.], 
        'clip-c': [1.], 
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
