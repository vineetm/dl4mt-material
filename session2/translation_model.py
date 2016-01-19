import cPickle as pkl
import numpy
from nltk.tokenize import word_tokenize as tokenizer
from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)


class TranslationModel:
    def __init__(self, trained_model, src_dict, target_dict):
        # load model model_options
        with open('%s.pkl' % trained_model, 'rb') as f:
            self.options = pkl.load(f)

        # load source dictionary and invert
        with open(src_dict, 'rb') as f:
            self.word_dict = pkl.load(f)
        self.word_idict = dict()

        for kk, vv in self.word_dict.iteritems():
            self.word_idict[vv] = kk

        self.word_idict[0] = 'EOS'
        self.word_idict[1] = 'UNK'

        # load target dictionary and invert
        if self.target_dict is None:
            self.word_dict_trg = self.word_dict
            self.word_idict_trg = self.word_idict
        else:
            with open(target_dict, 'rb') as f:
                self.word_dict_trg = pkl.load(f)
            self.word_idict_trg = dict()
            for kk, vv in self.word_dict_trg.iteritems():
                self.word_idict_trg[vv] = kk
            self.word_idict_trg[0] = 'EOS'
            self.word_idict_trg[1] = 'UNK'

        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        self.trng = RandomStreams(1234)

        # allocate model parameters
        params = init_params(self.options)

        # load model parameters and set theano shared variables
        self.params = load_params(trained_model, params)
        self.tparams = init_tparams(params)

        # word index
        self.f_init, self.f_next = build_sampler(self.tparams, self.options, self.trng)

    def sent2seq(self, sentence):
        sentence = sentence.lower()
        tokens = tokenizer(sentence)
        seq = [self.word_dict[token] if token in self.word_dict else 1 for token in tokens]
        seq = [w if w < self.options['n_words_src'] else 1 for w in seq]
        return seq

    def seq2words(self, seq):
        ww = []
        for w in seq:
            if w == 0:
                return ww
            ww.append(self.word_idict_trg[w])
        return ww

    def translate(self, input_text, k=16, maxlen=50):
        seq = self.sent2seq(input_text)

        sample, scores = gen_sample(self.tparams, self.f_init, self.f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   self.options, trng=self.trng, k=k, maxlen=maxlen,
                                   stochastic=False, argmax=False)

        results = []
        sorted_index = numpy.argsort(scores)
        for index in sorted_index:
            sample_sentence = ' '.join(self.seq2words(sample[index]))
            results.append((scores[index], sample_sentence))
        return results