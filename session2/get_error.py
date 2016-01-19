import logging, theano, argparse, os
import cPickle as pkl
from nmt import build_model, pred_probs, prepare_data, init_params, load_params, init_tparams
from data_iterator import TextIterator


def get_error(model, test_src, test_target):
    profile=False

    # reload options
    f = open('%s.pkl' % model, 'rb')
    model_options = pkl.load(f)
    logging.info(model_options)

    logging.info('Building model')
    params = init_params(model_options)

    # reload parameters
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    dict_src = os.path.join(model_options['baseDir'], model_options['dictionaries'][0])
    if len(model_options['dictionaries']) == 1:
        dict_target = None
    else:
        dict_target = os.path.join(model_options['baseDir'], model_options['dictionaries'][1])

    valid = TextIterator(test_src, test_target,
                         dict_src,
                         dict_target,
                         n_words_source=model_options['n_words_src'],
                         n_words_target=model_options['n_words'],
                         batch_size=model_options['valid_batch_size'],
                         maxlen=model_options['maxlen'])

    logging.info('Building f_log_probs...')
    f_log_probs = theano.function(inps, cost, profile=profile)
    valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
    valid_err = valid_errs.mean()
    logging.info('Valid Error:%s'% (str(valid_err)))


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('src')
    parser.add_argument('target')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = setup_args()
    logging.info(args)
    get_error(args.model, args.src, args.target)

if __name__ == '__main__':
    main()