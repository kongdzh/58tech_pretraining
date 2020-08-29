from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import pickle
import os


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)

def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            lines.append(line[-1])
    return lines

def extract_sentence(pre_train_data_path, path_list):
    ret = []
    lines = read_lines(pre_train_data_path)
    for path in path_list:
        lines += read_lines(path)
    for line in lines:
        ret.append(line.split())
    return ret

def build(pre_train_data_path, path_list, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(pre_train_data_path, path_list)

    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=sentences,
                   size=128, window=5, min_count=min_count, iter=10)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('-1', '0')
    print('-1 vs 0 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    
    build('E:/CodeSleepEatRepeat/data/58tech/data/pre_train_data',
          ['E:/CodeSleepEatRepeat/data/58tech/data/std_data',
          'E:/CodeSleepEatRepeat/data/58tech/data/train_data',
          'E:/CodeSleepEatRepeat/data/58tech/data/test_data'],
          out_path='E:/CodeSleepEatRepeat/data/58tech/data/word2vec.txt',
          sentence_path='E:/CodeSleepEatRepeat/data/58tech/data/sentences.txt')
    
    # sens = extract_sentence('E:/CodeSleepEatRepeat/data/58tech/data/pre_train_data',
    #       ['E:/CodeSleepEatRepeat/data/58tech/data/std_data',
    #       'E:/CodeSleepEatRepeat/data/58tech/data/train_data',
    #       'E:/CodeSleepEatRepeat/data/58tech/data/test_data'])

    # dic = {}
    # for sen in sens:
    #     for s in sen:
    #         dic[s] = 1
    # print(len(dic))