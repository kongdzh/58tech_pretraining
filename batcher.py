import tensorflow as tf
import random
import numpy as np
from utils import load_pkl, load_word2vec
import argparse


class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2id = {'<MASK>':0, '<UNK>':1, '<PAD>':2}
        self.id2word = {0:'<MASK>', 1:'<UNK>', 2:'<PAD>'}
        self.count = 3

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                v = line.strip()

                if v in self.word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % v)

                self.word2id[v] = self.count
                self.id2word[self.count] = v
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                          % (max_size, self.count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" %
              (self.count, self.id2word[self.count - 1]))

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id['<UNK>']
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


def mlm_generator(args, embedding_table):

    train_dataset = tf.data.TextLineDataset(args.train_data_path)
    # train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()
    # i = 0
    for line in train_dataset:
        per_sens_embedding = np.zeros((args.max_seq_len, args.embedding_dim))
        per_sens_ids = np.zeros(args.max_seq_len)
        per_sens_attention_masks = np.zeros(args.max_seq_len)  # huggingface transformers argument 用来分别不同的句子
        per_sens_label = np.zeros(args.max_seq_len)-100

        for word_idx, word in enumerate(line.numpy().decode('utf-8').split()[:args.max_seq_len]):
            per_sens_attention_masks[word_idx] = 1
            if random.random() < args.mlm_probability: # 0.15
                per_sens_label[word_idx] = args.vocab[word]

                if random.random() < 0.8:
                    per_sens_ids[word_idx] = args.vocab['<MASK>']
                elif random.random() < 0.9:
                    # per_sens_embedding[word_idx] = embedding_table[word]
                    # try:
                    #     per_sens_ids[word_idx] = args.vocab[word]
                    # except:
                    per_sens_ids[word_idx] = args.vocab['<UNK>']
                else:
                    while True:
                        random_word = random.sample(args.vocab.keys(), 1)[0]
                        if random_word != word:
                            break
                    per_sens_embedding[word_idx] = embedding_table[random_word] 
                    
                    per_sens_ids[word_idx] = args.vocab[random_word]
            else:
                per_sens_ids[word_idx] = args.vocab[word]
                per_sens_embedding[word_idx] = embedding_table[word]

        yield  per_sens_embedding, per_sens_ids, per_sens_attention_masks, per_sens_label

        # article = raw_record[0].numpy().decode("utf-8")
        # abstract = raw_record[1].numpy().decode("utf-8")

        # start_decoding = vocab.word_to_id(START_DECODING)
        # stop_decoding = vocab.word_to_id(STOP_DECODING)

        # article_words = article.split()[:max_enc_len]
        # enc_len = len(article_words)
        # # 添加mark标记
        # sample_encoder_pad_mask = [1 for _ in range(enc_len)]

        # enc_input = [vocab.word_to_id(w) for w in article_words]
        # enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

        # abstract_sentences = [""]
        # abstract_words = abstract.split()
        # abs_ids = [vocab.word_to_id(w) for w in abstract_words]
        # # abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
        # dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
        # # _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

        # dec_len = len(dec_input)
        # # 添加mark标记
        # sample_decoder_pad_mask = [1 for _ in range(dec_len)]

        # output = {
        #     "enc_len": enc_len,
        #     "enc_input": enc_input,
        #     "enc_input_extend_vocab": enc_input_extend_vocab,
        #     "article_oovs": article_oovs,
        #     "dec_input": dec_input,
        #     "target": target,
        #     "dec_len": dec_len,
        #     "article": article,
        #     "abstract": abstract,
        #     "abstract_sents": abstract_sentences,
        #     "sample_decoder_pad_mask": sample_decoder_pad_mask,
        #     "sample_encoder_pad_mask": sample_encoder_pad_mask,
        # }
        # yield output


def batch_generator(generator, args, embedding_table):
    dataset = tf.data.Dataset.from_generator(lambda: generator(args, embedding_table),
                                             output_types= (tf.float32, tf.int32, tf.int32, tf.int32)
                                             )

    return dataset


def batcher(args, embedding_table):
    dataset = batch_generator(mlm_generator, args, embedding_table)
    dataset_batch = dataset.batch(args.batch_size)
    return dataset_batch


if __name__ == '__main__':
    vocab = Vocab('E:/CodeSleepEatRepeat/data/58tech/data/vocab_new.txt', 50000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Input train file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimension of LSTM cell.")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Dimension of LSTM cell.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of LSTM cell.")
    parser.add_argument("--train_data_path", type=str, default="E:/CodeSleepEatRepeat/data/58tech/samplesdataset/pre_train_data", help="Dimension of LSTM cell.")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Dimension of LSTM cell.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Dimension of LSTM cell.")
    parser.add_argument("--intermediate_size", type=int, default=32, help="Dimension of LSTM cell.")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Dimension of LSTM cell.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Dimension of LSTM cell.")

    args = parser.parse_args()

    args.vocab = vocab.word2id

    embs = load_pkl('E:/CodeSleepEatRepeat/data/58tech/data/word2vec.txt')
    batches = batcher(args, embs)

    for batch in batches:
        print(batch[3][1])
        break

    # print(embs)