import tensorflow as tf 
from utils import load_word2vec


class Model(tf.keras.Model):
    def __init__(self, args, model):
        super(Model, self).__init__()
        self.embeddings_matrix = load_word2vec(args)
        self.embeddings = tf.keras.layers.Embedding(args.vocab_size,
                                                    args.embedding_dim,
                                                    weights = [self.embeddings_matrix],
                                                    trainable = False
                                                    )
        self.encoder = model
        self.dense = tf.keras.layers.Dense(units=args.vocab_size, 
                                           activation='softmax', 
                                           #input_shape=(args.max_seq_len, args.embedding_dim)
                                           )

    def call(self, inputs, inputs_ids, masks, labels):
        inputs_embeddings = self.embeddings(inputs_ids)
        # masks = tf.expand_dims(masks, axis=-1)
        output = self.encoder({'inputs_embeds':inputs_embeddings, 'attention_mask':masks})[0]
        output = self.dense(output)
        # labels = tf.expand_dims(labels, axis=-1)
        output_tmp = output[labels != -100]
        labels_tmp = labels[labels != -100]

        # loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tmp, output_tmp)
        return output_tmp, labels_tmp