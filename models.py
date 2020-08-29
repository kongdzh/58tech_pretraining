import tensorflow as tf 
from utils import load_word2vec
# from modeling_tf_bert import gelu
from transformers.modeling_tf_bert import gelu

# class TFRobertaLMHead(tf.keras.layers.Layer):
#     """Roberta Head for masked language modeling."""

#     def __init__(self, config, input_embeddings, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab_size = config.vocab_size
#         self.dense = tf.keras.layers.Dense(
#             config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
#         )
#         self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
#         self.act = tf.keras.layers.Activation(gelu)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = input_embeddings

#     def build(self, input_shape):
#         self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
#         super().build(input_shape)

#     def call(self, features):
#         x = self.dense(features)
#         x = self.act(x)
#         x = self.layer_norm(x)

#         # project back to size of vocabulary with bias
#         x = self.decoder(x, mode="linear") + self.bias

#         return x


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
        self.dense2 = tf.keras.layers.Dense(args.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.act = tf.keras.layers.Activation(gelu)
        self.bias = self.add_weight(shape=(args.vocab_size,), initializer="zeros", trainable=True, name="bias")
        self.word_embeddings = self.add_weight(
                "weight",
                shape=[args.vocab_size, args.hidden_size],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            )

    def _linear(self, inputs, args):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        x = tf.reshape(inputs, [-1, args.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, args.vocab_size])

    def call(self, inputs, inputs_ids, masks, labels, args):
        inputs_embeddings = self.embeddings(inputs_ids)
        # masks = tf.expand_dims(masks, axis=-1)
        inputs_embeddings = inputs + inputs_embeddings
        output = self.encoder({'inputs_embeds':inputs_embeddings, 'attention_mask':masks})[0]
        output = self.dense2(output)
        output = self.layer_norm(output)
        output = self.act(output)
        output = self._linear(output, args)
        # labels = tf.expand_dims(labels, axis=-1)
        output_tmp = output[labels != -100]
        labels_tmp = labels[labels != -100]

        # loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tmp, output_tmp)
        return output_tmp, labels_tmp


class Model_Roberta(tf.keras.Model):
    def __init__(self, args, model_roberta):
        super(Model_Roberta, self).__init__()
        self.encoder = model_roberta
        self.args = args
        self.embeddings_matrix = load_word2vec(args)
        self.embeddings = tf.keras.layers.Embedding(args.vocab.word_size(),
                                                    args.embedding_dim,
                                                    weights = [self.embeddings_matrix],
                                                    trainable = False
                                                    )

    def call(self, inputs, inputs_ids, masks, args):
        inputs_embeddings = self.embeddings(inputs_ids)
        # masks = tf.expand_dims(masks, axis=-1)
        inputs_embeddings = inputs + inputs_embeddings
        output = self.encoder({'inputs_embeds':inputs_embeddings, 'attention_mask':masks})[0]
        # output = self.dense2(output)
        # output = self.layer_norm(output)
        # output = self.act(output)
        # output = self._linear(output, args)
        # # labels = tf.expand_dims(labels, axis=-1)
        # output_tmp = output[labels != -100]
        # labels_tmp = labels[labels != -100]

        # # loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tmp, output_tmp)
        # return output_tmp, labels_tmp