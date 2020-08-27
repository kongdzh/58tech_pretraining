# -*- coding: utf-8 -*-
"""
pretrain a specified language model(modified bi-lstm as default)
"""

from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import argparse
import models
import utils
import gc
import time

from pre_train import pre_train

def main():
    # tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_file", type=str, default="", help="Input train file.")
    # parser.add_argument("--vocab_file", type=str, default="", help="Input vocab file.")
    # parser.add_argument("--model_save_dir", type=str, default="",
    #                     help="Specified the directory in which the model should stored.")
    # parser.add_argument("--lstm_dim", type=int, default=100, help="Dimension of LSTM cell.")
    # parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension of word embedding.")
    # parser.add_argument("--layer_num", type=int, default=2, help="LSTM layer num.")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    # parser.add_argument("--train_step", type=int, default=10000, help="Number of training steps.")
    # parser.add_argument("--warmup_step", type=int, default=1000, help="Number of warmup steps.")
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="The initial learning rate")
    # parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed value.")
    # parser.add_argument("--print_step", type=int, default=1000, help="Print log every x step.")
    # parser.add_argument("--max_predictions_per_seq", type=int, default=10,
    #                     help="For each sequence, predict x words at most.")
    # parser.add_argument("--weight_decay", type=float, default=0, help='Weight decay rate')
    parser.add_argument("--init_checkpoint", type=str, default="", help="Initial checkpoint")

    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Input train file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimension of LSTM cell.")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Dimension of LSTM cell.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of LSTM cell.")
    parser.add_argument("--train_data_path", type=str, default="E:/CodeSleepEatRepeat/data/58tech/data/pre_train_data", help="Dimension of LSTM cell.")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Dimension of LSTM cell.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Dimension of LSTM cell.")
    parser.add_argument("--intermediate_size", type=int, default=32, help="Dimension of LSTM cell.")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Dimension of LSTM cell.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Dimension of LSTM cell.")
    parser.add_argument("--w2v_output", type=str, default="E:/CodeSleepEatRepeat/data/58tech/data/word2vec.txt", help="Dimension of LSTM cell.")
    parser.add_argument("--vocab_file", type=str, default="E:/CodeSleepEatRepeat/data/58tech/data/vocab_new.txt", help="Dimension of LSTM cell.")
    parser.add_argument("--epochs", type=int, default=10, help="Dimension of LSTM cell.")
    parser.add_argument("--checkpoints_dir", type=str, default="E:/CodeSleepEatRepeat/competitions/58tech/checkpoints", help="Dimension of LSTM cell.")
    

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    pre_train(args)

    gc.collect()

    if args.init_checkpoint:
        print("Creating the checkpoint manager")
        checkpoint_dir = args.init_checkpoint
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    # total_loss = 0
    # num = 0
    # global_step = 0
    # while global_step < args.train_step:
    #     if not args.use_queue:
    #         iterator = utils.gen_batches(training_sens, args.batch_size)
    #     else:
    #         iterator = utils.queue_gen_batches(training_sens, args, word2id, id2word)
    #     for batch_data in iterator:
    #         feed_dict = {model.ph_tokens: batch_data[0],
    #                         model.ph_length: batch_data[1],
    #                         model.ph_labels: batch_data[2],
    #                         model.ph_positions: batch_data[3],
    #                         model.ph_weights: batch_data[4],
    #                         model.ph_dropout_rate: args.dropout_rate}
    #         _, global_step, loss, learning_rate = sess.run([model.train_op, \
    #                                                         model.global_step, model.loss_op,
    #                                                         model.learning_rate_op], feed_dict=feed_dict)

    #         total_loss += loss
    #         num += 1
    #         if global_step % args.print_step == 0:
    #             tf.logging.info("\nglobal step : " + str(global_step) +
    #                             ", avg loss so far : " + str(total_loss / num) +
    #                             ", instant loss : " + str(loss) +
    #                             ", learning_rate : " + str(learning_rate) +
    #                             ", time :" + str(time.strftime('%Y-%m-%d %H:%M:%S')))
    #             tf.logging.info("save model ...")
    #             saver.save(sess, args.model_save_dir + '/lm_pretrain.ckpt', global_step=global_step)
    #             gc.collect()

    #     if not args.use_queue:
    #         utils.to_ids(training_sens, word2id, args, id2word)  # MUST run this for randomization for each sentence
    #     gc.collect()


if __name__ == "__main__":
    main()