from batcher import Vocab, batcher
from utils import load_pkl
from models import Model
import tensorflow as tf 
from transformers import RobertaConfig, TFRobertaModel, TFRobertaForMaskedLM

import logging
logging.disable(30)

# @tf.function
def train_step(model, batch, loss_func, args):
    with tf.GradientTape() as tape:
        predictions, masks_labels = model(batch[0], batch[1], batch[2], batch[3], args)
        loss = loss_func(masks_labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients, loss, predictions, masks_labels

def pre_train(args):
    # 构建词表对象
    vocab = Vocab(args.vocab_file, 50000)

    # 取出词和id的字典
    args.vocab = vocab.word2id
    args.vocab_size = vocab.word_size()

    # 读取预训练好的embeddings
    embs = load_pkl('E:/CodeSleepEatRepeat/data/58tech/data/word2vec.txt')
    embs['<UNK>'] = [0] * args.embedding_dim
    embs['<MASK>'] = [0] * args.embedding_dim
    embs['<PAD>'] = [0] * args.embedding_dim

    # 构建mlm的训练数据
    batches = batcher(args, embs)

    # huggingface transformers 模型配置
    config = RobertaConfig()       
    config.num_hidden_layers = args.num_hidden_layers # 4
    config.hidden_size = args.hidden_size # 32
    config.intermediate_size = args.hidden_size * 4
    config.num_attention_heads = args.num_attention_heads # 8
    config.vocab_size = len(args.vocab) 

    model_roberta = TFRobertaModel(config)

    model = Model(args, model_roberta)
    # model.summary()
    
    optimizer = tf.keras.optimizers.Nadam()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # checkpoint_dir = args.checkpoints_dir
    # ckpt = tf.train.Checkpoint(model=model)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    if args.pretrain_checkpoints_dir:
        print("Creating the checkpoint manager")
        checkpoint_dir = args.pretrain_checkpoints_dir
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

        if ckpt_manager.latest_checkpoint:
            # ckpt.restore('./checkpoints/ckpt-53')
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    count = 0
    best_loss = 20
    for epoch in tf.range(1,args.epochs+1):
        
        for batch in batches:
            # inputs, inputs_ids, attention_masks, labels = batch[0], batch[1], batch[2], batch[3]
            gradients, loss, masks_labels, predictions = train_step(model, batch, loss_func, args)
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)
            train_metric.update_state(masks_labels, predictions)

            logs = 'Epoch={},Loss:{},Accuracy:{}'
            

            # print(predictions)
            # print('-'*20)
            # print(masks_labels)
            # print('*'*20)
            # print(tf.reduce_mean(loss))
            # print('='*20)
            # label = tf.argmax(predictions[0])
            # print(label)
            
            if count % 100 == 0 and count != 0:
                tf.print(tf.strings.format(logs,
                (epoch, train_loss.result(),train_metric.result())))
                tf.print("")
                if count % 1000 == 0 and train_loss.result() < best_loss:
                    best_loss = train_loss.result()
                    ckpt_save_path = ckpt_manager.save()
                    print('*'*20)
                    print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch, ckpt_save_path, best_loss))
                    print('*'*20)
            count += 1
            
        train_loss.reset_states()
        train_metric.reset_states()

    model.encoder.save_pretrained('./pretrained-roberta/')