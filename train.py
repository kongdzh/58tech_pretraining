from batcher import Vocab, batcher
from utils import load_pkl
from models import Model, Model_Roberta
import tensorflow as tf 
from transformers import RobertaConfig, TFRobertaModel, TFRobertaForMaskedLM

import logging
logging.disable(30)

# @tf.function
def train_step(model, batch, loss_func, args):
    with tf.GradientTape() as tape:
        inputs, inputs_ids, attention_masks, labels = batch[0], batch[1], batch[2], batch[3]
        predictions = model(inputs, inputs_ids, attention_masks, args)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients, loss, predictions, labels

def train(args):
    # 构建词表对象
    vocab = Vocab(args.vocab_file, 50000, args.train_data_path)

    # 取出词和id的字典
    args.vocab = vocab

    # 读取预训练好的embeddings
    embs = load_pkl('E:/CodeSleepEatRepeat/data/58tech/data/word2vec.txt')

    # 构建mlm的训练数据
    batches = batcher(args, embs)

    # load pretrained model
    if args.pre_trained_model:
        config = RobertaConfig.from_pretrained(args.pre_trained_model)
        model_roberta = TFRobertaModel.from_pretrained(args.pre_trained_model, config=config)
    else:
        # huggingface transformers 模型配置
        config = RobertaConfig() 
        config.num_hidden_layers = args.num_hidden_layers # 12
        config.hidden_size = args.hidden_size # 128
        config.intermediate_size = args.hidden_size * 4
        config.num_attention_heads = args.num_attention_heads # 8
        config.vocab_size = args.vocab.word_size()

        model_roberta = TFRobertaModel(config)

    model = Model_Roberta(args, model_roberta)
    # model.summary()
    
    optimizer = tf.keras.optimizers.Nadam()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # checkpoint_dir = args.checkpoints_dir
    # ckpt = tf.train.Checkpoint(model=model)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    if args.checkpoints_dir:
        print("Creating the checkpoint manager")
        checkpoint_dir = args.checkpoints_dir
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

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
            gradients, loss, predictions, labels = train_step(model, batch, loss_func, args)
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)
            train_metric.update_state(labels, predictions)

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