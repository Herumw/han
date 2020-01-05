"""change n_latent to 10.  change filter size to 50"""

import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn
import datetime
import sys
import random
import itertools
from keras.preprocessing.sequence import pad_sequences

import pickle
import os
from model.model_2019_2_25_TF_testing8_MTL_SA_Rec_vis3 import myDeepCoNN_2019_1_4_word_rev_level_attn_FM_MTL_SA_Rec as mymodel
#print(sys.path)

save_dir = "./saved_model/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


np.random.seed(2017)
random_seed = 2017

#tf.flags.DEFINE_string("word2vec", "../../data/glove.6B/glove.6B.50d.txt", "pre-trained embeddings (default: None)")
word2vec="../../data/glove.6B/glove.6B.50d.txt"

#tf.flags.DEFINE_string("para", "../../data/dataset/Digital_music/para_1", "Data parameters")
#tf.flags.DEFINE_string("data", "../../data/dataset/Digital_music/data_1", "Dataset")
para="../../data/dataset/Digital_music/para_1"
data="../../data/dataset/Digital_music/data_1"


# ==================================================

# Model Hyperparameters
embedding_dim=50
filter_sizes="3"
num_filters="50"
dropout_keep_prob=0.9
l2_reg_lambda=0.2
l2_reg_V=0

# tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding ")
# tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
#tf.flags.DEFINE_string("num_filters", "50", "Number of filters per filter size")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability ")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularizaion lambda")
# tf.flags.DEFINE_float("l2_reg_V", 0, "L2 regularizaion V")

# Training parameters
batch_size=128
num_epochs=4
evaluate_every=50
checkpoint_every=500
# tf.flags.DEFINE_integer("batch_size", 128, "Batch Size ")
# tf.flags.DEFINE_integer("num_epochs", 4, "Number of training epochs ")
# tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps ")
# tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps ")


# Misc Parameters
allow_soft_placement=True
log_device_placement=False
#tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#之前下面这句话是没有注释的
# FLAGS = tf.flags.FLAGS
# print(FLAGS.num_filters)

#FLAGS(sys.argv)


print("Loading data...")
para = pickle.load(open(para, 'rb'))
data = pickle.load(open(data, 'rb'))

user_num = para['user_num'] #5542
item_num = para['item_num'] #3569

user_length = para['user_length']
item_length = para['item_length']

train_length = para['train_length']#53624
val_length = para['val_length']#5541
test_length = para['test_length']#5541

train = data['train']#[[4501, 1507, 1.0], [405, 616, 4.0]]
val = data['val']
test = data['test']

u_rev_set = data['u_rev_set'] #每一个用户映射一个列表，列表里是很多个三元组(item_id,rev_id,rating)
i_rev_set = data['i_rev_set']

reviews = data['reviews'] #每条评论31个单词

user_map = data['user_map']
item_map = data['item_map']
word_map = data['word_map']

vocabulary_user = word_map
vocabulary_item = word_map
#print(len(vocabulary_user)) 共32927个单词

session_conf = tf.ConfigProto(
    allow_soft_placement=allow_soft_placement,
    log_device_placement=log_device_placement)
session_conf.gpu_options.allow_growth = True

#num_filters=list(map(int, num_filters.split(","))), #num_filters=50
#num_filters=50

print("Creating graph...")
deep = mymodel.DeepCoNN(
    user_num=user_num,
    item_num=item_num,
    user_length=user_length,
    item_length=item_length,
    num_classes=1,
    user_vocab_size=len(vocabulary_user),
    item_vocab_size=len(vocabulary_item),
    embedding_size=embedding_dim,
    fm_k=8,
    filter_sizes=list(map(int, filter_sizes.split(","))), #filter_sizes=3
    num_filters=list(map(int, num_filters.split(","))), #num_filters=50
    l2_reg_lambda=l2_reg_lambda,
    l2_reg_V=l2_reg_V,
    n_latent=10,
    Rec_weights=0.2,
    SA_weight=0.8,
    Helpfulness_weight=0.0)
saver = tf.train.Saver()


def train_step(u_text_batch, i_text_batch, uid_batch, iid_batch, y_batch, user_review_rating_batch, item_review_rating_batch, model, sess):
    """
    A single training step
    """
    feed_dict = {
        model.input_u: u_text_batch,
        model.input_i: i_text_batch,
        model.input_y: y_batch,
        model.input_uid: uid_batch,
        model.input_iid: iid_batch,
        model.input_u_rev_rating: user_review_rating_batch,
        model.input_i_rev_rating: item_review_rating_batch,
        model.dropout_keep_prob: dropout_keep_prob
    }
    _, step, loss, mse, mae = sess.run(
        [model.train_op, model.global_step, model.loss, model.mse, model.mae],
        feed_dict)

    return mse, mae


def dev_step(u_text_batch, i_text_batch, uid_batch, iid_batch, y_batch, user_review_rating_batch, item_review_rating_batch, model, sess,):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        model.input_u: u_text_batch,
        model.input_i: i_text_batch,
        model.input_y: y_batch,
        model.input_uid: uid_batch,
        model.input_iid: iid_batch,
        model.input_u_rev_rating: user_review_rating_batch,
        model.input_i_rev_rating: item_review_rating_batch,
        model.dropout_keep_prob: 1.0
    }
    step, loss, mse, mae = sess.run(
        [model.global_step, model.loss, model.mse, model.mae],
        feed_dict)

    return mse, mae


def prepare_data(batch, u_rev_set, i_rev_set, reviews, user_length, item_length, is_drop=True):
    uid_batch, iid_batch, y_batch = list(zip(*batch))
    # print(uid_batch[:5])
    # print(iid_batch[:5])
    # print(y_batch[:5])

    user_review_batch = []
    user_review_rating_batch = []
    for row, uid in enumerate(uid_batch):
        review_set = u_rev_set[uid]
        # drop the  review written to iid's review. keep 20 reviews. mapped rev_id to review text
        # a = [reviews[rev_id] for (iid, rev_id) in review_set]
        # b = [reviews[rev_id] for (iid, rev_id) in review_set if iid != iid_batch[row]]
        # print("user: ", len(a), len(b))

        if is_drop:
            review_set = [(rating, reviews[rev_id]) for (iid, rev_id, rating) in review_set if iid != iid_batch[row]][:20]
        else:
            review_set = [(rating, reviews[rev_id]) for (iid, rev_id, rating) in review_set][:20]
        #print(review_set)
        #[(4.0, [9, 178, 883, 193, 258, 7, 460, 5, 138, 44, 6, 36, 1284, 21, 9, 1674, 139, 7, 670, 11, 419, 9, 105, 5, 24845, 7, 3, 254, 13796, 195, 2]), (4.0, [28, 111, 144, 89, 1479, 58, 14, 3, 12, 4, 282, 12, 3, 382, 904, 1203, 7201, 8667, 3105, 6, 5, 2918, 23651, 307, 68, 253, 1382, 39, 9, 83, 2]), (4.0, [9, 178, 883, 7, 1798, 11, 27, 5, 208, 742, 71, 9, 67, 134, 10, 9, 491, 54, 6, 3, 77, 20, 10, 243, 24, 126, 251, 6, 46, 1982, 2]), (5.0, [11, 8, 175, 3176, 52, 97, 3033, 10, 12, 477, 309, 1595, 4, 72, 6, 31, 10, 12, 2477, 3794, 9, 1848, 7, 228, 610, 1008, 7, 10, 5, 179, 2]), (4.0, [171, 354, 9, 94, 3, 1617, 6, 1358, 6133, 8405, 16, 1069, 42, 448, 1915, 258, 33, 43, 2430, 111, 6, 3, 29, 33, 11, 50, 4, 6857, 3, 895, 2]), (5.0, [55, 25, 5, 66, 71, 9, 350, 7, 670, 5, 179, 6, 39, 553, 16, 11, 4482, 20, 2688, 9, 729, 24, 367, 111, 14, 1854, 271, 10, 12, 867, 2]), (5.0, [9, 35, 735, 11, 15, 2341, 21, 5, 7577, 161, 13, 19, 35, 7, 122, 7, 111, 252, 7, 861, 9, 370, 10, 8425, 13, 9, 301, 199, 5, 124, 2]), (5.0, [146, 116, 4, 162, 9, 433, 14, 16, 36, 151, 86, 7, 202, 53, 55, 35, 98, 129, 3806, 60, 3003, 6041, 188, 13, 9, 167, 123, 63, 7, 36, 2]), (5.0, [10, 12, 262, 121, 4830, 10, 31, 125, 13, 25, 36, 67, 297, 938, 502, 3, 1035, 939, 7, 5064, 12, 364, 3959, 3655, 33, 5, 18089, 2392, 1036, 13, 2]), (3.0, [107, 216, 7, 1544, 6289, 12, 3619, 176, 9, 25, 194, 1087, 7, 202, 31, 3, 1995, 553, 6251, 4969, 124, 11, 368, 447, 3303, 5, 21912, 14, 213, 33, 2]), (5.0, [55, 12, 30, 5, 179, 96, 9, 57, 130, 63, 11, 127, 790, 7, 674, 36, 137, 7, 3, 111, 89, 1980, 11, 15, 4, 11, 85, 22, 97, 2442, 2]), (3.0, [11, 3292, 9, 4562, 139, 7, 122, 7, 3, 67, 145, 33, 308, 12, 52, 11, 717, 10039, 96, 15, 75, 377, 16, 3, 6496, 9, 25, 2766, 1604, 6, 2]), (5.0, [28, 5, 179, 6, 45, 946, 20693, 174, 9, 25, 1988, 7, 11, 15, 49, 1511, 31085, 12, 7302, 3, 39, 14, 73, 25, 2341, 21, 3509, 4774, 4, 3, 2]), (5.0, [9, 25, 1988, 7, 11, 215, 6, 29, 14, 5, 204, 2795, 78, 9, 432, 237, 17, 96, 39, 112, 347, 9, 1848, 7, 228, 44, 63, 39, 33, 3, 2]), (5.0, [4408, 7, 1444, 25, 27, 6, 3, 72, 20692, 3296, 9, 35, 94, 10, 42, 903, 3, 3038, 6, 3, 333, 9, 456, 33, 154, 5, 1057, 89, 94, 7, 2]), (4.0, [1685, 3034, 3, 372, 142, 33, 11, 85, 3636, 52, 5, 66, 14, 36, 164, 71, 9, 25, 460, 600, 44, 6, 709, 17, 824, 25, 3412, 14, 39, 21, 2]), (5.0, [55, 23, 54, 86, 13, 2616, 5, 460, 5, 683, 13, 8, 30, 467, 521, 10, 12, 5, 280, 6, 8682, 13, 8, 1061, 4, 5, 565, 195, 71, 10, 2]), (5.0, [9, 1848, 7, 113, 6, 200, 5934, 822, 4, 11, 218, 21, 8063, 20, 14, 3028, 898, 5934, 822, 8, 2093, 63, 1438, 58, 14, 5, 25472, 460, 44, 6, 2]), (4.0, [36, 67, 7018, 15, 25, 4562, 3930, 65, 8, 49, 116, 131, 223, 13, 15, 3377, 46, 3635, 1815, 17, 43, 713, 4, 3446, 9476, 304, 3, 127, 84, 9, 2]), (5.0, [11, 25, 36, 245, 223, 12, 218, 107, 104, 248, 7, 756, 4, 10, 507, 43, 5, 208, 992, 7, 74, 99, 78, 56, 604, 306, 232, 10, 12, 480, 2])]
        #上面评论最多31个数字，即每条评论最多31个单词
        if len(review_set) == 0:
            review_set = [(0, [0] * user_length)]  # 0 is for unknown rating class
        rating_list, review_set = list(zip(*review_set))
        #print(rating_list)
        #(5.0, 5.0, 3.0, 5.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 4.0)
        #共20条评论的评分
        review_set = list(itertools.chain.from_iterable(review_set))  # concat reviews (list of word_id)
        user_review_rating_batch.append(rating_list)  # (list of list of rating)
        user_review_batch.append(review_set)  # (list of list of word_id)

    item_review_batch = []
    item_review_rating_batch = []
    for row, iid in enumerate(iid_batch):
        review_set = i_rev_set[iid]
        # drop the  review written to uid's review. keep 20 reviews. mapped rev_id to review text
        # a = [reviews[rev_id] for (uid, rev_id) in review_set]
        # b = [reviews[rev_id] for (uid, rev_id) in review_set if uid != uid_batch[row]]
        # print("item: ", len(a), len(b))
        if is_drop:
            review_set = [(rating, reviews[rev_id]) for (uid, rev_id, rating) in review_set if uid != uid_batch[row]][:20]
        else:
            review_set = [(rating, reviews[rev_id]) for (uid, rev_id, rating) in review_set][:20]

        # print("review_set\n", review_set)
        if len(review_set) == 0:
            review_set = [(0, [0] * item_length)]
        rating_list, review_set = list(zip(*review_set))
        review_set = list(itertools.chain.from_iterable(review_set))  # concat reviews
        item_review_rating_batch.append(rating_list)
        item_review_batch.append(review_set)

    # pad to 31*20
    user_review_batch = pad_sequences(
        user_review_batch, maxlen=user_length, dtype='int32', padding='post', truncating='post', value=0.0)
    item_review_batch = pad_sequences(
        item_review_batch, maxlen=item_length, dtype='int32', padding='post', truncating='post', value=0.0)

    user_review_rating_batch = pad_sequences(
        user_review_rating_batch, maxlen=20, dtype='int32', padding='post', truncating='post', value=0.0)
    item_review_rating_batch = pad_sequences(
        item_review_rating_batch, maxlen=20, dtype='int32', padding='post', truncating='post', value=0.0)

    # print("user_review_batch\n", user_review_batch[:5])
    # print("user_review_rating_batch\n", user_review_rating_batch[:5])
    # print("item_review_batch\n", item_review_batch[:5])
    # print("item_review_rating_batch\n", item_review_rating_batch[:5])
    #第一行输出：
    #[[197    9 1479 ...    0    0    0]  共31*20个数字
    #[                                 ]
    #[                                ]]
    #第三行输出:
    #[[5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0 0 0 0 0] 共20个评分，每个评分对应31个单词
    #[                                        ]
    #[                                        ]]

    # print("uid_batch length")
    # print(len(uid_batch)) 每一个batch都是128
    # print(len(user_review_batch)) 共128行 31*20个数字


    return uid_batch, iid_batch, y_batch, \
           user_review_batch, item_review_batch, user_review_rating_batch, item_review_rating_batch


def main():
    with tf.device('/gpu:0'):
        with tf.Session(config=session_conf) as sess:
            tf.set_random_seed(random_seed)
            sess.run(tf.initializers.global_variables())

            # create emb matrix
            print("Creating Embedding matrix...")
            if word2vec:
                print("Load word2vec u file {}\n".format(word2vec))
                initW = np.random.uniform(-1.0, 1.0, (len(word_map), embedding_dim))
                with open(word2vec, "r", encoding="utf-8") as f:
                    for line in f:
                        l = line.split()
                        word = l[0]
                        if word in word_map:
                            idx = word_map[word]
                            initW[idx] = np.array(l[1:], dtype=np.float32)
                sess.run(deep.W1.assign(initW))
                sess.run(deep.W2.assign(initW))
                del initW

            # training
            print("Training...")
            #batch_size = batch_size
            #num_epochs = num_epochs
            save_path = ""

            # data feed:  uid_batch, iid_batch, y_batch, user_review_batch, item_review_batch
            best_mse = 2.0
            for epoch in range(num_epochs):
                mse_epoch = 0
                mae_epoch = 0

                mse_train = 0
                mae_train = 0
                random.shuffle(train)
                n_batches = len(train) // batch_size
                print("n_batches in one epoch: ", n_batches) if epoch == 0 else None
                for step in range(n_batches):
                    batch = train[step * batch_size: (step+1) * batch_size]
                    # prepare data batch
                    uid_batch, iid_batch, y_batch, user_review_batch, item_review_batch, user_review_rating_batch, item_review_rating_batch = prepare_data(
                        batch=batch, u_rev_set=u_rev_set, i_rev_set=i_rev_set, reviews=reviews, user_length=user_length,
                        item_length=item_length, is_drop=True)

                    # train step
                    mse, mae = train_step(
                        user_review_batch, item_review_batch, uid_batch, iid_batch, y_batch,
                        user_review_rating_batch, item_review_rating_batch, deep, sess)
                    mse_train += mse
                    mae_train += mae
                    mse_epoch += mse
                    mae_epoch += mae

                    # validation
                    if step % evaluate_every == 0 and step > 0:
                        # print train stats
                        print("step {}:".format(step))
                        print("     train:  mse {},  mae {}".format(mse_train / evaluate_every, mae_train / evaluate_every))
                        mse_train = 0
                        mae_train = 0

                        # eval val stats
                        mse_valid = 0
                        mae_valid = 0
                        n_batches_val = len(val) // batch_size
                        for step_val in range(n_batches_val):
                            batch_val = val[step_val * batch_size: (step_val + 1) * batch_size]

                            # prepare data batch
                            uid_batch_val, iid_batch_val, y_batch_val, user_review_batch_val, item_review_batch_val, \
                            user_review_rating_batch, item_review_rating_batch = \
                                prepare_data(batch=batch_val, u_rev_set=u_rev_set, i_rev_set=i_rev_set, reviews=reviews,
                                             user_length=user_length, item_length=item_length, is_drop=True)

                            # dev step
                            mse, mae = dev_step(
                                user_review_batch_val, item_review_batch_val, uid_batch_val, iid_batch_val, y_batch_val,
                                user_review_rating_batch, item_review_rating_batch, deep, sess)
                            mse_valid += mse
                            mae_valid += mae

                        # save model
                        if best_mse > mse_valid / n_batches_val:
                            best_mse = mse_valid / n_batches_val
                            # save_path = saver.save(sess, save_dir + "model_" + "epoch_" + str(epoch) + ".ckpt")
                            save_path = saver.save(sess, save_dir + "model" + ".ckpt")
                            print("Model saved in path: %s" % save_path)
                        print("     valid:  mse {},  mae {}".format(mse_valid / n_batches_val, mae_valid / n_batches_val))

                print("\nepoch {}:  mse {},  mae {}\n".format(epoch, mse_epoch / n_batches, mae_epoch / n_batches) + "-"*80)
                print("           best valid mse {}\n".format(best_mse) + "-" * 80)
            print("best mse of all {}\n".format(best_mse))


            # test
            saver.restore(sess, save_path)
            print("Model restored at  ", save_path)

            mse_test = 0
            mae_test = 0
            n_batches_test = len(test) // batch_size
            for step_test in range(n_batches_test):
                batch_test = test[step_test * batch_size: (step_test + 1) * batch_size]

                # prepare data batch
                uid_batch_test, iid_batch_test, y_batch_test, user_review_batch_test, item_review_batch_test, \
                user_review_rating_batch_test, item_review_rating_batch_test = \
                    prepare_data(batch=batch_test, u_rev_set=u_rev_set, i_rev_set=i_rev_set, reviews=reviews,
                                 user_length=user_length, item_length=item_length, is_drop=True)

                # dev step
                mse, mae = dev_step(
                    user_review_batch_test, item_review_batch_test, uid_batch_test, iid_batch_test, y_batch_test,
                    user_review_rating_batch_test, item_review_rating_batch_test, deep, sess)
                mse_test += mse
                mae_test += mae

            print("-" * 80)
            print("test:  mse {},  mae {}".format(mse_test / n_batches_test, mae_test / n_batches_test))
            return mse_test / n_batches_test


if __name__ == '__main__':
    main()