# for reproducibility
from numpy.random import seed

seed(142)
from tensorflow import set_random_seed

set_random_seed(242)

import os
import sys
import datetime
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit
from keras_emo_models import keras_cnn, keras_bilstm
from keras_monitoring import model_monitoring
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle


def data_coverage(num_of_tokens, doc_mean_len):
    # calculates the maxlen
    # cuts off the concatenated text after this length
    # making a compromise by covering maximum amount of word tokens from each user
    maxlen = int(np.mean(doc_mean_len) + (2.8 * np.std(num_of_tokens)))
    print("new maxlen for documents: {}".format(maxlen))
    # the percentage of users that will be covered under the new maxlen
    coverage = np.sum(num_of_tokens <= maxlen) / len(num_of_tokens)
    print("user coverage: {}".format(coverage))
    print("users above the current maxlen: {}".format(np.sum(num_of_tokens > maxlen)))

    return maxlen


def tokenize_and_padding(docs, labels, max_words, maxlen):
    # Tokenizer removes basic punctuations, tabs and newline characters
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token='<UNK>', lower=False)
    tokenizer.fit_on_texts(docs)
    sequences = tokenizer.texts_to_sequences(docs)

    word_index = tokenizer.word_index

    data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen)
    labels = np.asarray(labels)
    print("data shape: {}".format(data.shape))
    print("labels shape: {}".format(labels.shape))
    return data, labels, word_index, tokenizer


def token_stats(docs):
    num_of_tokens = [len(doc.split()) for doc in docs]
    num_of_tokens = np.array(num_of_tokens)

    doc_max_len = np.max(num_of_tokens)
    doc_mean_len = np.mean(num_of_tokens)
    # getting the number of unique tokens
    numof_unique_tokens = unique_tokens(docs)
    total_numof_tokens = np.sum(num_of_tokens)
    # print number of docs above the mean
    above_mean = [token_len > doc_mean_len for token_len in num_of_tokens]
    print("total number of tokens: {}".format(total_numof_tokens))
    print("total number of unique tokens: {}".format(numof_unique_tokens))
    print("document mean length: {}".format(doc_mean_len))
    print("document max length: {}".format(doc_max_len))
    print("documents above mean count: {}".format(above_mean.count(True)))

    return num_of_tokens, doc_mean_len, total_numof_tokens, numof_unique_tokens


def unique_tokens(docs):
    # to identify the number of unique tokens
    all_tokens = []
    for doc in docs:
        tokens = doc.split()
        all_tokens += tokens

    num_tokens_unique = len(set(all_tokens))
    print("number of unique tokens: {}".format(num_tokens_unique))
    return num_tokens_unique


def data_sampling(data, labels, n_splits, test_size):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=1337)
    for train_index, val_index in sss.split(data, labels):
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

    print("train data shape: {}".format(x_train.shape))
    print("validation data shape: {}".format(x_val.shape))

    return x_train, y_train, x_val, y_val


def prepare_test_data(docs, labels, tokenizer):
    sequences = tokenizer.texts_to_sequences(docs)
    x_test = keras.preprocessing.sequence.pad_sequences(sequences)
    y_test = np.asarray(labels)

    return x_test, y_test


def test_prediction(model, test_data, test_labels):
    y_test_onehot = keras.utils.to_categorical(test_labels, num_classes=4)
    loss, acc = model.evaluate(test_data, y_test_onehot)
    print("loss and accuracy on test data: loss = {}, accuracy = {}".format(loss, acc))
    predictions = model.predict(test_data)
    y_hat = np.argmax(predictions, axis=1)
    return y_hat


def evaluation(y_true, y_pred):
    precision_micro = precision_score(
        y_true=y_true, y_pred=y_pred, average='micro')
    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    f_measure_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    precision_macro = precision_score(
        y_true=y_true, y_pred=y_pred, average='macro')
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f_measure_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred,
                          labels=[0, 1, 2, 3])

    cr = classification_report(y_true=y_true, y_pred=y_pred, target_names=[
        'fear', 'anger', 'sadness', 'joy'])

    print("micro | precision: {} | recall: {} | f1_score: {}\n".format(
        precision_micro, recall_micro, f_measure_micro))
    print("macro | precision: {} | recall: {} | f1_score: {}\n".format(
        precision_macro, recall_macro, f_measure_macro))
    print("confusion matrix:\n{}".format(cm))
    print("\nclassification report:\n {}".format(cr))


if __name__ == '__main__':
    # set the printing options
    pd.set_option('display.max_colwidth', -1)
    df = pd.read_csv(os.path.join(os.getcwd(), 'data/df_cleaned_v02.csv'))
    # pp_pre_emoemo: text after removing the emoji and emoticons
    # pp_pre_emopunc: text after removing the punctuations
    # all the special characters are replaced and not removed
    print("class distribution:\n{}".format(df['emo_label'].value_counts()))
    # fear(0)       2252
    # anger(1)      1701
    # sadness(2)    1533
    # joy(3)        1616

    # replace the labels with integer values
    df['emo_label'] = df.emo_label.str.replace('fear', '0').str.replace(
        'anger', '1').str.replace('sadness', '2').str.replace('joy', '3')
    df['emo_label'] = df.emo_label.astype(np.int32)

    print("dataframe shape before removing the duplicates: {}".format(df.shape))
    # data types of different fields
    print("data types of the dataframe:\n{}".format(df.dtypes))

    # class distribution
    print("class distribution after categorical replacement:\n{}".format(df['emo_label'].value_counts()))

    # make the multiclass multilabel task to a multiclass task
    print("number of duplicated records:\n{}".format(df.tweet.duplicated().sum()))
    # remove duplicate tweets
    df.drop_duplicates(subset=['tweet'], keep='first', inplace=True)
    print("class distribution after removing duplicates:\n{}".format(df['emo_label'].value_counts()))
    print("data frame shape after removing duplicates: {}".format(df.shape))

    # separating the train/test data
    train_data = df.loc[(df.type == 'train') | (
            df.type == 'test'), ('pp_pre_emopunc', 'emo_label')]

    # train data class distribution
    print("train data class distribution:\n{}".format(train_data.emo_label.value_counts()))

    test_data = df.loc[df.type == 'dev', ('pp_pre_emopunc', 'emo_label')]

    # test data class distribution
    print("test data class distribution:\n{}".format(test_data.emo_label.value_counts()))

    train_tweets = np.array(train_data.pp_pre_emopunc.tolist())
    train_labels = np.array(train_data.emo_label.tolist())
    test_tweets = np.array(test_data.pp_pre_emopunc.tolist())
    test_labels = np.array(test_data.emo_label.tolist())

    num_of_tokens, doc_mean_len, total_numof_tokens, numof_unique_tokens = token_stats(train_tweets)
    maxlen = data_coverage(num_of_tokens, doc_mean_len)

    # for the training data
    max_words = numof_unique_tokens
    data, labels, word_index, tokenizer = tokenize_and_padding(train_tweets, train_labels, numof_unique_tokens,
                                                               maxlen)

    # if the number of unique tokens are larger than the word_index
    if numof_unique_tokens > len(word_index):
        max_words = len(word_index)

    with open(os.path.join(os.getcwd(), 'embedding_matrix.pickle'), 'rb') as input_pickle:
        embedding_matrix = pickle.load(input_pickle)

    x_train, y_train, x_val, y_val = data_sampling(data=data, labels=labels, n_splits=1, test_size=0.1)
    one_hot_train_labels = keras.utils.to_categorical(y_train, num_classes=4)
    one_hot_eval_labels = keras.utils.to_categorical(y_val, num_classes=4)

    print("train labels:")
    print(y_train[:5])
    print("train data one-hot encoding sample:")
    print(one_hot_train_labels[:5])

    # embedding_matrix = cnn_model.get_embeddings(word_index=word_index)

    # hyperparameters
    hyper_params = {'learning_rate': 1e-1,
                    'epochs': 50,
                    'batch_size': 32,
                    'maxlen': maxlen,
                    'embedding_dim': 300,
                    'max_words': max_words,
                    'optimizer': 'adam',
                    'loss': 'categorical_crossentropy'}
    # get embedding matrix

    # for calculating the process time
    tic = time.process_time()
    # # creating the models
    model = keras_cnn(hyper_params)

    # compile the model
    model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], metrics=['accuracy'])

    d = datetime.datetime.today()
    timestamp = d.strftime('%Y%m%d_%H%M%S')
    # creating folders to store the model file and the tensorlogs
    model_checkpoint_folder = os.path.abspath(os.path.join(os.path.curdir, 'models/emo', timestamp))
    tensorlog_folder = os.path.abspath(os.path.join(os.path.curdir, 'tensorlogs/emo', timestamp))

    if not os.path.exists(model_checkpoint_folder):
        os.mkdir(model_checkpoint_folder)
    model_checkpoint_path = os.path.join(model_checkpoint_folder, 'current_best.h5')

    if not os.path.exists(tensorlog_folder):
        os.mkdir(tensorlog_folder)

    model.fit(x_train, one_hot_train_labels,
              epochs=hyper_params['epochs'],
              batch_size=hyper_params['batch_size'],
              callbacks=model_monitoring(model_checkpoint_path, tensorlog_folder),
              validation_data=(x_val, one_hot_eval_labels))

    print("evaluate on test data...")
    x_test, y_test = prepare_test_data(test_tweets, test_labels, tokenizer)

    print("loading the saved model: {}".format(model_checkpoint_path))
    model_saved = keras.models.load_model(model_checkpoint_path)
    y_hat = test_prediction(model_saved, x_test, y_test)

    # evaluation metrics
    evaluation(y_true=y_test, y_pred=y_hat)
    print("completion time in seconds: {}".format(time.process_time() - tic))
