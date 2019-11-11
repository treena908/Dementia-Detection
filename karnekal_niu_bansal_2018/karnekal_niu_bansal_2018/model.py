





import os
import xlrd
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import pickle
import numpy as np
import pandas as pd
import sys, time, os, warnings
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score,mean_absolute_error

from keras import Input, Model,initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, concatenate, Bidirectional, Layer
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,TensorBoard
from keras.optimizers import Adagrad
import keras as k
from keras import backend as K
from sklearn.utils import class_weight
#from utilities import Ml_sets,train_test_split_same_patient

### Attention Layer, a contribution from:https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])







def format_example(wordlist):
    string = ''
    for word in wordlist:
        string = string + ' ' + word
    return string

def binary_conversion(decimal_list:list,threshold:float):
    return_list = []
    for element in decimal_list:
        if element >= threshold:
            return_list.append(1)
        if element < threshold:
            return_list.append(0)
    return return_list


def create_CNN_model(vocabulary_size, sequence_len, embedding_matrix, EMB_SIZE):
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, EMB_SIZE, input_length=sequence_len, weights=[embedding_matrix], trainable=False))
    #model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(100, (3), activation='relu'))
    #TODO: add other filters sizes in parallel
    ##voglio in parallelo anche le altre conv1D con le varie dimensioni per i filtri,
    model_conv.add(GlobalMaxPooling1D())
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model_conv.summary()
    return model_conv


def create_CNN_multifilter_model(vocabulary_size, sequence_len, embedding_matrix, EMB_SIZE):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))
    embedding = Embedding(vocabulary_size, EMB_SIZE, input_length=sequence_len, weights=[embedding_matrix], trainable=False)(deep_inputs)  # line A
    conv1 = Conv1D(100, (3), activation='relu',kernel_regularizer=k.regularizers.l1_l2(0.01,0.001))(embedding)
    dropout_1 = Dropout(0.3)(conv1)
    conv2 = Conv1D(100, (4), activation='relu',kernel_regularizer=k.regularizers.l1_l2(0.01,0.001))(embedding)
    dropout_2 = Dropout(0.3)(conv2)
    conv3 = Conv1D(100, (5), activation='relu',kernel_regularizer=k.regularizers.l1_l2(0.01,0.001))(embedding)
    dropout_3 = Dropout(0.3)(conv3)
    conv4 = Conv1D(100, (6), activation='relu',kernel_regularizer=k.regularizers.l1_l2(0.01,0.001))(embedding)
    dropout_4 = Dropout(0.3)(conv4)
    maxpool1 = MaxPooling1D(pool_size=48)(dropout_1)
    maxpool2 = MaxPooling1D(pool_size=47)(dropout_2)
    maxpool3 = MaxPooling1D(pool_size=46)(dropout_3)
    maxpool4 = MaxPooling1D(pool_size=45)(dropout_4)
    cc1 = concatenate([maxpool1, maxpool2,maxpool3,maxpool4], axis=2)
    flatten = Flatten()(cc1)
    output = Dense(1, activation="sigmoid")(flatten)
    model = Model(inputs=deep_inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def create_LSTM_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE):
    model_lstm = Sequential()
    model_lstm.add(Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix], trainable=False))
    model_lstm.add(LSTM(128))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model_lstm


def create_CNN_LSTM_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE):
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix], trainable=False))
    #model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(100, (3), activation='relu'))
    ##voglio in parallelo anche le altre conv1D con le varie dimensioni per i filtri,
    model_conv.add(MaxPooling1D())
    model_conv.add(LSTM(128))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model_conv.summary()
    return model_conv


def create_CNN_multifilter_LSTM_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))
    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix], trainable=False)(deep_inputs)  # line A

    def convolution_and_max_pooling(input_layer):
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(0.50)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(0.50)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(0.50)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(0.50)(conv4)
        maxpool1 = MaxPooling1D(pool_size=48)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=47)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=46)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=45)(dropout_4)
        return (maxpool1,maxpool2,maxpool3,maxpool4)

    max_pool_emb = convolution_and_max_pooling(embedding)

    cc1 = concatenate([max_pool_emb[0], max_pool_emb[1],max_pool_emb[2],max_pool_emb[3]], axis=2)
    lstm = LSTM(300)(cc1)
    output = Dense(1, activation="sigmoid")(lstm)
    model = Model(inputs=deep_inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def create_CNN_LSTM_POS_anagrafic_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_list_len):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))

    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix],
                          trainable=False)(deep_inputs)  # line A

    pos_tagging = Input(shape=(pos_tag_list_len,1))

    other_features = Input(shape=(2,1))

    dense_1 = Dense(4, activation="sigmoid")(other_features)
    dense_2 = Dense(8, activation="sigmoid")(dense_1)
    dense_3 = Dense(16, activation="sigmoid")(dense_2)

    dropout_rate = 0.5

    def convolution_and_max_pooling(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=48)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=47)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=46)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=45)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    def convolution_and_max_pooling2(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=33)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=32)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=31)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=30)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)
    max_pool_emb = convolution_and_max_pooling(embedding)
    max_pool_pos = convolution_and_max_pooling2(pos_tagging)

    cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
                      axis=2)
    lstm = LSTM(300)(cc1)
    flat_classifier = Flatten()(dense_3)
    concatenation_layer = concatenate([lstm, flat_classifier ])
  #  output = Dense(1, activation="sigmoid")(concatenation_layer)
    output = Dense(1, activation="linear")(concatenation_layer)
    model = Model(inputs=[deep_inputs, pos_tagging,other_features], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def create_CNN_LSTM_POS_model_attention(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_list_len):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))

    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix],
                          trainable=False)(deep_inputs)  # line A

    pos_tagging = Input(shape=(pos_tag_list_len,1))


    dropout_rate = 0.5

    def convolution_and_max_pooling(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=48)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=47)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=46)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=45)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    def convolution_and_max_pooling2(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=33)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=32)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=31)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=30)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)
    max_pool_emb = convolution_and_max_pooling(embedding)
    max_pool_pos = convolution_and_max_pooling2(pos_tagging)

    cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
                      axis=2)


    lstm = Bidirectional(LSTM(300, return_sequences=True))(cc1)
    attention = AttLayer(300)(lstm)
    #output = Dense(1, activation="sigmoid")(attention)
    output = Dense(1, kernel_initializer='normal', activation="linear")(attention)

    """
    lstm = LSTM(300)(cc1)
    #concatenation_layer = concatenate(lstm)
    output = Dense(1, activation="sigmoid")(lstm)
    """
    model = Model(inputs=[deep_inputs, pos_tagging], outputs=output)

    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    print('Printing Model Summary')
    model.summary()

    return model

def create_CNN_LSTM_POS_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_list_len):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))

    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix],
                          trainable=False)(deep_inputs)  # line A

    pos_tagging = Input(shape=(pos_tag_list_len,1))


    dropout_rate = 0.5

    def convolution_and_max_pooling(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=48)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=47)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=46)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=45)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    def convolution_and_max_pooling2(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=33)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=32)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=31)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=30)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)
    max_pool_emb = convolution_and_max_pooling(embedding)
    max_pool_pos = convolution_and_max_pooling2(pos_tagging)

    cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
                      axis=2)


    lstm = LSTM(300)(cc1)
    #concatenation_layer = concatenate(lstm)
    output = Dense(1, activation="sigmoid")(lstm)

    model = Model(inputs=[deep_inputs, pos_tagging], outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


def create_CNN_LSTM_POS_model_single(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_list_len):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))

    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix],
                          trainable=False)(deep_inputs)  # line A

    pos_tagging = Input(shape=(pos_tag_list_len,1))

    other_features = Input(shape=(2,1))

    dense_1 = Dense(4, activation="sigmoid")(other_features)
    dense_2 = Dense(8, activation="sigmoid")(dense_1)
    dense_3 = Dense(16, activation="sigmoid")(dense_2)

    dropout_rate = 0.5

    def convolution_and_max_pooling(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=48)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=47)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=46)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=45)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    def convolution_and_max_pooling2(input_layer):
        print(input_layer)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = MaxPooling1D(pool_size=33)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=32)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=31)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=30)(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)
    max_pool_emb = convolution_and_max_pooling(embedding)
    max_pool_pos = convolution_and_max_pooling2(pos_tagging)

    cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
                      axis=2)
    lstm = LSTM(300)(cc1)
    flat_classifier = Flatten()(dense_3)
    concatenation_layer = concatenate([lstm, flat_classifier ])
    output = Dense(1, activation="sigmoid")(concatenation_layer)
    model = Model(inputs=[deep_inputs, pos_tagging,other_features], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model



#Dataset loading
# complete interview dataset
#df = pd.read_pickle('../data/pitt_dataframe.pickle')
# single utterances dataset
#df = pd.read_pickle('../data/pitt_complete_dataframe_single_utterances.pickle')
#df = pd.read_pickle('../data/pitt_single_utterances_sex_age.pickle')
print(os.getcwd())
df = pd.read_excel('../data/full_conversation__mmse.xlsx',sheet_name='Sheet1')
#print(df.head())
numeric_label = []
for index, row in df.iterrows():
    numeric_label.append(row['mmse']/30.0)
    #numeric_label.append(row['mmse'] )

# for string in df.label:
#     if string == 'Dementia':
#         numeric_label.append(1)
#     if string == 'Control':
#         numeric_label.append(0)

result_list = []
for seed in [4,10,95]:
    result_dictionary = {}
    X_train, X_test, y_train, y_test = train_test_split(df.text, np.array(numeric_label), test_size=0.1, random_state=seed)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train, test_size=0.1,random_state=seed)

    ## TODO: questa parte è per il single utterances
    #train_set, validation_set, test_set = train_test_split_same_patient(df,0.8,0.1,0.1)

    #X_train = train_set.X
    #y_train = train_set.y

    #X_test = test_set.X
    #y_test = test_set.y

    #Tokenize and crete sequence.
    ### Create sequence
    vocabulary_size = 30000
    sequence_len = 50
    EMBEDDING_SIZE = 300

    pos_tag_len = len(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                         'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                         'VBZ', 'WDT', 'WP', 'WP$', 'WRB'])

    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(X_train)

    def unpack_tup(tuple):
        return tuple[1]

    def compute_pos_tagged_sequence(input_list):
        print('Computing POS ')
        pos_tags_set = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                         'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                         'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

        # two example documents
        sentences_tag = []

        for item in input_list:
            tagged = nltk.pos_tag(item)
            tokenized_list = list(map(unpack_tup,tagged))
            sentences_tag.append(tokenized_list)

        # split documents to tokens
        #tokens_docs = [doc.split(" ") for doc in docs]
        #print('tokens docs: ',tokens_docs)
        # convert list of of token-lists to one flat list of tokens
        # and then create a dictionary that maps word to id of word,
        # like {A: 1, B: 2} here
        word_to_id = {token: idx for idx, token in enumerate(set(pos_tags_set))}
        # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
        token_ids = [[word_to_id[token] if token != "''" and token!="," else 0 for token in tokens_doc] for tokens_doc in sentences_tag]
        # convert list of token-id lists to one-hot representation
        #print(token_ids)
        X = []
        for lst in token_ids:
            one_hot_encoded_list = np.zeros(len(word_to_id))
            for element in lst:
                one_hot_encoded_list[element] +=1
            X.append(one_hot_encoded_list)
        return X

    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_sequences = pad_sequences(train_sequences, maxlen=sequence_len)
    train_tagged = compute_pos_tagged_sequence(X_train)

    #validation_sequences = tokenizer.texts_to_sequences(validation_set.X)
    validation_sequences = tokenizer.texts_to_sequences(X_validation)
    validation_sequences = pad_sequences(validation_sequences, maxlen=sequence_len)
    validation_tagged = compute_pos_tagged_sequence(X_validation)
    #validation_tagged = compute_pos_tagged_sequence(validation_set.X)

    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences = pad_sequences(test_sequences, maxlen=sequence_len)
    test_tagged = compute_pos_tagged_sequence(X_test)

    #Word embeddings initialization
    embeddings_index = dict()
    f = open('../glove.6B/glove.6B.'+str(EMBEDDING_SIZE)+'d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_SIZE))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    #Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    tensor_borad = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    optimizer = Adagrad(lr=0.001, epsilon=None, decay=0.0)

    #model = create_LSTM_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE)
    model = create_CNN_LSTM_POS_model_attention(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,pos_tag_len)

    ## Fit train data
    #if not os.path.isfile("model_weights/karnek_weights.h5"):
    if not os.path.isfile("results/model_mmse_weights_without_features_with_normalize.h5"):
        print("[LOG] Training the model...")
        #model.fit(train_sequences, y_train, validation_split=0.111, epochs = 300,callbacks=[early_stopping,tensor_borad])
        #x_input = [train_sequences, np.array(train_tagged).reshape((np.array(train_tagged).shape[0],np.array(train_tagged).shape[1],1)), train_set.other_features.reshape((train_set.other_features.shape[0],train_set.other_features.shape[1],1))]
        x_input = [train_sequences, np.array(train_tagged).reshape((np.array(train_tagged).shape[0],np.array(train_tagged).shape[1],1))]
        #validation_dat = [validation_sequences, np.array(validation_tagged).reshape((np.array(validation_tagged).shape[0],np.array(validation_tagged).shape[1],1)), validation_set.other_features.reshape(validation_set.other_features.shape[0],validation_set.other_features.shape[1],1)]
        validation_dat = [validation_sequences, np.array(validation_tagged).reshape(
            (np.array(validation_tagged).shape[0], np.array(validation_tagged).shape[1], 1))]

        # class_weight = {
        #     0: 1.,
        #     1: 1.,
        # }
        class_weight = class_weight.compute_class_weight('balanced'
                                                          , np.unique(y_train)
                                                          , y_train)
        # with open('results/class_weight_2.pickle', 'wb') as f:
        #     pickle.dump(class_weight, f)
        start = time.time()
        hist =model.fit(x=x_input,y=y_train, validation_data=(validation_dat, y_validation),epochs=300,callbacks=[early_stopping,tensor_borad],verbose=0,class_weight=class_weight)
        #hist = model.fit(x=x_input, y=y_train, validation_data=(validation_dat, y_validation), epochs=300,
        #                 callbacks=[early_stopping, tensor_borad], verbose=0)
       # model.save_weights("../model_weights/karnek_weights.h5")
        end = time.time()
        print("TIME TOOK %3.2f MIN" % ((end - start) / 60))
        model.save_weights("results/model_mmse_weights_without_features_normalize_weighted.h5")
        with open('results/hist_mmse_without_features_with_normalize_weighted.pickle', 'wb') as f:
            pickle.dump(hist.history, f)
        print("[LOG] Saved weights to disk")
    else:
        print("[LOG] Loading weights from disk...")
        #model.load_weights("../model_weights/karnek_weights.h5")
        model.load_weights("results/model_mmse_weights_2.h5")

    #result = model.predict([test_sequences, np.array(test_tagged).reshape((np.array(test_tagged).shape[0],np.array(test_tagged).shape[1],1)), test_set.other_features.reshape(test_set.other_features.shape[0],test_set.other_features.shape[1],1)])
    result = model.predict([test_sequences, np.array(test_tagged).reshape((np.array(test_tagged).shape[0],np.array(test_tagged).shape[1],1))])
    test_MAE= mean_absolute_error(y_test, result)
    print("MAE")
    print(test_MAE)
    print(y_test)
    print(result)

#     y_score = binary_conversion(result,0.5)
#     test_accuracy = accuracy_score(y_test,y_score)
#     test_f1 = f1_score(y_test,y_score)
#     from sklearn.metrics import confusion_matrix,precision_score,recall_score
#     confusion_matrix = confusion_matrix(y_test,y_score)
#     precision = precision_score(y_test,y_score)
#     recall = recall_score(y_test,y_score)
#     print("Test accuracy: {}, Test F1 score: {}, with classification threshold 0.5".format(test_accuracy,test_f1))
#     print(confusion_matrix.ravel())
#     fpr, tpr, _ = roc_curve(y_test, result)
#     roc_auc = auc(fpr, tpr)
#
#     result_dictionary = {'Test Accuracy':test_accuracy,'Test F1':test_f1, 'Precision': precision, 'Recall': recall, 'Confusion': confusion_matrix, 'AUC':roc_auc}
#     result_list.append(result_dictionary)
#     print("Precision: {}, Recall: {}, AUC: {}".format(precision,recall,roc_auc))
#
#     unique, counts = np.unique(y_test, return_counts=True)
#     count_dictionary = dict(zip(unique, counts))
#     print("Situa : {}".format(count_dictionary))
#     # Compute the ROC curve for the classifier
#
#     print()
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic CNN-LSTM Karnekal')
#     plt.legend(loc="lower right")
#     plt.savefig('roc_curve.png')
#
# print(result_list)
# df = pd.DataFrame(result_list)
dictionary={'test':[],'result':[]}
dictionary['test'].extend(y_test)
dictionary['result'].extend(result)
df_result = pd.DataFrame(dictionary)

# with open('results/original_model_000.pickle', 'wb') as f:
#     pickle.dump(df, f)
with open('results/model_mmse_weights_without_features_with_normalize_weighted.pickle', 'wb') as f:
     pickle.dump(df_result, f)

