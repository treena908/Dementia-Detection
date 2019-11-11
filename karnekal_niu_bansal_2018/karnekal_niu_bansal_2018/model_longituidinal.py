import os
import pickle
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,LeaveOneOut

from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score,mean_absolute_error
import sys, time, os, warnings
from sklearn.utils import class_weight
TRAIN_MODEL = True

### Create sequence
vocabulary_size = 30000
sequence_len = 73
EMBEDDING_SIZE = 300

from keras import Input, Model, initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, Concatenate, Bidirectional, Layer, TimeDistributed, Lambda

from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,TensorBoard
from keras.optimizers import Adagrad
from keras import backend as K
class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None
def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean

def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return (shape[0], shape[2])
### Attention Layer, a contribution from: https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
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
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def lambda_mask_average(x,mask=None):
    return K.batch_dot(x,mask,axes=1) / K.sum(mask, axis=-1, keepdims=True)
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


## The following function is used to create the CNN-LSTM model with attention mechanism and handcrafted features dense layers
## at the end.
def create_longitudinal_CNN_LSTM_POS_model_attention(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,
                                                     pos_tag_list_len, len_features,time_step):
    max_seq_length = sequence_len
    deep_inputs=Input(shape=(time_step,sequence_len),name="deep_input")



    deep_input_1=Lambda(lambda x:x[:,0])(deep_inputs)
    deep_input_2 = Lambda(lambda x: x[:, 1])(deep_inputs)
    print("deep input new")
    print(deep_input_1.shape)
    print(deep_input_2.shape)
    embedding_1 = Embedding(vocabulary_size, EMBEDDING_SIZE,  weights=[embedding_matrix],
                          trainable=False)(deep_input_1)  # line A
    embedding_2 = Embedding(vocabulary_size, EMBEDDING_SIZE,  weights=[embedding_matrix],
                            trainable=False)(deep_input_2)  # line A

    print("before concat")
    print(embedding_1.shape)
    print(embedding_2.shape)
    embedding_1_new=Lambda(lambda x:x[:,None,:,:])(embedding_1)
    embedding_2_new = Lambda(lambda x: x[:, None, :, :])(embedding_2)
    print("reshape")
    print(embedding_1_new.shape)
    print(embedding_2_new.shape)
    concatenate_embedding = Concatenate(axis=1)([embedding_1_new, embedding_2_new] )
    print("after concatenate ")
    print(concatenate_embedding.shape)




    pos_tagging = Input(shape=(time_step,pos_tag_list_len,1),name="pos_tag_feature")

    other_features = Input(shape=(time_step,len_features,1 ),name="other_feature")

    # dense_1 = Dense(16, activation="sigmoid")(other_features)
    # dense_2 = Dense(8, activation="sigmoid")(dense_1)
    # dense_3 = Dense(4, activation="sigmoid")(dense_2)
    dense_1 = TimeDistributed(Dense(128, kernel_initializer='normal',activation='relu'))(other_features)
    dense_2 = TimeDistributed(Dense(64, kernel_initializer='normal',activation='relu'))(dense_1)
    dense_3 = TimeDistributed(Dense(32, kernel_initializer='normal',activation='relu'))(dense_2)
    dropout_rate = 0.5

    def convolution_and_max_pooling(input_layer):
        print(input_layer)
        conv1 = TimeDistributed(Conv1D(100, (3), activation='relu'))(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = TimeDistributed(Conv1D(100, (4), activation='relu'))(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = TimeDistributed(Conv1D(100, (5), activation='relu'))(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = TimeDistributed(Conv1D(100, (6), activation='relu'))(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = TimeDistributed(MaxPooling1D(pool_size=sequence_len-2))(dropout_1)
        maxpool2 = TimeDistributed(MaxPooling1D(pool_size=sequence_len-3))(dropout_2)
        maxpool3 = TimeDistributed(MaxPooling1D(pool_size=sequence_len-4))(dropout_3)
        maxpool4 = TimeDistributed(MaxPooling1D(pool_size=sequence_len-5))(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    def convolution_and_max_pooling2(input_layer):
        print(input_layer)
        conv1 = TimeDistributed(Conv1D(100, (3), activation='relu'))(input_layer)
        dropout_1 = Dropout(dropout_rate)(conv1)
        conv2 = TimeDistributed(Conv1D(100, (4), activation='relu'))(input_layer)
        dropout_2 = Dropout(dropout_rate)(conv2)
        conv3 = TimeDistributed(Conv1D(100, (5), activation='relu'))(input_layer)
        dropout_3 = Dropout(dropout_rate)(conv3)
        conv4 = TimeDistributed(Conv1D(100, (6), activation='relu'))(input_layer)
        dropout_4 = Dropout(dropout_rate)(conv4)
        maxpool1 = TimeDistributed(MaxPooling1D(pool_size=33))(dropout_1)
        maxpool2 = TimeDistributed(MaxPooling1D(pool_size=32))(dropout_2)
        maxpool3 = TimeDistributed(MaxPooling1D(pool_size=31))(dropout_3)
        maxpool4 = TimeDistributed(MaxPooling1D(pool_size=30))(dropout_4)
        return (maxpool1, maxpool2, maxpool3, maxpool4)

    # news_words_concat = concatenate(embeddings, axis=-1)
    # news_words_transformation = TimeDistributed(Dense(20))(news_words_concat)
    max_pool_emb = convolution_and_max_pooling(concatenate_embedding)

    max_pool_pos = convolution_and_max_pooling2(pos_tagging)
    print("after conv")
    print(max_pool_emb[0].shape)
    print(max_pool_pos[0].shape)
    cc1 = Concatenate(axis=3)([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]])
    print("after concat all")
    print(cc1.shape)
    print("before flatten dense3")
    print(dense_3.shape)
    # ccl_flatten=TimeDistributed(Flatten())(cc1)
    # print("after flatten")
    # print(ccl_flatten.shape)
    # ## New concat
    # # concatenated_inputs = concatenate([pos_tagging, news_words_transformation], axis=-1)
    # # concatenated_inputs = concatenate([pos_tagging] + embeddings, axis=-1)
    # # print(concatenated_inputs.shape)
    # # max_pool_emb = convolution_and_max_pooling(concatenated_inputs)
    # # max_pool_emb = convolution_and_max_pooling(embeddings)
    # # max_pool_pos = convolution_and_max_pooling2(pos_tagging)
    #
    # # cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
    # #                    max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
    # #                   axis=2)
    # # cc1 = concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3]],axis=2)
    lstm = TimeDistributed(Bidirectional(LSTM(300, return_sequences=True)))(cc1)

    print("after lstm")
    print(lstm.shape)

    # lstm = Bidirectional(LSTM(300, return_sequences=True))(cc1)
    attention = TimeDistributed(AttLayer(300))(lstm)
    print("after attention")
    print(attention.shape)

    flat_classifier = TimeDistributed(Flatten())(dense_3)
    print("afte flatten dense3")
    print(flat_classifier.shape)
    concatenation_layer = Concatenate(axis=2)([attention, flat_classifier])
    # print("afte final concat")
    # print(concatenation_layer.shape)
    # #output = Dense(1, activation="sigmoid")(concatenation_layer)
    # # output = Dense(1, kernel_initializer='normal', activation="linear")(concatenation_layer)
    # output = TimeDistributed(Dense(2, kernel_initializer='normal', activation="linear"))(concatenation_layer)
    output = TimeDistributed(Dense(1, kernel_initializer='normal', activation="linear"))(concatenation_layer)
    print("output 1 shape")
    print(output.shape)
    output=TimeDistributed(Flatten())(output)
    print("output 2 shape")
    print(output.shape)
    #concatenation_layer = concatenate(lstm)
    # model = Model(inputs=[deep_inputs, pos_tagging,other_features], outputs=output)
    model = Model(inputs=[deep_inputs, pos_tagging,other_features], outputs=output)
    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model.summary()
    return model


def create_CNN_LSTM_POS_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE,
                                        pos_tag_list_len, len_features):
    max_seq_length = sequence_len
    deep_inputs = Input(shape=(max_seq_length,))

    embedding = Embedding(vocabulary_size, EMBEDDING_SIZE, input_length=sequence_len, weights=[embedding_matrix],
                          trainable=False)(deep_inputs)  # line A

    pos_tagging = Input(shape=(pos_tag_list_len, 1))

    other_features = Input(shape=(len_features, 1))

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

    cc1 = Concatenate([max_pool_emb[0], max_pool_emb[1], max_pool_emb[2], max_pool_emb[3],
                       max_pool_pos[0], max_pool_pos[1], max_pool_pos[2], max_pool_pos[3]],
                      axis=2)

    lstm = LSTM(300)(cc1)
    flat_classifier = Flatten()(dense_3)
    concatenation_layer = Concatenate([lstm, flat_classifier])
    output = Dense(1, activation="sigmoid")(concatenation_layer)
    # concatenation_layer = concatenate(lstm)
    model = Model(inputs=[deep_inputs, pos_tagging, other_features], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

df = pd.read_pickle('../data/pitt_mmse_features_new.pickle')
# df = pd.read_pickle('../data/pitt_full_interview_features.pickle')
# df = pd.read_excel('../data/pitt_full_mmse_feature_2.xlsx', sheet_name='Sheet1')
# print(list(df))
print(type(df.feature1))
print(len(df.feature2.values))
print(type(df.feature1[0]))

numeric_label = []
for index, row in df.iterrows():
    list_resut=[]
    list_resut.append(row.mmse1/30)
    list_resut.append(row.mmse2/30)
    #numeric_label.append(row['mmse']/30)

    numeric_label.append(list_resut)
# for string in df.label:

# for string in df.label:
#     if string == 'Dementia':
#         numeric_label.append(1)
#     if string == 'Control':
#         numeric_label.append(0)

def manual_features_conversion(manual_features_1,manual_features_2,time_step):


    final_list_1 = []
    final_list_2 = []

    for element in manual_features_1:


        print(element)

        j_list = []
        for j in element:

            # print(type(j))
            if type(j) == float or type(j) == int:

                j_list.append(j)
        final_list_1.append(j_list)

    len_features = len(final_list_1[0])
    for element in manual_features_2:

        print(element)
        j_list = []
        for j in element:

            if type(j) == float or type(j) == int:

                j_list.append(j)

        final_list_2.append(j_list)

    final_list_1=np.array(final_list_1)
    final_list_2 = np.array(final_list_2)
    # print("shape")
    # print(final_list_1.shape)
    # print(final_list_2.shape)

    concat_array=np.concatenate([final_list_1,final_list_2],axis=1)
    # print(concat_array.shape)
    return np.array(concat_array).reshape(len(manual_features_1),time_step,len_features,1)

def unpack_tup(tuple):
    return tuple[1]
def prepare_train_text(df):
    df_train=[]
    for index,row in df.iterrows():
        df_train.append(row.text1)
        df_train.append(row.text2)
    return df_train

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
    token_ids = [[word_to_id[token] if token != "''" and token != ","  else 0 for token in tokens_doc] for tokens_doc in sentences_tag]
    # convert list of token-id lists to one-hot representation
    #print(token_ids)
    X = []
    for lst in token_ids:
        one_hot_encoded_list = np.zeros(len(word_to_id))
        for element in lst:
            one_hot_encoded_list[element] +=1
        X.append(one_hot_encoded_list)
    return X

result_list = {"test":[],"result":[],"MAE":[]}
loo = LeaveOneOut()
y=np.array(numeric_label)
# print("printing y")
# print(y[0])
seed =4
time_step=2
count=0
for train_index, test_index in loo.split(df):

    #print("TRAIN:", train_index, "TEST:", test_index)
    print("iteration count %d out of %d" %(count, len(df)))
    count=count+1
    df_train, df_test = df.loc[df.index.isin(train_index)], df.loc[df.index.isin(test_index)]

    y_train, y_test = y[train_index], y[test_index]

    df_train, df_validation, y_train, y_validation = train_test_split(df_train,y_train, test_size=0.1,random_state=seed)

# for seed in [4,10,95]:
#     df_train, df_test, y_train, y_test = train_test_split(df, np.array(numeric_label), test_size=0.1, random_state=seed)
#     df_train, df_validation, y_train, y_validation = train_test_split(df_train,y_train, test_size=0.1,random_state=seed)

    manual_feat_train = manual_features_conversion(df_train["feature1"].values,df_train["feature2"].values,time_step)




    manual_feat_test = manual_features_conversion(df_test.feature1.values,df_test.feature2.values,time_step)

    manual_feat_validation = manual_features_conversion(df_validation.feature1.values,df_validation.feature2.values,time_step)
    print("manual_shape")
    print(manual_feat_train.shape)
    print(manual_feat_test.shape)
    print(manual_feat_validation.shape)
    y_train=np.array(y_train).reshape(len(df_train),2,1)
    y_test = np.array(y_test).reshape(len(df_test), 2, 1)
    y_validation = np.array(y_validation).reshape(len(df_validation), 2, 1)


    # manual_feat_validation=manual_feat_validation.reshape(1,time_step,sequence_len)

    # len_features = manual_feat_train.shape[1]
    len_features = 17

    text_train = prepare_train_text(df_train)
    text_validation = prepare_train_text(df_validation)
    text_testing = prepare_train_text(df_test)



    pos_tag_len = len(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                         'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                         'VBZ', 'WDT', 'WP', 'WP$', 'WRB'])

    tokenizer = Tokenizer(num_words= vocabulary_size)

    tokenizer.fit_on_texts(text_train)
    print("len %d" % len(text_train))
    print( len(text_train))
    print("count %d"%tokenizer.document_count)

    train_sequences = tokenizer.texts_to_sequences(text_train)
    train_sequences = pad_sequences(train_sequences, maxlen=sequence_len)
    train_tagged = compute_pos_tagged_sequence(text_train)
    print("train_sequence")
    print(np.array(train_sequences).shape)
    train_sequences_new=np.array(train_sequences).reshape(len(df_train),time_step,sequence_len)
    # train_sequences_new=train_sequences_new[0]
    # train_sequences_new=train_sequences_new.reshape(1,time_step,sequence_len)
    # batch_size=train_sequences_new.shape[0]
    # train_sequences_new=[train_sequences_new[:, :, i] for i in range(sequence_len)]
    train_tagged_new = np.array(train_tagged).reshape(len(df_train), time_step, pos_tag_len,1)
    # train_tagged_new=train_tagged_new[0,:,:,:]
    # train_tagged_new=train_tagged_new.reshape(1,time_step,pos_tag_len,1)
    print("tagged_sequence_new")
    print(np.array(train_tagged_new).shape)
    print("y shape")
    print(np.array(y_train).shape)
    print(np.array(y_validation).shape)
    print(np.array(y_test).shape)
    # print(train_sequences_new[0])
    # print(train_sequences_new[0][1])
    # y_train_new=np.array(y_train)[0,:]
    # y_train_new=y_train_new.reshape(1,time_step,1)
    # y_validation_new=np.array(y_validation)[0,:]
    # y_validation_new = y_validation_new.reshape(1,time_step,1)
    # y_test_new=np.array(y_test)[0,:]
    # y_test_new=y_test_new.reshape(1,time_step,1)
    # print("y_train")
    # print(np.array(y_train_new).shape)


    #validation_sequences = tokenizer.texts_to_sequences(validation_set.X)
    validation_sequences = tokenizer.texts_to_sequences(text_validation)
    validation_sequences = pad_sequences(validation_sequences, maxlen=sequence_len)
    validation_tagged = compute_pos_tagged_sequence(text_validation)
    validation_sequences_new = np.array(validation_sequences).reshape(len(df_validation), time_step, sequence_len)
    # validation_sequences_new=validation_sequences_new[0,:,:]
    # validation_sequences_new=validation_sequences_new.reshape(1,time_step,sequence_len)
    # print("validation_sequences_new")
    # print(np.array(validation_sequences_new).shape)


    # validation_sequences_new = [validation_sequences_new[:, :, i] for i in range(sequence_len)]
    validation_tagged_new = np.array(validation_tagged).reshape(len(df_validation), time_step, pos_tag_len,1)
    # validation_tagged_new=validation_tagged_new[0,:,:,:]
    # validation_tagged_new=validation_tagged_new.reshape(1,time_step,pos_tag_len,1)
    #validation_tagged = compute_pos_tagged_sequence(validation_set.X)

    test_sequences = tokenizer.texts_to_sequences(text_testing)
    test_sequences = pad_sequences(test_sequences, maxlen=sequence_len)
    test_tagged = compute_pos_tagged_sequence(text_testing)
    test_sequences_new = np.array(test_sequences).reshape(len(df_test), time_step, sequence_len)

    # test_sequences_new=test_sequences_new[0,:,:]
    # test_sequences_new = test_sequences_new.reshape(1,time_step,sequence_len)
    # test_sequences_new = [test_sequences_new[:, :, i] for i in range(sequence_len)]
    test_tagged_new = np.array(test_tagged).reshape(len(df_test), time_step, pos_tag_len,1)
    # test_tagged_new=test_tagged_new[0,:,:,:]
    # test_tagged_new=test_tagged_new.reshape(1,time_step,pos_tag_len,1)
    # print("test_sequences_new")
    # print(np.array(test_sequences_new).shape)
    #Word embeddings initialization
    embeddings_index = dict()
    f = open('../glove.6B/glove.6B.'+str(EMBEDDING_SIZE)+'d.txt','r',encoding="utf8")
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

    # #Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    tensor_borad = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    optimizer = Adagrad(lr=0.001, epsilon=None, decay=0.0)

    #model = create_LSTM_model(vocabulary_size, sequence_len, embedding_matrix, EMBEDDING_SIZE)
    model = create_longitudinal_CNN_LSTM_POS_model_attention(vocabulary_size, sequence_len, embedding_matrix,
                                                EMBEDDING_SIZE,pos_tag_len,len_features,time_step)

    ## Fit train data
    print("[LOG] Training the model...")

    # train_POS = np.array(train_tagged).reshape((np.array(train_tagged).shape[0],np.array(train_tagged).shape[1],1))


    x_input = [ train_sequences_new,train_tagged_new, manual_feat_train]
    # x_input = [train_sequences_new, train_tagged_new]

    # validation_POS = np.array(validation_tagged).reshape((np.array(validation_tagged).shape[0], np.array(validation_tagged).shape[1], 1))

    validation_dat = [validation_sequences_new, validation_tagged_new, manual_feat_validation]
    # validation_dat = [validation_sequences_new, validation_tagged_new]

    # class_weight = {
    #     0: 6.,
    #     1: 1.,
    # }
    # class_weights = class_weight.compute_class_weight('balanced'
    #                                                  , np.unique(y_train)
    #                                                  , y_train)
    if TRAIN_MODEL:
        #model.fit(x=x_input,y=y_train, validation_data=(validation_dat, y_validation),epochs=300,callbacks=[early_stopping,tensor_borad],verbose=0, class_weight=class_weight)
        start = time.time()
        hist=model.fit(x=x_input, y=y_train, validation_data=(validation_dat, y_validation), epochs=300,
                  callbacks=[early_stopping, tensor_borad], verbose=0)
        end = time.time()
        print("TIME TOOK %3.2f MIN" % ((end - start) / 60))
        model.save_weights("results/model_mmse_long1.h5")
        with open('results/hist_long1.pickle', 'wb') as f:
            pickle.dump(hist.history, f)
        print("[LOG] Saved weights to disk")


    if not TRAIN_MODEL:
        model.load_weights("../model_weights/model_mmse_weights_4.h5")
    #result = model.predict([test_sequences, np.array(test_tagged).reshape((np.array(test_tagged).shape[0],np.array(test_tagged).shape[1],1)), test_set.other_features.reshape(test_set.other_features.shape[0],test_set.other_features.shape[1],1)])
    # test_POS = np.array(test_tagged).reshape((np.array(test_tagged).shape[0],np.array(test_tagged).shape[1],1))

    # result = model.predict([ test_sequences_new,test_POS, manual_feat_test])
    result = model.predict([test_sequences_new,test_tagged_new,manual_feat_test])
    print("test and resut shape")
    print(y_test[0,0,:]*30)
    print(result[0,0,:]*30)
    print(y_test[0, 1, :] * 30)
    print(result[0, 1, :] * 30)
    test_MAE_1 = mean_absolute_error(y_test[0,0,:]*30, result[0,0,:]*30)
    test_MAE_2 = mean_absolute_error(y_test[0, 1, :]*30, result[0, 1,:]*30)
    test_MAE=[test_MAE_1,test_MAE_2]
    print("MAE 1")
    print(test_MAE_1)
    print("MAE 2")
    print(test_MAE_2)
    result_list["MAE"].append(test_MAE)
    result_list["test"].append([y_test[0,0,0]*30,y_test[0,1,0]*30])
    result_list["result"].append([result[0,0,0]*30,result[0,1,0]*30])



    with open("results/test_value_long1", "a", encoding="utf8") as test:
        test.write(str(y_test[0,0,0]*30)+", "+str(y_test[0,1,0]*30)+"\n")
    with open("results/result_value_long1", "a", encoding="utf8") as res:
        res.write(str(result[0,0,0]*30)+", "+str(result[0,1,0]*30)+"\n")
    with open("results/mae_value_long1", "a", encoding="utf8") as mae:
        mae.write(str( test_MAE_1)+",  "+str(test_MAE_2)+"\n")
    if count==50:
        break
#     y_score = binary_conversion(result,0.5)
#     test_accuracy = accuracy_score(y_test,y_score)
#     test_f1 = f1_score(y_test,y_score)
#     from sklearn.metrics import confusion_matrix,precision_score,recall_score
#     confusion_matrix = confusion_matrix(y_test,y_score)
#     precision = precision_score(y_test,y_score)
#
#     ## Results printing
#
#     for i in range(len(y_test)):
#         if y_test[i] == 0 and y_score[i] == 1:
#             print('False Positive')
#             print(text_testing.iloc[i])
#             #print(tokenizer.sequences_to_texts(test_sequences[i,:]))
#
#         if y_test[i] == 1 and y_score[i] == 1:
#             print('True Positive')
#             print(text_testing.iloc[i])
#             #print(tokenizer.sequences_to_texts(test_sequences[i,:]))
#
#         if y_test[i] == 0 and y_score[i] == 0:
#             print('True Negative')
#             print(text_testing.iloc[i])
#
#         if y_test[i] == 1 and y_score[i] == 0:
#             print('False Negative')
#             print(text_testing.iloc[i])
#
#
#     recall = recall_score(y_test,y_score)
#     print("Test accuracy: {}, Test F1 score: {}, with classification threshold 0.5".format(test_accuracy,test_f1))
#     print(confusion_matrix.ravel())
#     fpr, tpr, _ = roc_curve(y_test, result)
#     roc_auc = auc(fpr, tpr)
#     print("Precision: {}, Recall: {}, AUC: {}".format(precision,recall,roc_auc))
#
#     result_dictionary = {'Test Accuracy': test_accuracy, 'Test F1': test_f1, 'Precision': precision, 'Recall': recall,
#                          'Confusion': confusion_matrix, 'AUC': roc_auc}
#     result_list.append(result_dictionary)
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
df1 = pd.DataFrame(result_list)
#
# import time
# ts = time.time()
# timestamp = str(int(ts))
# with open('results/manual_feature_results'+'111_attention_manual_class_weights6-0'+'.pickle', 'wb') as f:
#     pickle.dump(df, f)
# dictionary={'test':[],'result':[],"MAE":[]}
# dictionary['test'].extend(y_test)
# dictionary['result'].extend(result)
# dictionary['MAE'].extend(result_list)
# df_result = pd.DataFrame(result_list)
# print(y_test)
# print(result)
# # with open('results/original_model_000.pickle', 'wb') as f:
# #     pickle.dump(df, f)
# # with open('results/mmse_result_extended_feature.pickle', 'wb') as f:
# #      pickle.dump(df_result, f)
#
writer = pd.ExcelWriter("results/mmse_result_long1.xlsx", engine='xlsxwriter')

df1.to_excel(writer, sheet_name='Sheet1', columns=['test','result','MAE'])
writer.save()