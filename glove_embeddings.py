import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,f1_score,roc_curve,auc
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,TensorBoard
from keras.optimizers import Adagrad

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

#Dataset loading
df = pd.read_pickle('pitt_dataframe.pickle')

numeric_label = []
for string in df.label:
    if string == 'Dementia':
        numeric_label.append(1)
    if string == 'Control':
        numeric_label.append(0)

X_train, X_test, y_train, y_test = train_test_split(df.text, np.array(numeric_label), test_size=0.2, random_state=42)


#Tokenize and create sequence.
### Create sequence
vocabulary_size = 30000
sequence_len = 250
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
train_sequences = pad_sequences(train_sequences, maxlen=sequence_len)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_sequences = pad_sequences(test_sequences, maxlen=sequence_len)

#Word embeddings initialization
embeddings_index = dict()
f = open('glove.6B/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocabulary_size, 100))
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

## create model
model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 100, input_length=sequence_len, weights=[embedding_matrix], trainable=False))
model_glove.add(LSTM(40))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

## Fit train data
if not os.path.isfile("model_weights/glove_embeddings_classifier_weights.h5"):
    print("[LOG] Training the model...")
    model_glove.fit(train_sequences,y_train , validation_split=0.2, epochs = 300,callbacks=[early_stopping,tensor_borad])
    model_glove.save_weights("model_weights/glove_embeddings_classifier_weights.h5")
    print("[LOG] Saved weights to disk")
else:
    print("[LOG] Loading weights from disk...")
    model_glove.load_weights("model_weights/glove_embeddings_classifier_weights.h5")

result = model_glove.predict(test_sequences)
y_score = binary_conversion(result,0.5)
test_precision = precision_score(y_test,y_score)
test_f1 = f1_score(y_test,y_score)
print("Test precision: {}, Test F1 score: {}, with classification threshold 0.5".format(test_precision,test_f1))

#Compute the ROC curve for the classifier
fpr, tpr, _ = roc_curve(y_test, result)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic GloVe classifier')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

selection_list = []
#Printing some examples of correctly classified patients:
#Here there are some index of samples in the test set that have been correctly classifieds by our model or not (they have been hand picked).
true_positive = 272
true_negative = 140
false_negative = 547
false_positive = 81

print('False Negative: {}'.format(format_example(X_test[false_negative])))
print('True Negative: {}'.format(format_example(X_test[true_negative])))
print('True Positive: {}'.format(format_example(X_test[true_positive])))
print('False Positive: {}'.format(format_example(X_test[false_positive])))

