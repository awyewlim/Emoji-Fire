import tensorflow as tf
import numpy as np
from emo_utils import *

model = tf.keras.models.load_model('model')
X_train, Y_train = read_csv('data/train.csv')
X_test, Y_test = read_csv('data/test.csv')

maxLen = len(max(X_train, key=len).split())
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('assets/glove.6B.50d.txt')

myInput = input("Please enter a sentence that you want to add emoji to ('exit' to exit): ")
while True:
    if myInput == 'exit':
        break
    x = np.array([myInput])
    x_indices = sentences_to_indices(x, word_to_index, maxLen)
    print(x[0] +' '+  label_to_emoji(np.argmax(model.predict(x_indices))))
    print()
    myInput = input("Please enter a sentence that you want to add emoji to: ")

print('Thank you for using!')