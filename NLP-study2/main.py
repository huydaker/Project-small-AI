import numpy as np 
import keras 
from keras.models import Model 
from keras. layers import Input 
from keras.layers import LSTM, Dense 
import pandas as pd
import string
from gensim import corpora
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import os
# Định nghĩa tham số cho quá trình dịch
n_units = 400
epochs = 1000 # số lần mô hình sẽ chạy qua toàn bộ tập dữ liệu huấn luyện 1000 # số lượng mẫu dữ liệu được sử dụng trong mỗi lần cập nhật trọng số của mô hình 
batch_size = 50 # 50
max_pairs = 10000 # số lượng cặp câu được sử dụng trong quá trình đào tạo 10000

def remove_non_ascii (text):
    return ''.join([ word for word in text if ord(word) < 128  ])

def load_data():
    input_characters, output_characters = set(), set()
    input, output = [], []
    sentence_pairs = open ('phython3/deu.txt', encoding='utf-8').read().split('\n')
    for line in sentence_pairs[: min(max_pairs, len(sentence_pairs))-1]:
        _input,_output = line.split('\t')
        output. append (_output) 
        input.append (_input)
        for i in _input:
            if i not in input_characters:
                input_characters.add (i.lower())
        for o in _output:
            if o not in output_characters:
                output_characters.add (o. lower ())
    input_characters = sorted (list (input_characters))
    output_characters = sorted (list(output_characters))
    n_encoder_tokens, n_decoder_tokens = len(
        input_characters), len(output_characters)
    max_encoder_len = max([len(text) for text in input])
    max_decoder_len = max([len(text) for text in output])
    input_dictionary = {word: i for i, word in enumerate(input_characters)}
    output_dictionary = {word: i for i, word in enumerate (output_characters)}
    label_dictionary = {i: word for i, word in enumerate(output_characters)}
    x_encoder = np.zeros ((len(input), max_encoder_len,n_encoder_tokens), dtype=float)
    x_decoder = np. zeros ((len(input), max_decoder_len,n_decoder_tokens), dtype=float)
    y_decoder = np.zeros ((len(input), max_decoder_len,n_decoder_tokens), dtype=float)
    for i, (input, _output) in enumerate(zip(input, output)): 
        for _character, character in enumerate(_input):
            x_encoder [i, _character, input_dictionary[character.lower()]] = 1
        for _character, character in enumerate (_output) :
            x_decoder[i,_character, output_dictionary[character.lower()]] = 1
            if _character > 0:
                y_decoder [i, _character-1,output_dictionary[character.lower()]] = 1
    data = list([x_encoder, x_decoder, y_decoder])
    variables = list([label_dictionary, n_decoder_tokens, n_encoder_tokens])
    return data, variables


def encoder_decoder(n_encoder_tokens, n_decoder_tokens):
    encoder_input = Input(shape=(None, n_encoder_tokens))
    encoder = LSTM(n_units, return_state=True)
    encoder_output, hidden_state, cell_state = encoder (encoder_input)
    encoder_states = [hidden_state, cell_state]
    decoder_input = Input (shape=(None, n_decoder_tokens))
    decoder = LSTM(n_units, return_state=True, return_sequences=True)
    decoder_output, _, _= decoder (decoder_input, initial_state=encoder_states)
    decoder = Dense(n_decoder_tokens, activation='softmax') (decoder_output)
    model = Model([encoder_input, decoder_input], decoder)
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary ()
    return model

def train_encoder_decoder ():
    input_data_objects = load_data()
    x_encoder, x_decoder, y_decoder = input_data_objects [0][0], input_data_objects[0][1], input_data_objects[0][2]
    label_dictionary, n_decoder_tokens = input_data_objects[1][0], input_data_objects[1][1]
    n_encoder_tokens = input_data_objects[1][2]
    seq2seq_model = encoder_decoder(n_encoder_tokens, n_decoder_tokens)
    seq2seq_model.fit([x_encoder, x_decoder], y_decoder,epochs=epochs, batch_size=batch_size, shuffle=True)
    
    input_sentence = input( "Nhập cầu cần dịch: ")
    user_sentence = remove_non_ascii (input_sentence).lower()
    x = x_encoder[0:1]
    print(x)
    y_predict = seq2seq_model.predict([x_encoder[0:1], x_decoder[0:1]])
    print(y_predict)




def readfile ():
    df1 = pd.read_csv("phython3/topic1.txt",delimiter="\t", header=None)
    df2 = pd.read_csv("phython3/topic2.txt",delimiter="\t", header=None)
    frames = [pd.Series (df1[0]) .str.cat (sep=' '), pd.Series(df2[0]).str.cat (sep=' ')]
    return frames

def clean(doc, lemma) :
    exclude = set(string.punctuation)
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop]) 
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# nltk.download('stopwords')

if __name__ == "__main__":
    # encoder_decoder()
    # train_encoder_decoder() 
    frames = readfile()
    stop = set(stopwords.words ('english'))
    lemma = WordNetLemmatizer()
    doc_clean = [clean(doc,lemma).split() for doc in frames]
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    Idamodel = Lda(doc_term_matrix, num_topics=3,id2word=dictionary, passes=50)
    for idx, topic in Idamodel.print_topics(-1):
        print("Topic: {} InWords: {}". format(idx, topic))
        print ("\n")