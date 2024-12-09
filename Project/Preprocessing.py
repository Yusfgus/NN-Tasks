import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, LayerNormalization, MultiHeadAttention # type: ignore
from tensorflow.keras.models import Model # type: ignore

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

category_encoding = {
    "Politics":0,
    "Sports":1,
    "Media":2,
    "Market & Economy":3,
    "STEM":4
}

def lemmatization(tokens, tagging=False):
    lemmatized_tokens = []
    if tagging:
        tagged_tokens = nltk.pos_tag(tokens)
        for word, tag in tagged_tokens:
            if tag.startswith('NN'):    # Nouns
                lemma = lemmatizer.lemmatize(word, pos='n')
            elif tag.startswith('VB'):  # Verbs
                lemma = lemmatizer.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):  # Adjectives
                lemma = lemmatizer.lemmatize(word, pos='a')
            else:
                lemma = lemmatizer.lemmatize(word)
            lemmatized_tokens.append(lemma)
    else:
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmatized_tokens

def stemming(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def cleanText(tokens, choice=2):
    cleaned_text = ""

    if choice == 1: # Steaming only
        stemmed_tokens = stemming(tokens)
        cleaned_text = ' '.join(stemmed_tokens)

    elif choice == 2: # Lemmatization without tagging
        lemmatized_tokens = lemmatization(tokens, False)
        cleaned_text = ' '.join(lemmatized_tokens)

    elif choice == 3: # Lemmatization with tagging
        lemmatized_tokens = lemmatization(tokens, True)
        cleaned_text = ' '.join(lemmatized_tokens)

    elif choice == 4: # Lemmatization without tagging and Steaming
        lemmatized_tokens = lemmatization(tokens, False)
        stemmed_tokens = stemming(lemmatized_tokens)
        cleaned_text = ' '.join(stemmed_tokens)

    elif choice == 5: # Lemmatization with tagging and Steaming
        lemmatized_tokens = lemmatization(tokens, True)
        stemmed_tokens = stemming(lemmatized_tokens)
        cleaned_text = ' '.join(stemmed_tokens)


    return cleaned_text

def preprocess_text(text, pre_method=2):
    # Tokenization
    text = text.replace('\\n', ' ')
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    
    cleaned_text = cleanText(tokens, pre_method)
    return cleaned_text


def load_glove_embeddings(glove_file, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def preprocess(train_data, test_data, pre_method, fx_opt, glove_path=None, embedding_dim=100):
    print('Drop Nan...')
    print(f"\ttrain_data.shape before {train_data.shape}")
    train_data = train_data.dropna(subset=['Discussion'])
    print(f"\ttrain_data.shape after {train_data.shape}")

    print('start preprocessing...')
    train_Discussion_preprocessed = [preprocess_text(discussion, pre_method) for discussion in train_data['Discussion']]
    test_Discussion_preprocessed = [preprocess_text(discussion, pre_method) for discussion in test_data['Discussion']]

    print('Encoding Y_train...')
    Y_train = train_data['Category'].map(category_encoding)

    if fx_opt == 3:
        print("Using GloVe embeddings with Transformer model...")

        # Tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_Discussion_preprocessed)

        X_train_seq = tokenizer.texts_to_sequences(train_Discussion_preprocessed)
        X_test_seq = tokenizer.texts_to_sequences(test_Discussion_preprocessed)

         # Set max_sequence_length to the longest sequence in the dataset
        max_sequence_length =100
        
        X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
        
        X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

        # Load GloVe embeddings
        word_index = tokenizer.word_index
        embedding_matrix = load_glove_embeddings(glove_path, word_index, embedding_dim)

        return X_train_padded, Y_train, X_test_padded, embedding_matrix, max_sequence_length


    elif fx_opt == 1:
        print('TF-IDF...')
        vectorizer = TfidfVectorizer()
        vectorizer.fit(train_Discussion_preprocessed)

        X_train = vectorizer.transform(train_Discussion_preprocessed)
        X_test = vectorizer.transform(test_Discussion_preprocessed)

        return X_train, Y_train, X_test

    elif fx_opt == 2:
        # print("Calc unique words...")
        # unique_words = set()
        # for sentence in train_Discussion_preprocessed:
        #     words = sentence.split()  # Split
        #     unique_words.update(words)       # Add words to the set

        # num_unique_words = len(unique_words)
        num_unique_words = 10000
        print("\tNum of Unique words:", num_unique_words)

        print('Tokenizer...')
        tokenizer = Tokenizer(num_words=num_unique_words)  # Set max vocabulary size
        tokenizer.fit_on_texts(train_Discussion_preprocessed)         # Fit tokenizer on training data 

        X_train_seq = tokenizer.texts_to_sequences(train_Discussion_preprocessed)
        X_test_seq = tokenizer.texts_to_sequences(test_Discussion_preprocessed)

        # Pad sequences to ensure uniform length
        max_sequence_length = int(sum(len(s) for s in X_train_seq) / len(X_train_seq))
        # max_sequence_length = max(len(sublist) for sublist in X_train_seq)
        X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

        return X_train_padded, Y_train, X_test_padded, num_unique_words, max_sequence_length