import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re

# from nltk.corpus import words
# english_words = set(words.words())
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
    
    elif choice == 0: # None
        cleaned_text = ' '.join(tokens)

    return cleaned_text

def preprocess_text(text, pre_method=2):
    # text = text.replace('\\n', ' ')
    text = re.sub(r'[^a-zA-Z0-9\s]|[\\n]', ' ', text) # Remove non-alphanumeric characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    # Tokenization
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

