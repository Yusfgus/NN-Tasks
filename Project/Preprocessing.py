import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

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

def preprocess(train_data, test_data, pre_method):
    print('Drop Nan...')
    print(f"\ttrain_data.shape before {train_data.shape}")
    train_data = train_data.dropna(subset=['Discussion'])
    print(f"\ttrain_data.shape after {train_data.shape}")

    print('start preprocessing...')
    train_Discussion_preprocessed = [preprocess_text(discussion, pre_method) for discussion in train_data['Discussion']]
    test_Discussion_preprocessed = [preprocess_text(discussion, pre_method) for discussion in test_data['Discussion']]

    print('TF-IDF...')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_Discussion_preprocessed)

    X_train = vectorizer.transform(train_Discussion_preprocessed)
    X_test = vectorizer.transform(test_Discussion_preprocessed)

    print('Encoding Y_train...')
    Y_train = train_data['Category'].map(category_encoding)

    return X_train, Y_train, X_test