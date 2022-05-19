from gensim.models import Word2Vec
import pandas as pd
from nltk import RegexpTokenizer, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

path = '../data/'
train = pd.read_csv(path + 'dreaddit-train.csv', encoding = "ISO-8859-1")
test = pd.read_csv(path + 'dreaddit-test.csv', encoding = "ISO-8859-1")
full = pd.concat([train, test])

tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}') # remove number and words that length = 1
full['processed_text'] = full['text'].apply(lambda x: tokenizer.tokenize(x.lower()))

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

full['processed_text'] = full['processed_text'].apply(lambda x: remove_stopwords(x))

stemmer = PorterStemmer()

def word_stemmer(text):
    stem_text = [stemmer.stem(i) for i in text]
    return stem_text

full['processed_text'] = full['processed_text'].apply(lambda x: word_stemmer(x))

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

full['processed_text'] = full['processed_text'].apply(lambda x: word_lemmatizer(x))


model = Word2Vec(sentences=full['processed_text'], vector_size=100, window=5, min_count=1, workers=4)
model.save("domain-word2vec.model") #vector = model.wv['computer']