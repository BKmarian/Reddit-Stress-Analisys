from nltk import WordNetLemmatizer, PorterStemmer, RegexpTokenizer
from nltk.corpus import stopwords

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import neattext
import re
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

sns.set()
warnings.filterwarnings('ignore')

# path = '/content/Insight_Stress_Analysis/data/'
path = '../data/'
train = pd.read_csv(path + 'dreaddit-train.csv', encoding = "ISO-8859-1")
test = pd.read_csv(path + 'dreaddit-test.csv', encoding = "ISO-8859-1")

def removeWordsWithNumbers(text):
    return re.sub(r'\S*\d\S*', '', text).strip()

def clean_data(content):
    content = str(content)
    content = removeWordsWithNumbers(content)
    docx = neattext.TextFrame(text=content)
    docx.remove_emojis()
    docx.remove_html_tags()
    docx.remove_puncts()
    docx.remove_urls()
    docx.remove_stopwords(lang='en')
    docx.remove_special_characters()
    docx.fix_contractions()
    docx.remove_numbers()
    content = nlp(content.strip())
    return " ".join([token.lemma_ for token in content])


### Tokenization & Remove punctuations

train['processed_text'] = train['text'].apply(lambda x: clean_data(x.lower()))
test['processed_text'] = test['text'].apply(lambda x: clean_data(x.lower()))

train.to_csv(path + 'dreaddit-train-processed.csv', encoding = "ISO-8859-1")
test.to_csv(path + 'dreaddit-test-processed.csv', encoding = "ISO-8859-1")
