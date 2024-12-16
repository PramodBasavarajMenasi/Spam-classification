import streamlit as stm
import pickle
import string
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(st.stem(i))

    return " ".join(y)


# Load the vectorizer
with open('vectorizer2.pkl', 'rb') as f:
    tfidf = pickle.load(f)
    
model = pickle.load(open('model2.pkl','rb'))

stm.title(" Email / SMS Spam Classifier ")
input_sms = stm.text_input("Enter the messages")

if stm.button("Predict"):
    transformed_sms = transform_text(input_sms)

    vectorized_text = tfidf.transform([transformed_sms])

    result = model.predict(vectorized_text)[0]

    if result == 1:
        stm.header("Spam")
    else:
        stm.header("Not spam")
