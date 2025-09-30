import pickle
import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def text_transfor(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
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
        y.append(ps.stem(i))
        
    text = y[:]
    y.clear()
    
    return text


tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS-Spam Classifier")


input_msg = st.text_input("Enter the msg")

# preprocess
transfor_msg = " ".join(text_transfor(input_msg))
# vectorize
vector_input = tf.transform([transfor_msg])
# predict
result = model.predict(vector_input)[0]
# display
if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")