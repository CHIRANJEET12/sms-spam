import pickle
import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def text_transfor(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]

    return tokens

@st.cache_resource
def load_model():
    tf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tf, model

tf, model = load_model()

st.title("Email/SMS Spam Classifier")

input_msg = st.text_input("Enter your message:")

if input_msg:
    # Preprocess
    transfor_msg = " ".join(text_transfor(input_msg))

    # Vectorize
    vector_input = tf.transform([transfor_msg])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam 🚫")
    else:
        st.header("Not Spam ✅")
