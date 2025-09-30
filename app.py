import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    
    try:
        text = nltk.word_tokenize(text)
    except LookupError:
        try:
            nltk.data.find('tokenizers/punkt')
            text = nltk.word_tokenize(text)
        except LookupError:
            text = text.split()

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
        y.append(ps.stem(i))

    return " ".join(y)

try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model1 = pickle.load(open('model2.pkl','rb'))
except FileNotFoundError:
    st.error("Model files (vectorizer.pkl or model.pkl) not found. Please make sure they are in the same directory.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model1.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
                st.error("This message appears to be spam!")
            else:
                st.header("Not Spam")
                st.success("This message appears to be legitimate!")
                
            # Show processed text for debugging
            with st.expander("Show processed text"):
                st.write(transformed_sms)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")