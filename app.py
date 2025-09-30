import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        # Try to find the new punkt_tab tokenizer first
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            # If punkt_tab not found, download it
            nltk.download('punkt_tab')
        except:
            # Fallback to traditional punkt if punkt_tab fails
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Download NLTK data when app starts
download_nltk_data()

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    
    # Try both tokenizers with fallback
    try:
        # First try the new punkt_tab tokenizer
        text = nltk.word_tokenize(text)
    except LookupError:
        try:
            # Fallback to traditional punkt
            nltk.data.find('tokenizers/punkt')
            text = nltk.word_tokenize(text)
        except LookupError:
            # Final fallback - simple split if NLTK tokenizers fail
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

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model1.pkl','rb'))
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
            result = model.predict(vector_input)[0]
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