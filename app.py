import streamlit as st
import dill
import requests
import pandas as pd
import re
from pyvi import ViTokenizer, ViPosTagger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


with open("stopwords/vietnamese-stopwords.txt",encoding='utf-8') as file:
    stopwords = file.readlines()
    stopwords = [word.rstrip() for word in stopwords]

punctuations = '''!()-–=[]{}“”‘’;:'"|\,<>./?@#$%^&*_~'''

special_chars = ['\n', '\t']

regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain
        r'localhost|' # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ip
        r'(?::\d+)?' # port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)   

def tokenize(text):
    tokenized_text = ViPosTagger.postagging(ViTokenizer.tokenize(text))
    return tokenized_text[0]

def is_punctuation(token):
    global punctuations
    return True if token in punctuations else False

def is_special_chars(token):
    global special_chars
    return True if token in special_chars else False

def is_link(token):
    return re.match(regex, token) is not None

def lowercase(token):
    return token.lower()

def is_stopword(token):
    global stopwords
    return True if token in stopwords else False

# ===============================================================
# Process:
# Text -> Tokenize (pyvi) -> Remove punctuations -> Remove special chars 
# -> Remove links -> Lowercase -> Remove stopwords -> Final Tokens
# ===============================================================
def vietnamese_text_preprocessing(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if not is_punctuation(token)]
    tokens = [token for token in tokens if not is_special_chars(token)]
    tokens = [token for token in tokens if not is_link(token)]
    tokens = [lowercase(token) for token in tokens]
    tokens = [token for token in tokens if not is_stopword(token)]
    # return tokens
    return tokens



@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def model_predict(model, text):
    model_pd = pd.DataFrame()
    model_pd["text"] = [text]
    pred_result = model.predict(model_pd)[0]
    return pred_result


def main():
    st.set_page_config(
        page_title="Fake news detector",
        page_icon=":star:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(":newspaper: FAKE NEWS DETECTOR")
    sess = load_session()

    model_dict = {
            "MLP Classifier": "models/mlpclassifier.pkl",
            "Decision Tree": "models/decisiontree.pkl", 
            }
    resources_dir = "Fake-News-Detection"

    for model_name, model_dir in model_dict.items():
        with open(model_dir, "rb") as f:
            model = dill.load(f)
            model_dict[model_name] = model
        
    col1, col2 = st.columns([6, 4])
    with col2:
        st.image(f"myplot.png", width=700)

    with col1:
        model_name = st.selectbox("Choose model", index=0, options=list(model_dict.keys()))

        news = st.text_area("News to predict")
        entered_items = st.empty()

    button = st.button("Predict")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )

    if button:
        with st.spinner("Predicting..."):
            if not len(news):
                entered_items.markdown("In put at least a piece of news")
            else:
                model = model_dict[model_name]
                
                pred = model_predict(model, news)

                if pred == 0:
                    st.markdown("Non fake news")
                else:
                    st.markdown("Fake news")


if __name__ == "__main__":
    main()

