import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
#from flask import Flask, request, jsonify, render_template
import streamlit as st
import spacy
from build_library.utils import customNlp
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')


file = open('pipeline.pickle', 'rb')

# dump information to that file
mytest = pickle.load(file)

# close the file
file.close()


def welcome():
    return "Welcome All"


def predictionFunction(inputText):
    result = mytest[1].predict([inputText])[0]
    return result


def main():
    st.title("Sentimental Prediction")

    st.text('This app is an sentemantal analysis project. For detail information follow the below link.')
    link = '[GitHub](https://github.com/serdarkuyuk/spam2finder/blob/master/spamClassifierTfIDF.ipynb)'
    st.markdown(link, unsafe_allow_html=True)
    #st.text_input("Text-Email", "Type your email here.")
    st.header('Try this two sentence')
    st.text('1. I feel great.')
    st.text('2. I am not felling, ok')

    inputText = st.text_area("Your input text", "Type your sentence here.")

    if st.button("Predict"):
        output = predictionFunction(inputText)
        if output == 1.0:
            st.success('This text has positive feeling')
            # st.text(output)
            #st.markdown('This email seems to be ** _normal_ email **')
        else:
            st.error('This text has negative feeling')
            # st.text(output)
        return output

    # return render_template('index.html')
if __name__ == '__main__':
    result = main()
