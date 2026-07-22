import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


for x in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(x)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


st.set_page_config(
    page_title="NLP Pipeline Analyzer",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1F4E79;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#666666;
    margin-bottom:25px;
}

.info-card{
    background:#ffffff;
    border:1px solid #d9d9d9;
    border-radius:12px;
    padding:18px;
    margin-bottom:25px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.info-card p{
    margin:6px 0;
    font-size:16px;
}

.stTextArea textarea{
    border-radius:10px;
    border:2px solid #1F4E79;
    font-size:16px;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:10px;
    font-size:18px;
    font-weight:600;
}

.result-box{
    background:white;
    border-left:5px solid #1F4E79;
    border-radius:8px;
    padding:15px;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
<p><b>Name :</b> Prathmesh Gumgaonkar</p>
<p><b>Batch :</b> B2</p>
<p><b>Date :</b> 22/07/2026</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
'<div class="title">NLP Pipeline Analyzer</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Natural Language Processing using NLTK</div>',
unsafe_allow_html=True
)

st.markdown("---")

text = st.text_area(
    "Enter Text",
    "Natural Language Processing (NLP) is a field of Artificial Intelligence that enables computers to understand and process human language."
)

if st.button("Analyze"):

    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    filtered = [
        w for w in words
        if w.isalpha() and w.lower() not in stop_words
    ]

    lemmas = [
        lemmatizer.lemmatize(w)
        for w in filtered
    ]


    st.markdown("### Analysis Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Words", len(words))
    col2.metric("Sentences", len(sentences))
    col3.metric("Filtered Words", len(filtered))

    st.markdown("---")

    st.subheader("Word Tokens")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(words)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Sentence Tokenization")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(sentences)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Stopword Removal")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Lemmatization")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(dict(zip(filtered, lemmas)))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:gray;font-size:15px;">
Developed using Streamlit & NLTK
</div>
""",
unsafe_allow_html=True
)