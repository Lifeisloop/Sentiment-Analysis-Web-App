import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

#load model
model = pickle.load(open('emotion_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

#page config
st.set_page_config(
    page_title = "Sentiment Analysis",
    page_icon = "😊",
    layout = "centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.big-title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 15px;
    border: 2px solid #334155;
    font-size: 18px;
}

.stButton>button {
    width: 100%;
    height: 60px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
}

.result-card {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 30px;
}

.positive {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
}

.negative {
    background: linear-gradient(90deg, #dc2626, #ef4444);
    color: white;
}
</style>
""", unsafe_allow_html=True)

#Title
st.title("Sentiment Analysis Web App")
st.write("Enter text to check sentiment")

#Cleaning function
def clean_text(text):
    return text.lower()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("📝 Enter your text", height=250)

with col2:
    st.info("""
### Features
✅ NLP Text Cleaning  
✅ Stopword Removal  
✅ TF-IDF Vectorization  
✅ Logistic Regression Prediction  
✅ Real-Time Analysis
""")

#Prediction
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)

        emotion_dict = {
            0: ("😡 ANGER", "#ff4b4b"),
            1: ("😨 FEAR", "#9b59b6"),
            2: ("😊 JOY", "#2ecc71"),
            3: ("❤️ LOVE", "#e91e63"),
            4: ("😢 SADNESS", "#f1c40f"),
            5: ("😲 SURPRISE", "#3498db")
        }

        emotion, css_class = emotion_dict[int(prediction[0])]

        st.markdown(
            f'<div class="result-card {css_class}">{emotion}</div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit | NLP Project</center>",
    unsafe_allow_html=True)
