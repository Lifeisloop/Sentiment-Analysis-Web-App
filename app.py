import streamlit as st
import pickle
import string

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

.stButton > button {
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
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    margin-top: 30px;
    color: white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
}

/* Emotion Colors */
.anger {
    background: linear-gradient(90deg, #dc2626, #ef4444);
}

.fear {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
}

.joy {
    background: linear-gradient(90deg, #eab308, #facc15);
}

.love {
    background: linear-gradient(90deg, #db2777, #ec4899);
}

.sadness {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
}

.surprise {
    background: linear-gradient(90deg, #16a34a, #22c55e);
}

</style>
""", unsafe_allow_html=True)

#Title
st.title("Sentiment Analysis Web App")
st.write("Enter text to check sentiment")

#Cleaning function matching the training notebook
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = ''.join([c for c in text if not c.isdigit()])
    # Remove emojis (keep ASCII only)
    text = ''.join([c for c in text if c.isascii()])
    return text

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
            0: ("😡 ANGER", "anger"),
            1: ("😨 FEAR", "fear"),
            2: ("😊 JOY", "joy"),
            3: ("❤️ LOVE", "love"),
            4: ("😢 SADNESS", "sadness"),
            5: ("😲 SURPRISE", "surprise")
        }

        emotion, css_class = emotion_dict[prediction[0]]

        st.markdown(
        f'<div class="result-card {css_class}">{emotion}</div>',
        unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit | NLP Project</center>",
    unsafe_allow_html=True)
