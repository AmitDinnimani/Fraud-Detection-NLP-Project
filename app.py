from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import helper
import streamlit as st

word2vec_model = Word2Vec.load("your_model.model") 
nn_model = load_model("your_model.h5")

def pipeline(text):

    processed_text = helper.preprocess(text)
    processed_text = helper.lemmatize_text(processed_text)
    processed_text = helper.words_num(processed_text)
    processed_text = pad_sequences([processed_text],maxlen=250,padding='post')
    
    return processed_text

st.title("🔍 Fraud Detection System")
st.write("Enter a text description to check if it's fraudulent or not.")

user_input = st.text_area("Enter text:", placeholder="Type here...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        processed_input = pipeline(user_input)
        prediction = nn_model.predict(processed_input)[0][0]
        # Display Result
        if prediction > 0.5:
            st.error(f"🚨 Prediction: Fraud ({prediction:.2f})")
        else:
            st.success(f"✅ Prediction: Not Fraud ({prediction:.2f})")

