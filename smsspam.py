import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

@st.cache_data
def load_train():
    data = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'msg'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    ham = data[data['label'] == 0]
    spam = data[data['label'] == 1]
    spam_up = resample(spam, replace=True, n_samples=len(ham), random_state=42)
    data_bal = pd.concat([ham, spam_up])

    X_train, X_test, y_train, y_test = train_test_split(
        data_bal['msg'], data_bal['label'], test_size=0.2, random_state=42)

    vec = CountVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, vec, acc

model, vec, acc = load_train()

st.title("SMS Spam Detection System")
st.markdown("Welcome to the SMS Spam Classifier. Enter your message below to check if it's spam or safe.")

msg = st.text_area("Enter your message:", height=150)

if st.button("Classify Message"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        msg_vec = vec.transform([msg])
        pred = model.predict(msg_vec)[0]
        res = "This message is likely SPAM." if pred == 1 else "This message is likely SAFE."
        st.subheader("Prediction:")
        st.success(res)

st.caption(f"Model Accuracy: {acc * 100:.2f}%")


    
   

