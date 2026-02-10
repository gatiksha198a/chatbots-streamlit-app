import streamlit as st
import pickle
import nltk
import string
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLP data
st.markdown(
    """
    <style>
    .stChatMessage {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD CHATBOT DATA ----------------
with open("chatbot_data.sav", "rb") as file:
    chatbot_text = pickle.load(file)
from nltk.tokenize import sent_tokenize
sent_tokens = sent_tokenize(chatbot_text)


lemmer = nltk.stem.WordNetLemmatizer()

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    return [lemmer.lemmatize(word.lower())
            for word in nltk.word_tokenize(text)
            if word not in string.punctuation]

# ---------------- GREETING LOGIC ----------------
greetings = ("hi", "hello", "hey")
responses = ("Hi ", "Hello ", "Hey there!")

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetings:
            return random.choice(responses)

# ---------------- CHATBOT RESPONSE ----------------
def chatbot_reply(user_input):
    sent_tokens.append(user_input)

    tfidf = TfidfVectorizer(tokenizer=clean_text, stop_words='english')
    matrix = tfidf.fit_transform(sent_tokens)

    similarity = cosine_similarity(matrix[-1], matrix)
    index = similarity.argsort()[0][-2]

    sent_tokens.remove(user_input)
    return sent_tokens[index]

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# Sidebar
st.sidebar.title(" Project Info")
st.sidebar.write("""NLP Based Chatbot
- Python
- NLTK
- Scikit-learn
- Streamlit
""")

st.sidebar.write("Made for learning & resume project")

# Main title
st.title(" Student Chatbot")
st.write("Ask me questions related to programming and data science")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate chatbot reply
    if greeting(user_input):
        bot_reply = greeting(user_input)
    else:
        bot_reply = chatbot_reply(user_input)

    # Show chatbot message
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

