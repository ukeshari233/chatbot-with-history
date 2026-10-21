
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Groq AI Chat",
    page_icon="💬",
    layout="centered"
)

# API key: Streamlit Cloud Secrets or local .env
try:
    api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.getenv("GROQ_API_KEY", "")

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #212121;
    color: white;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# Model configuration
@st.cache_resource
def get_model(key):
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=key,
        temperature=0.4,
        max_tokens=150
    )

# Stop safely if API key is missing
if not api_key:
    st.error("Groq API key is missing.")
    st.info(
        "Streamlit Cloud: Add GROQ_API_KEY in "
        "App Settings > Secrets."
    )
    st.code('GROQ_API_KEY = "your_groq_api_key"')
    st.stop()

model = get_model(api_key)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "Give clear and useful answers."
            )
        )
    ]

# Sidebar
with st.sidebar:
    st.title("💬 Groq AI")
    st.caption("Powered by LangChain + Groq")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant. "
                    "Give clear and useful answers."
                )
            )
        ]
        st.rerun()

    st.divider()
    st.caption("Model: GPT-OSS 20B")
    st.caption("Built with Streamlit")

# Header
st.title("Groq AI Assistant")
st.caption("How can I help you today?")

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "user" if isinstance(message, HumanMessage) else "assistant"

    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Message Groq AI..."):

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = model.invoke(
                    st.session_state.messages
                )

            answer = response.content

            if not isinstance(answer, str):
                answer = str(answer)

            st.markdown(answer)

            st.session_state.messages.append(
                AIMessage(content=answer)
            )

        except Exception:
            # Remove failed user message
            st.session_state.messages.pop()

            st.error(
                "Unable to generate a response. "
                "Check your Groq API key, model access, "
                "and API limits."
            )
