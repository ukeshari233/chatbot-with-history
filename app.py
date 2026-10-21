
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
[data-testid="stChatInput"] {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# Model
@st.cache_resource
def get_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.4,
        max_tokens=150
    )

model = get_model()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a helpful AI assistant. "
                    "Give clear and useful answers."
        )
    ]

# Sidebar
with st.sidebar:
    st.title("💬 Groq AI")
    st.caption("Powered by LangChain + Groq")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. "
                        "Give clear and useful answers."
            )
        ]
        st.rerun()

    st.divider()
    st.caption("Model: GPT-OSS 20B")
    st.caption("Built with Streamlit")

# Header
st.title("Groq AI Assistant")
st.caption("How can I help you today?")

# Display conversation
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "user" if isinstance(message, HumanMessage) else "assistant"

    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Message Groq AI..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = model.invoke(
                    st.session_state.messages
                )

            answer = response.content
            st.markdown(answer)

            st.session_state.messages.append(
                AIMessage(content=answer)
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.caption(
                "Check your GROQ_API_KEY, internet connection, "
                "model availability, and API limits."
            )
            # Remove the failed user message so it
            # is not included in the next request.
            st.session_state.messages.pop()
