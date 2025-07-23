import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Load your personal info
with open('my_info.txt', 'r', encoding='utf-8') as file:
    personal_info = file.read()

# Streamlit UI
st.title("Ask Mohammed's Local AI Chatbot (LLaMA 3)")

user_question = st.text_input("Ask me anything about Mohammed:")

if user_question:
    # Stricter prompt: answer ONLY from given info, else say "I don't know."
    template = """
You are an AI assistant. ONLY answer based on the information below. If the answer is not found in this information, say: "I don't know."

INFORMATION:
{personal_info}

QUESTION:
{question}

ANSWER:
"""
    prompt = PromptTemplate.from_template(template).format(
        personal_info=personal_info,
        question=user_question
    )

    # Connect to local Ollama model (llama3)
    llm = Ollama(model="llama3")

    # Generate answer
    answer = llm(prompt)

    # Display result
    st.write(answer)
