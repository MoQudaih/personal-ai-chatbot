import streamlit as st
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama


loader = TextLoader("my_info.txt", encoding="utf-8")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(docs, embeddings)


llm = Ollama(model="llama3")


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=False 
)


st.title("Mohammed's Conversational RAG Chatbot")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


query = st.text_input("Ask me anything about Mohammed:")


if query:
  
    result = qa_chain.run({"question": query, "chat_history": st.session_state.chat_history})


    st.session_state.chat_history.append((query, result))


for i, (question, answer) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {question}")
    st.markdown(f"**Bot:** {answer}")
