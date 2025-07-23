import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

# Step 1: Load your personal info text file into memory
loader = TextLoader("my_info.txt", encoding="utf-8")
documents = loader.load()

# Step 2: Split the loaded text into smaller chunks
# This helps the vector search find relevant pieces better
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings (numerical representations) of the text chunks
# Using a pre-trained Sentence Transformer model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 4: Build a FAISS vector store from the embedded chunks
# This is your searchable database of text chunks
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 5: Initialize Ollama local LLaMA 3 language model
llm = Ollama(model="llama3")

# Step 6: Create a conversation memory object
# This will keep track of past interactions in the current chat session
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 7: Set up a conversational retrieval chain that:
# - Uses the LLM (Ollama)
# - Retrieves relevant documents from vectorstore based on the question
# - Maintains conversation context with memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=False  # We don't show source docs to user here
)

# Step 8: Setup Streamlit UI title
st.title("Mohammed's Conversational RAG Chatbot")

# Step 9: Initialize chat history in Streamlit's session state
# Streamlit session state keeps info during the user's interaction
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 10: Text input box for user question
query = st.text_input("Ask me anything about Mohammed:")

# Step 11: When user submits a question...
if query:
    # Run the conversational retrieval chain with current question and chat history
    result = qa_chain.run({"question": query, "chat_history": st.session_state.chat_history})

    # Append this Q&A pair to chat history
    st.session_state.chat_history.append((query, result))

# Step 12: Display the full conversation history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {question}")
    st.markdown(f"**Bot:** {answer}")
