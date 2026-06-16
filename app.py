import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.write("API KEY:", os.getenv("OPENAI_API_KEY"))
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Credit Card AI Assistant")

st.title("💳 Credit Card AI Assistant")
st.write("Ask questions about credit card policies, fees, and charges.")

# Load and process documents (run once)
@st.cache_resource
def setup_rag():
    docs = []
    files = ["data/doc1.pdf", "data/doc2.pdf", "data/doc3.pdf"]

    for file in files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

    retriever = vectorstore.as_retriever()
    return retriever, llm
def ask_question(query, retriever):
    docs = retriever.invoke(query)


    context = ""
    for doc in docs[:3]:
        context += doc.page_content + "\n\n"

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content, docs


    return retriever, llm


retriever, llm = setup_rag()


# User input
query = st.text_input("💬 Ask your question:")

# Sample questions
st.write("### Try asking:")
st.write("- What are late payment charges?")
st.write("- What happens if I miss a payment?")
st.write("- How is interest calculated?")

if query:
    with st.spinner("Thinking..."):
        answer, docs = ask_question(query, retriever)


        st.write("### ✅ Answer:")
        st.write(answer)

        st.write("### 📄 Source:")
        for doc in docs[:2]:
            st.write(doc.page_content[:300])
            st.write("---")
