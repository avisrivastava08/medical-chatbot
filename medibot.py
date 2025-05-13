import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Vector database not found at {DB_FAISS_PATH}. Ensure FAISS index is created.")
        return None  # Stop execution properly
    
    try:
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS vector store: {str(e)}")
        return None


def set_custom_prompt():
    """Creates a custom LangChain prompt."""
    template = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])


def load_llm():
    """Loads the Hugging Face model with correct authentication."""
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
    
    if not HF_TOKEN:
        st.error("Hugging Face API token is missing. Please add it in your environment variables.")
        st.stop()
    
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"max_length": 512},
        huggingfacehub_api_token=HF_TOKEN
    )


def main():
    st.title("Ask Chatbot!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Pass your prompt here")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                return
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )
            
            response = qa_chain.invoke(prompt)
            result = response.get("result", "No response received.")
            source_documents = response.get("source_documents", [])
            
            result_to_show = f"{result}\n\n**Source Docs:**\n{str(source_documents)}"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
