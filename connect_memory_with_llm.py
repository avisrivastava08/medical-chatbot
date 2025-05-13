import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
# from langchain.core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from huggingface_hub import login

login(token="Your Hugging Face token")


# Step1 : Setup LLm (mistral with huggingface)
# import os
HF_TOKEN = os.environ.get("HF_TOKEN")  # Returns None if the variable is not set

HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

# Step 2 :" Connect the LLM with FAISS and create chain"
DB_FAISS_PATH = "vectorstore/db_faiss"
custom_prompt_template = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please."""

def set_custom_prompt(custom_propmt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )
    return prompt

# Load Database from FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)

# Create a retriever from the database(creating QA chain)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGING_FACE_REPO_ID),
    chain_type="stuff", 
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("Response: ", response['result'])
# print("Source Documents: ", response['source_documents'])
