import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

class PDFQuery:
    def __init__(self, openai_api_key=None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="&&&&&")
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, folder_path: os.PathLike) -> None:
        documents = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                loader = PyPDFium2Loader(file_path)
                doc = loader.load()
                documents.extend(doc)
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever()
        self.chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")

    def forget(self) -> None:
        self.db = None
        self.chain = None

def main():
    st.title("PDF Query")

    openai_api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
    pdf_query = PDFQuery(openai_api_key=openai_api_key)

    folder_path = st.text_input("Enter the folder path")

    if st.button("Ingest Folder"):
        if folder_path == "":
            st.warning("Please enter the folder path.")
        elif not os.path.isdir(folder_path):
            st.warning("Invalid folder path.")
        else:
            pdf_query.ingest(folder_path)
            st.success("Folder ingested successfully.")

    question = st.text_input("Ask a question")

    if st.button("Ask"):
        if pdf_query.chain is None:
            st.warning("Please ingest a folder.")
        elif question == "":
            st.warning("Please enter a question.")
        else:
            response = pdf_query.ask(question)
            st.success(response)

if __name__ == "__main__":
    main()
