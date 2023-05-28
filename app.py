import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.chains.question_answering import load_qa_chain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub

from langchain.llms import OpenAI
# sidebar
with st.sidebar:

    st.title("😊💭 PDF CHAT GPT APP")
    st.markdown('''
    # About App 
    This app is an LLM powered chatbot built using:
     -[Streamlit]
     -[LangChain]
     -[OpenAI]  LLM model''')

    add_vertical_space(5)

    st.write("Made by  Amit malik")

load_dotenv()


def main():

    st.header(" CHAT WITH PDF 💭")

    # upload a PDF file

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # computing embedding for chunks

        embeddings = HuggingFaceEmbeddings()
        Vectors = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user question / query

        query = st.text_input("Ask questions about your PDF file.")
        # st.write(query)

        if query:

            docs = Vectors.similarity_search(query=query, k=3)
            # st.write(docs)
            # feeding ranked results to LLM AI

            llm = HuggingFaceHub(repo_id="google/flan-t5-xl",
                                 model_kwargs={"temperature": 0, "max_length": 512})

            chain = load_qa_chain(llm=llm, chain_type="stuff")

            response = chain.run(input_documents=docs, question=query)

            st.write(response)


if __name__ == '__main__':
    main()
