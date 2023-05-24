import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle as pkl
import pickle
from dotenv import load_dotenv
import os
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

        '''
        Dividing text into smaller chaunks to process in llm model as text limit size
        '''

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # computing embedding for chunks

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                Vectors = pickle.load(f)

        else:
            embeddings = OpenAIEmbeddings()

            Vectors = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(Vectors, f)


if __name__ == '__main__':
    main()
