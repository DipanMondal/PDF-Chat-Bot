import streamlit as st
from pdf_processor import *
from chunk_process import *
from vector_encoding import *
from vector_store import *

embedding = Embedding()


def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs :books:")
    st.text_input("Ask a question about your documents:")

    # sidebar
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and press 'Process'",
            accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get raw text
                raw_text = get_pdf_text(pdf_docs)
                print("Get raw text : ")
                print(raw_text,"\n\n","-"*100,"\n\n")
                # get chunks
                chunks,data = get_chunks(raw_text)
                print("Get chunks : ")
                print(data[0])
                print(len(data), len(data[0]), "\n\n", "-" * 100, "\n\n")
                # vectorize the chunks and store them
                embeddings = embedding.get_embeddings(data)
                print("GET EMBEDDINGS")
                print(embeddings.shape,"\n\n", "-" * 100, "\n\n")
                # store invector database
                store_data(chunks=chunks,embeddings=embeddings)
                print("Stored data")


if __name__ == '__main__':
    main()