import streamlit as st
from pdf_processor import *
from chunk_process import *
from vector_encoding import *
from vector_store import *
from vector_encoding import ob as embedding
from memory import memory as mem

# embedding = Embedding()


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
                need_to_process = []
                need_to_delete = []
                mem.pdf_lists_new = [pdf.name for pdf in pdf_docs]
                print("NEW LIST : ",mem.pdf_lists_new)
                print("OLD LIST : ",mem.pdf_lists_old)
                # pdfs to be deleted
                for each in mem.pdf_lists_old:
                    if each not in mem.pdf_lists_new:
                        need_to_delete.append(each)
                for each in mem.pdf_lists_new:
                    if each not in mem.pdf_lists_old:
                        need_to_process.append(each)
                print("NEED TO DELETE : ",need_to_delete)
                print("NEED TO PROCESS : ",need_to_process)
                # update old list
                mem.pdf_lists_old = [name for name in mem.pdf_lists_new]
                # delete data
                for name in need_to_delete:
                    delete_data(name=name)
                # process data
                for pdf_doc in pdf_docs:
                    if pdf_doc.name in need_to_process:
                        print("processing...",pdf_doc.name)
                        raw_text = get_pdf_text(pdf_doc)
                        # get chunks
                        chunks,data = get_chunks(raw_text)
                        # vectorize the chunks and store them
                        embeddings = embedding.get_embeddings(data)
                        # store invector database
                        store_data(chunks=chunks,embeddings=embeddings,name=pdf_doc.name)
                        # print("Stored data")


if __name__ == '__main__':
    main()
