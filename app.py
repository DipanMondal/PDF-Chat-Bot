import streamlit as st
from pdf_processor import *
from chunk_process import *
from vector_encoding import *
from vector_store import *
from vector_encoding import ob as embedding
from memory import memory as mem
from chatbot import chatbot
from templates import css, user_template, bot_template


def handle_user_input(user_input):
    for chat in st.session_state.chats:
        st.write(user_template.replace('{{MSG}}',chat[0]), unsafe_allow_html=True)
        st.write(bot_template.replace('{{MSG}}', chat[1]), unsafe_allow_html=True)
    embd = embedding.get_embeddings(user_input)
    results = get_query_match(embd, 3)
    context=""
    for i, each in enumerate(results):
        context += f"\n{i+1}."+each['text']+"\n"

    answer = chatbot.get_response(context, user_input)

    st.write(user_template.replace('{{MSG}}', user_input), unsafe_allow_html=True)
    st.write(bot_template.replace('{{MSG}}', answer), unsafe_allow_html=True)
    st.session_state.chats.append([user_input, answer])


def main():
    if "chats" not in st.session_state:
        st.session_state.chats = []
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs :books:")
    user_input = st.text_input("Ask a question about your documents:")

    if user_input:
        handle_user_input(user_input)

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
                        embeddings = embedding.get_embeddings(chunks)
                        # store invector database
                        store_data(chunks=chunks,embeddings=embeddings,name=pdf_doc.name)
                        print("Stored data")


if __name__ == '__main__':
    main()
