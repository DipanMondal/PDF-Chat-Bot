import streamlit as st
from pdf_processor import *
from chunk_process import *
from vector_store import *
from vector_encoding import Embedding
from memory import Memory
from chatbot import chatbot
from templates import css, user_template, bot_template
import io


def handle_user_input(user_input):
    for chat,images in zip(st.session_state.chats,st.session_state.images):
        st.write(user_template.replace('{{MSG}}',chat[0]), unsafe_allow_html=True)
        st.write(bot_template.replace('{{MSG}}', chat[1]), unsafe_allow_html=True)
        for img_bytes in images:
            st.image(io.BytesIO(img_bytes), caption=f"image", use_column_width=True)

    embd = st.session_state.embedding.get_query_embedding(user_input)[0]
    # results = get_query_match(embd)
    texts, images = get_relavant_data(embd)
    print(len(texts),len(images))
    context=""
    for i, each in enumerate(texts):
        context += f"\n{i+1}."+each+"\n"

    answer = chatbot.get_response(context, user_input)

    st.write(user_template.replace('{{MSG}}', user_input), unsafe_allow_html=True)
    st.write(bot_template.replace('{{MSG}}', answer), unsafe_allow_html=True)
    for img_bytes in images:
        st.image(io.BytesIO(img_bytes), caption=f"image", use_column_width=True)
    st.session_state.chats.append([user_input, answer])
    st.session_state.images.append(images)


if "my_text" not in st.session_state:
    st.session_state.my_text = ""

if "embedding" not in st.session_state:
    st.session_state.embedding = Embedding()


def submit():
    st.session_state.my_text = st.session_state.widget
    st.session_state.widget = ""


def main():
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "mem" not in st.session_state:
        st.session_state.mem = Memory()
    if "images" not in st.session_state:
        st.session_state.images = []
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs :books:")
    st.text_input("Ask a question about your documents:",key="widget", on_change=submit)
    
    user_input = st.session_state.my_text

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
                st.session_state.mem.pdf_lists_new = [pdf.name for pdf in pdf_docs]
                print("NEW LIST : ",st.session_state.mem.pdf_lists_new)
                print("OLD LIST : ",st.session_state.mem.pdf_lists_old)
                # pdfs to be deleted
                for each in st.session_state.mem.pdf_lists_old:
                    if each not in st.session_state.mem.pdf_lists_new:
                        need_to_delete.append(each)
                for each in st.session_state.mem.pdf_lists_new:
                    if each not in st.session_state.mem.pdf_lists_old:
                        need_to_process.append(each)
                print("NEED TO DELETE : ",need_to_delete)
                print("NEED TO PROCESS : ",need_to_process)
                # update old list
                st.session_state.mem.pdf_lists_old = [name for name in st.session_state.mem.pdf_lists_new]
                # delete data
                for name in need_to_delete:
                    delete_data(name=name)
                # process data
                for pdf_doc in pdf_docs:
                    if pdf_doc.name in need_to_process:
                        print("processing...",pdf_doc.name)

                        # pdf extraction
                        text_data, image_data = pdf_extract(pdf_doc)
                        # chunk data
                        chunk_data = get_chunks(text_data)

                        text_embeddings = st.session_state.embedding.get_text_embeddings(chunk_data)
                        image_embeddings = st.session_state.embedding.get_image_embeddings(image_data)
                        store_text_data(chunk_data,text_embeddings,pdf_doc.name)
                        store_image_data(image_data,image_embeddings,pdf_doc.name)


if __name__ == '__main__':
    main()
