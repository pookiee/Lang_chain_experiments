import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from helper_functions import get_online_consulation, handle_userInput, get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain, get_summarization, get_text_chunks_summary
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template, summarization_template, answer_template, sources_template
import pyautogui

def main():
    load_dotenv()
    
    
    # Handles Conversation QA
    
    st.set_page_config(page_title = "Advanced Homework Helper", 
                       page_icon = ':books:')
    
    st.write(css, unsafe_allow_html = True)
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None 
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None 
    
    if 'text' not in st.session_state:
        st.session_state.text = None 
        
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = None 
        
    if 'text_chunks_summary' not in st.session_state:
        st.session_state.text_chunks_summary = None 
        
    if 'user_input_online' not in st.session_state:
        st.session_state.user_input_online = None 

    
    st.header('Chat with multiple textbook PDFs :books:')
    user_input = st.text_input('Ask a homework question')
    
    st.write(user_template.replace("{{MSG}}", 'Hello'), unsafe_allow_html = True)
    st.write(bot_template.replace("{{MSG}}", 'hello student'), unsafe_allow_html = True)

    if user_input:
        handle_userInput(user_input, user_template, bot_template)
    

    # online = st.toggle('Activate Online Mode')
    # if online:
    #     st.write('Online Mode activated!!!')
    
    # handles input data
    with st.sidebar:
        st.subheader('Your textbook')
        pdf_docs = st.file_uploader('Upload PDFs', 
                                    accept_multiple_files = True) # list of pdfs
        if st.button('Process') and pdf_docs is not None:
            st.session_state.text = None
            st.session_state.text_chunks = None
            st.session_state.text_chunks_summary = None

            with st.spinner('Loading'):
                st.session_state.text = get_pdf_text(pdf_docs) # raw text from all pdfs
                #st.write(text)
                
                st.session_state.text_chunks = get_text_chunks(st.session_state.text)
                st.session_state.text_chunks_summary = get_text_chunks_summary(st.session_state.text)

                #st.write(text_chunks)

                vectorstore = get_vectorstore(st.session_state.text_chunks)
                #st.write(vectorstore)
                
                st.session_state.conversation = get_conversation_chain(vectorstore)
                

    st.header('Summarize Multiple textbook PDFs :books:')
    result = st.button('Create Summary')
  
    if result and pdf_docs is not None:
        get_summarization(st.session_state.text_chunks_summary, summarization_template)

    
    st.header('Consult the web')
    st.session_state.user_input_online = st.text_input('Ask online')
    
    if st.session_state.user_input_online:
        get_online_consulation(st.session_state.user_input_online,answer_template, sources_template)
                
if __name__ == '__main__':
    main()