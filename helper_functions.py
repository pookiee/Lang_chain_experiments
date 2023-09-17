from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import streamlit as st 
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from langchain.llms import SelfHostedHuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import logging

load_dotenv()

def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs: # loop over pdfs
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: # loop over every page for a pdf
            text += page.extract_text()
    return text 


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators = [" ", ",", "\n"],
                                          chunk_size = 400,
                                          chunk_overlap = 50
                                           )
    
    chunks = text_splitter.split_text(text)
    return chunks 

def get_text_chunks_summary(text):
    text_splitter = RecursiveCharacterTextSplitter(separators = [" ", ",", "\n"],
                                          chunk_size = 1000,
                                          chunk_overlap = 100
                                           )
    
    chunks = text_splitter.create_documents([text])
    return chunks 

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings )
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


def handle_userInput(userinput, user_template,bot_template ):
    response = st.session_state.conversation({'question': userinput})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True )
        else: 
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True )


def get_summarization(pdfs, summarization_template): 
    # loader  = PyPDFDirectoryLoader(pdfs)
    # docs = loader.load()
    #docs = [Document(page_content=t) for t in pdfs]
    # st.write(docs)
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    #model_id = "lmsys/fastchat-t5-3b-v1.0"
    
    # model_id = "facebook/bart-large-cnn"
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_id,
    #     task="summarization",
    #     model_kwargs={"temperature": 0.5, "max_length": 1000, 'do_sample': False},
    #      truncation=True
    # )
        
    # llm = SelfHostedHuggingFaceLLM(
    #     model_id=model_id,
    #     task="summarization",
    #     model_kwargs={"temperature": 0.5, "max_length": 1000}
    # )
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    #print(chain.run(docs))
    st.write(summarization_template.replace("{{MSG}}", chain.run(pdfs)), unsafe_allow_html = True )
    
# def get_summarization(pdfs, summarization_template): 
#     # loader  = PyPDFDirectoryLoader(pdfs)
#     # docs = loader.load()
#     docs = pdfs
#     st.write(docs[0])
#     #llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
#     #model_id = "lmsys/fastchat-t5-3b-v1.0"
    
#     model_id = "facebook/bart-large-cnn"
    
#     tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_len=512)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     pipe = pipeline(
#     "summarization", model=model, tokenizer=tokenizer, max_new_tokens=10
# )       
#     tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':1000}
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # llm = SelfHostedHuggingFaceLLM(
#     #     model_id=model_id,
#     #     task="summarization",
#     #     model_kwargs={"temperature": 0.5, "max_length": 1000}
#     # )
    
#     chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
#     st.write(summarization_template.replace("{{MSG}}", chain.run(docs)), unsafe_allow_html = True )
     
# def get_online_consulation(user_input): 
#     embeddings_func = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")
#     #vectorstore = FAISS(embedding_function = embeddings_func)
#     vectorstore = Chroma(embedding_function=embeddings_func,persist_directory="./chroma_db_oai")

#     llm1 = HuggingFaceHub(repo_id = "google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 500})
#     llm2 = HuggingFaceHub(repo_id = "google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 500})

#     search = GoogleSearchAPIWrapper()

#     web_research_retriever = WebResearchRetriever.from_llm(
#     vectorstore=vectorstore,
#     llm=llm1, 
#     search=search,
#     num_search_results = 3)
    

#     qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm2,retriever=web_research_retriever, 
#                                                            chain_type = 'map_reduce',
#                                                            verbose = True, 
#                                                            reduce_k_below_max_tokens=True,
#                                                            max_tokens_limit=2000,
#                                                            return_source_documents = True)
#     result = qa_chain({"question": user_input})
    
    
#     st.write(result)
#     st.write(result)

def get_online_consulation(user_input, answer_template, sources_template): 
    embeddings_func = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")
    #vectorstore = FAISS(embedding_function = embeddings_func)
    vectorstore = Chroma(embedding_function=embeddings_func,persist_directory="./chroma_db_oai")

    llm1 = HuggingFaceHub(repo_id = "google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 500})
    llm2 = HuggingFaceHub(repo_id = "google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 500})

    search = GoogleSearchAPIWrapper()

    web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm1, 
    search=search,
    num_search_results = 3)
    
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

    docs = web_research_retriever.get_relevant_documents(user_input)

    chain = load_qa_chain(llm2, chain_type = 'stuff')
    
    output = chain({"input_documents": docs, "question": user_input},return_only_outputs=False)

    # st.write(' # Answer')
    # st.write(output['output_text'])
    # st.write(' # Sources')
    # st.write(output['input_documents'])
    

    st.write(answer_template.replace("{{MSG}}", output['output_text']), unsafe_allow_html = True )
    
    if output['input_documents']:
        st.write(sources_template.replace("{{MSG}}", output['input_documents']), unsafe_allow_html = True )
    else:
        st.write(sources_template.replace("{{MSG}}", 'Online Consultation not needed'), unsafe_allow_html = True )
