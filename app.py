import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

from rag import upload_docs_db, load_repo, get_document_info

load_dotenv()

st.set_page_config(layout="wide")

st.title('CodeGPT 🤖')
if 'github_repo' not in st.session_state:
    st.session_state['github_repo'] = ''
if 'openapi_key' not in st.session_state:
    st.session_state['openapi_key'] = ''
if 'repo_scripts' not in st.session_state:
    st.session_state['repo_scripts'] = ''
if 'repo_script_names' not in st.session_state:
    st.session_state['repo_script_names'] = ''
if 'documents' not in st.session_state:
    st.session_state['documents'] = ''


def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.button('Reset Chat', on_click=reset_conversation)


with st.sidebar:
    openai_api_key = st.text_input(
        "OPENAI KEY", type="password", value=st.session_state['openapi_key'])
    github_repo_url = st.text_input(
        "GitHub URL", value=st.session_state['github_repo'])
    if st.button('Submit'):
        if os.path.isdir("temp_repo"):
            shutil.rmtree("temp_repo")
        st.session_state['github_repo'] = github_repo_url
        st.session_state['openapi_key'] = openai_api_key
        st.session_state['documents'] = load_repo(
            st.session_state['github_repo'])
        st.session_state['repo_script_names'], st.session_state['repo_scripts'] = get_document_info(
            st.session_state['documents'])

    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat", key=reset_button_key)
    if reset_button:
        del st.session_state["messages"]
        st.experimental_rerun()

st.radio('Model', ["GPT4", "CodeLlama"], horizontal=True)

with st.container():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can query your codebase. What can I answer today?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    # search = DuckDuckGoSearchRun(name="Search")
    # search_agent = initialize_agent(
    #     [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    # )

    llm = ChatOpenAI(model_name="gpt-4",
                     api_key=st.session_state['openapi_key'])
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=upload_docs_db(st.session_state['documents']), memory=memory)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        # response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        with st.spinner('Generating response...'):
            response = qa(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": response['answer']})
        st.write(response['answer'])
