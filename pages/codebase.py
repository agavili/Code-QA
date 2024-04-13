import streamlit as st
import pandas as pd
import numpy as np

repo_url = ''
if 'github_repo' not in st.session_state:
    st.session_state['github_repo'] = ''

if st.session_state['github_repo'] != '':
    repo_url = st.session_state['github_repo']
else:
    st.write('repo url not updated')

if repo_url:
    scripts, script_code = st.session_state['repo_script_names'], st.session_state['repo_scripts']
    option = st.selectbox(
        'Select script to inspect..',
        scripts)

    for i, script in enumerate(scripts):
        if option == script:
            st.code(script_code[i], language='python')
