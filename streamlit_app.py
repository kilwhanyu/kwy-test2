import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import SummarizationChain
from langchain import PromptTemplate

# Streamlit UI
st.set_page_config(page_title="LangChain Text Summarizer", layout="centered")
st.title("LangChain GPT-4o-Mini Text Summarizer")

# Sidebar API Key Input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    try:
        # LLM Initialization
        llm = OpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
        summarization_chain = SummarizationChain(llm=llm)

        # Text Input for Summarization
        text_to_summarize = st.text_area("Enter text to summarize:")

        if st.button("Summarize"):
            if text_to_summarize:
                # Perform Summarization
                summary = summarization_chain.run(text_to_summarize)
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.warning("Please enter text to summarize.")
    except ModuleNotFoundError as e:
        st.error(f"ModuleNotFoundError: {str(e)}. Please make sure all required modules are installed.")
else:
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")