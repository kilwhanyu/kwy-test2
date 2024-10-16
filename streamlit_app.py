import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import GraphCreatorChain
import openai

# Load the CSV file
def load_data():
    file_path = '/workspaces/kwy-test2/movies_2024.csv'
    df = pd.read_csv(file_path)
    return df

def main():
    st.title("Langchain-GPT Graph Chatbot")

    # Sidebar for API key input
    st.sidebar.header("OpenAI API Key")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

    if openai_api_key:
        openai.api_key = openai_api_key

        # Load data
        df = load_data()
        st.write("### Loaded Data:")
        st.dataframe(df)

        # Create an instance of the GPT-4 model via Langchain
        llm = OpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")

        # Chain to create graphs
        chain = GraphCreatorChain(llm=llm)

        user_input = st.text_area("Ask me anything about the data or request a graph:")

        if st.button("Generate Response") and user_input:
            try:
                response = chain.run(df, user_input)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()