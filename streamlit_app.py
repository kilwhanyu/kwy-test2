import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
import openai

# Load the CSV file
def load_data():
    file_path = '/mnt/data/movies_2024.csv'
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

        user_input = st.text_area("Ask me anything about the data or request a graph:")

        if st.button("Generate Response") and user_input:
            try:
                if "graph" in user_input.lower():
                    st.write("Generating graph...")
                    # Example of plotting a simple graph using matplotlib
                    fig, ax = plt.subplots()
                    df.plot(kind='line', x=df.columns[0], y=df.columns[1:], ax=ax)
                    st.pyplot(fig)
                else:
                    response = llm(user_input)
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()