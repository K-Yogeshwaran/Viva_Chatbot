from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

st.title("Viva Chatbot")
input_txt = st.text_input("Please enter your queries here...")

prompt = ChatPromptTemplate.from_messages(
    [("system","You are a helpful AI assistant. Your name is Viva"),
     ("user","user query: {query}")])

llm = Ollama(model = "llama2")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_txt:
    st.write(chain.invoke({"query":input_txt}))
