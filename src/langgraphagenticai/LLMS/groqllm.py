import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input=user_controls_input
        

    def get_llm_model(self):
        try:
            selected_groq_model=self.user_controls_input['selected_groq_model']
            llm = ChatGroq(model=selected_groq_model)

        except Exception as e:
            raise ValueError(f"Error Occurred with Exception : {e}")
        return llm