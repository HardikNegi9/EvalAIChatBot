import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input
        
    def get_llm_model(self):
        try:
            selected_model = self.user_controls_input.get('selected_model', 'gemini-2.0-flash')
            api_key = os.environ.get("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(model=selected_model, api_key=api_key)
        except Exception as e:
            raise ValueError(f"Error Occurred with Exception : {e}")
        return llm
