import streamlit as st
import json
import os
from datetime import date
from src.langgraphagenticai.ui.uiconfigfile import Config
# from dotenv import load_dotenv
from src.langgraphagenticai.vectorstore.retriever import Retriever
import asyncio

# load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()  # config
        self.user_controls = {}

    def initialize_session(self):
        return {
            "current_step": "requirements",
            "requirements": "",
            "user_stories": "",
            "po_feedback": "",
            "generated_code": "",
            "review_feedback": "",
            "decision": None
        }

    def save_data(self, pdf_file, urls):
        data = {}

        if pdf_file:
            pdf_dir = "uploaded_pdfs"
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_file_path = os.path.join(pdf_dir, pdf_file.name)
            with open(pdf_file_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            data["pdf_file"] = pdf_file_path

        if urls:
            data["urls"] = urls

        with open("data.json", "w") as f:
            json.dump(data, f)

        # Call set_retriever function
        retriever = Retriever()
        asyncio.run(retriever.set_retriever())
        st.success("Retriever setup completed successfully.")

    def load_streamlit_ui(self):
        st.set_page_config(page_title="🤖 " + self.config.get_page_title(), layout="wide")
        st.header("🤖 " + self.config.get_page_title())
        st.session_state.timeframe = ''
        st.session_state.IsFetchButtonClicked = False
        st.session_state.IsSDLC = False

        with st.sidebar:
            # Model selection
            model_options = self.config.get_groq_model_options()
            self.user_controls["selected_groq_model"] = st.selectbox("Select Model", model_options)

            # PDF and URL input for retriever
            self.user_controls["pdf_file"] = st.file_uploader("Upload PDF", type=["pdf"])
            self.user_controls["urls"] = st.text_area("Enter URLs (one per line)")

            if st.button("Submit"):
                if self.user_controls["pdf_file"] or self.user_controls["urls"]:
                    st.session_state.IsFetchButtonClicked = True
                    st.session_state.timeframe = {
                        "pdf_file": self.user_controls["pdf_file"],
                        "urls": self.user_controls["urls"].splitlines()
                    }
                    self.save_data(self.user_controls["pdf_file"], self.user_controls["urls"].splitlines())
                else:
                    st.warning("⚠️ Please upload at least one PDF or enter at least one URL to proceed.")

            if "state" not in st.session_state:
                st.session_state.state = self.initialize_session()

        return self.user_controls