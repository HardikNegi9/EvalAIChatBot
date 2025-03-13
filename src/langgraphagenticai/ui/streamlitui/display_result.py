import streamlit as st

class DisplayResultStreamlit:
    def __init__(self, graph, user_message):
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        graph = self.graph
        user_message = self.user_message

        with st.chat_message("User"):
            st.write(user_message)
        
        new_question = None

        placeholder = st.empty()
        for output in graph.stream({'question': user_message, 'generated': 0, 'question_rewritten': False}):
            for key, value in output.items():

                with placeholder.container():
                    with st.chat_message("assistant"):
                        if key == "transform_query":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Transforming the question...</p>", unsafe_allow_html=True)
                            new_question = value["question"]
                        elif key == "retrieve":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Retrieved documents ✅</p>", unsafe_allow_html=True)
                        elif key == "grade_documents":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Graded documents ✅</p>", unsafe_allow_html=True)
                        elif key == "generate":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Generating Response...</p>", unsafe_allow_html=True)
                        elif key == "grade_generation_v_documents_and_question":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Grading Response...</p>", unsafe_allow_html=True)
                        elif key == "HumanSupport":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Routing to Human Support...</p>", unsafe_allow_html=True)
                        elif key == "vectorstore":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>Routing to RAG...</p>", unsafe_allow_html=True)
                        elif key == "END":
                            st.markdown(f"<p style='font-size: small; color: lightgray;'>END</p>", unsafe_allow_html=True)
            if new_question:
                with st.chat_message("assistant"):
                    st.markdown(f"<p style='font-size: small; color: lightgray;'>{"Updated Question: ", new_question}</p>", unsafe_allow_html=True)
                new_question = None
        
        placeholder = st.empty()
                        
        with st.chat_message("assistant"):
            st.write(value["generation"])