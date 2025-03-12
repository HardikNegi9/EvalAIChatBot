from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from langchain_core.prompts import ChatPromptTemplate
from src.langgraphagenticai.state.state import RouteQuery, GradeDocuments, GradeHallucinations, GradeAnswer
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

class StructuredLLMs:
    def __init__(self, user_controls_input):
        self.llm_instance = GroqLLM(user_controls_input)
        self.llm = self.llm_instance.get_llm_model()

    # Router
    def Router(self):
        # LLM with function call
        structured_llm_router = self.llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or HumanSupport.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router

        return question_router
    

    # Grade Documents
    def Grader(self):
        # Retrieval Grader

        # LLM with function call
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader

        return retrieval_grader
    

    # Generate
    def Generator(self):

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()

        return rag_chain


    # Hallucination Grader
    def Hallucination_grader(self):

        # LLM with function call
        structured_llm_Hal_grader = self.llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | structured_llm_Hal_grader

        return hallucination_grader
    

    # Answer Grader
    def Answer_grader(self):

        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | structured_llm_grader
        return answer_grader


    # Question Re-writer
    def Rewriter(self):

        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        return question_rewriter