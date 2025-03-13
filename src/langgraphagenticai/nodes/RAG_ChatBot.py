from src.langgraphagenticai.state.state import GraphState
from src.langgraphagenticai.tools.structured_llm import StructuredLLMs
from src.langgraphagenticai.vectorstore.retriever import Retriever

class RagChatBot:
    """
    Basic chatbot logic implementation.
    """
    def __init__(self, user_controls_input):
        self.llm = StructuredLLMs(user_controls_input)
        self.rag_chain = self.llm.Generator()
        self.retrieval_grader = self.llm.Grader()
        self.hallucination_grader = self.llm.Hallucination_grader()
        self.answer_grader = self.llm.Answer_grader()
        self.question_rewriter = self.llm.Rewriter()
        self.question_router = self.llm.Router()
        self.retriever = Retriever().get_retriever()

    def retrieve(self, state: GraphState):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state: GraphState):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        state["generated"] += 1
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state: GraphState):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state: GraphState):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question, "question_rewritten": True}

    def HumanSupport(self, state: GraphState):
        """
        Human support agent

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---Human Support---")
        question = state["question"]

        # Human Support Agent
        response = "Hello Hardik here, I am here to help you with your query"

        return {"generation": response, "question": question}

    def route_question(self, state: GraphState):
        """
        Route question to HumanSupport or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        if source.datasource == "HumanSupport":
            print("---ROUTE QUESTION TO HumanSupport---")
            return "HumanSupport"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(self, state: GraphState):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                if state["question_rewritten"]:
                    print("---DECISION: GENERATION DOES NOT ADDRESS THE REWRITTEN QUESTION, SWITCH TO HUMAN SUPPORT---")
                    return "Looping_Conditon"

                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"