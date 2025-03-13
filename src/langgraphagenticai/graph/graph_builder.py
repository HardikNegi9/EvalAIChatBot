from langgraph.graph import StateGraph, START,END
from src.langgraphagenticai.state.state import GraphState
from src.langgraphagenticai.nodes.RAG_ChatBot import RagChatBot




class GraphBuilder:

    def __init__(self,model, user_controls_input):
        self.user_controls_input=user_controls_input
        self.llm=model
        self.graph_builder=StateGraph(GraphState)


    
    def RAG_ChatBot(self):
        
        rag_node_obj = RagChatBot(self.user_controls_input)

        # Define the nodes
        self.graph_builder.add_node("HumanSupport", rag_node_obj.HumanSupport)  # HumanSupport
        self.graph_builder.add_node("retrieve", rag_node_obj.retrieve)  # retrieve
        self.graph_builder.add_node("grade_documents", rag_node_obj.grade_documents)  # grade documents
        self.graph_builder.add_node("generate", rag_node_obj.generate)  # generatae
        self.graph_builder.add_node("transform_query", rag_node_obj.transform_query)  # transform_query

        # Build graph
        self.graph_builder.add_conditional_edges(
            START,
            rag_node_obj.route_question,
            {
                "HumanSupport": "HumanSupport",
                "vectorstore": "retrieve",
            },
        )
        self.graph_builder.add_edge("HumanSupport", END)
        self.graph_builder.add_edge("retrieve", "grade_documents")
        self.graph_builder.add_conditional_edges(
            "grade_documents",
            rag_node_obj.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.graph_builder.add_edge("transform_query", "retrieve")
        self.graph_builder.add_conditional_edges(
            "generate",
            rag_node_obj.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
                "Looping_Conditon": "HumanSupport",
            },
        )

        # Compile
        app = self.graph_builder.compile()

    
    
    
    def setup_graph(self):
        """
        Sets up the graph for the RAG chatbot.
        """
        self.RAG_ChatBot()
        
        return self.graph_builder.compile()
    




    

