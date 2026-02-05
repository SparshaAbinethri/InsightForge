from langgraph.graph import StateGraph, END
from graph.state import AppState
from graph.router import router

from nodes.data_engineer import data_engineer_node
from nodes.data_analyst import data_analyst_node
from nodes.sql_node import sql_node
from nodes.visualization import visualization_node
from nodes.questionnaire import questionnaire_node


def build_graph():
    graph = StateGraph(AppState)

    # Register nodes (names MUST match router)
    graph.add_node("Data Engineer", data_engineer_node)
    graph.add_node("Data Analyst", data_analyst_node)
    graph.add_node("SQL", sql_node)
    graph.add_node("Visualization", visualization_node)
    graph.add_node("Questionnaire", questionnaire_node)

    # Conditional entry
    graph.set_conditional_entry_point(router)

    # All nodes terminate
    graph.add_edge("Data Engineer", END)
    graph.add_edge("Data Analyst", END)
    graph.add_edge("SQL", END)
    graph.add_edge("Visualization", END)
    graph.add_edge("Questionnaire", END)

    return graph.compile()
