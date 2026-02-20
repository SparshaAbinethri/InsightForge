from graph.graph_builder import build_graph
from backend.routes.upload import DATA_STORAGE
from utils.dataframe_io import extract_schema

graph = build_graph()

def run_graph(role, user_input):
    raw_df = DATA_STORAGE.get("raw_df")

    if raw_df is None:
        return {"error": "No dataset uploaded"}

    schema = extract_schema(raw_df)

    result = graph.invoke({
        "role": role,
        "user_input": user_input,
        "dataframe": raw_df,
        "raw_df": raw_df,
        "cleaned_df": None,
        "dataframe_schema": schema
    })

    return result
