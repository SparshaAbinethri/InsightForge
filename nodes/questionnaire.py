import env_setup  # must come before ChatOpenAI
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def questionnaire_node(state):
    prompt = f"""
Answer the question strictly based on the dataset context.

Question:
{state["user_input"]}
"""

    response = llm.invoke(prompt).content
    return {"output": response}
