def router(state):
    role = state["role"]

    if role == "SQL":
        return "SQL"
    elif role == "Data Engineer":
        return "Data Engineer"
    elif role == "Data Analyst":
        return "Data Analyst"
    elif role == "Visualization":
        return "Visualization"
    elif role == "Questionnaire":
        return "Questionnaire"
    else:
        raise ValueError(f"Unknown role: {role}")
