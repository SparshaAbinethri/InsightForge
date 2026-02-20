from fastapi import APIRouter
from backend.models.request_models import RunRequest
from backend.services.graph_service import run_graph

router = APIRouter()

@router.post("/run")
def execute_role(request: RunRequest):
    result = run_graph(request.role, request.user_input)
    return result
