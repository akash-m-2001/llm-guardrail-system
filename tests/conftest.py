import warnings

import pytest
from fastapi.testclient import TestClient

from app.core.state import clear_sessions
from app.main import app

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(autouse=True)
def fresh_sessions():
    clear_sessions()
    yield
    clear_sessions()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def check(client):
    def _check(prompt, session_id="s1"):
        response = client.post("/guardrail", json={"session_id": session_id, "prompt": prompt})
        assert response.status_code == 200, response.text
        return response.json()

    return _check
