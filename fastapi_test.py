from fastapi.testclient import TestClient
from sentiment import app
import httpx

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Тест сервера FastAPI ОК!"}

def test_predict_positive():
    response = client.post("/predict/",
        json={"text": "Я люблю машинное обучение!"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'

def test_predict_negative():
    response = client.post("/predict/",
        json={"text": "Я ненавижу машинное обучение!"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'
