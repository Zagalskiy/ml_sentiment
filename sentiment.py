from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

@app.get("/")
def root():
    return {"message": "Тест сервера FastAPI ОК!"}

@app.post("/predict/")
def predict(item: Item):
    return classifier("Я люблю машинное обучение!")[0]
# classifier("Я обожаю инженерию машинного обучения!")
# classifier("Я обожаю собачек!")
# classifier("Я пытаюсь изучить GitHub!!!")
# classifier("Я люблю Git") #Проба пера
# classifier("Я изучаю инженерию машинного обучения") 
# --- для отката изменений необходимо ввести команду в терминале: git revert <Номер временной шкалы>
# для изменения имени набрать в терминале git config --global user.name "FIRST_NAME LAST_NAME"