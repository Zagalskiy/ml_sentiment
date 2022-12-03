# Импортируем библиотеки
from fastapi import FastAPI
from transformers import pipeline # определяет тональность текста
from pydantic import BaseModel # получает из HTTP поле text текстового вида

# Автоматическая установка всех библиотек из файла requirements.txt командой
# pip install -r requirements.txt

class Item(BaseModel): # создаём класс Item на основе BaseModel
    text: str # с одним полем text типа str

app = FastAPI() # создаём объект FastAPI
# Создаём классификатор pipeline для определения тональности текста
classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

# Функция, вызываемая при обращении к корневому каталогу сервера с помощью метода GET
@app.get("/") # декоратор позволяет менять поведение функции, не меняя саму функцию 
def root():
    return {"message": "Тест сервера FastAPI ОК!"}

# Функция, вызываемая при обращении к каталогу /predict/ с помощью метода POST
@app.post("/predict/") # декоратор позволяет менять поведение функции, не меняя саму функцию 
def predict(item: Item):
    return classifier(item.text)[0] # передаём в классификатор текст из тела сообщения

# Запускаем сервер FastAPI командой uvicorn sentiment:app

# classifier("Я обожаю инженерию машинного обучения!")
# classifier("Я обожаю собачек!")
# classifier("Я пытаюсь изучить GitHub!!!")
# classifier("Я люблю Git") #Проба пера
# classifier("Я изучаю инженерию машинного обучения") 
# --- для отката изменений необходимо ввести команду в терминале: git revert <Номер временной шкалы>
# для изменения имени набрать в терминале git config --global user.name "FIRST_NAME LAST_NAME"