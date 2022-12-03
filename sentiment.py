# Импортируем библиотеки
from fastapi import FastAPI # для работы с сервером FastAPI
from transformers import pipeline # для определения тональности текста
from pydantic import BaseModel # для получения из HTTP поле text текстового вида

# Автоматическая установка всех библиотек из файла requirements.txt командой
# pip install -r requirements.txt

class Item(BaseModel): # создаём класс Item на основе BaseModel
    text: str # с одним полем text типа str

app = FastAPI() # создаём объект FastAPI
# Создаём классификатор pipeline для определения тональности текста
classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

# Декоратор позволяет менять поведение функции, не меняя саму функцию
@app.get("/") # обращаемся к корневому каталогу сервера с помощью метода GET
def root(): # функция, вызываемая при обращении к корневому каталогу сервера
    return {"message": "Тест сервера FastAPI ОК!"} # тестовое сообщение

@app.post("/predict/") # обращаемся к ссылке /predict/ с помощью метода POST
def predict(item: Item): # функция, вызываемая при обращении к ссылке /predict/
    return classifier(item.text)[0] # передаём в классификатор текст из тела сообщения

# Запускаем сервер FastAPI командой uvicorn sentiment:app

# classifier("Я обожаю инженерию машинного обучения!")
# classifier("Я обожаю собачек!")
# classifier("Я пытаюсь изучить GitHub!!!")
# classifier("Я люблю Git") #Проба пера
# classifier("Я изучаю инженерию машинного обучения") 
# --- для отката изменений необходимо ввести команду в терминале: git revert <Номер временной шкалы>
# для изменения имени набрать в терминале git config --global user.name "FIRST_NAME LAST_NAME"