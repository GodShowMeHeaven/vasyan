import os
import requests
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, CallbackContext, filters
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Токены из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# История сообщений
conversation_history = []

# Помощник: вырезать слишком длинную историю
def trim_history(history, max_tokens=3000):
    trimmed = []
    tokens = 0
    for msg in reversed(history):
        tokens += len(msg.get("content", "").split())
        if tokens > max_tokens:
            break
        trimmed.insert(0, msg)
    return trimmed

# Генерация текста
async def generate_text(prompt):
    history = trim_history(conversation_history + [{"role": "user", "content": prompt}])
    response = client.chat.completions.create(
        model="gpt-4",
        messages=history,
        temperature=0.8,
    )
    message = response.choices[0].message
    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": message.content})
    return message.content.strip()

# Генерация изображения
async def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    return image_url

# Загрузка изображения
def download_image(image_url, file_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return True
    return False

# Обработка сообщений
async def handle_message(update: Update, context: CallbackContext):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    chat_id = update.message.chat_id
    reply_to = update.message.message_id

    if "нарисуй" in text.lower():
        prompt = text.lower().split("нарисуй", 1)[-1].strip()
        await context.bot.send_message(chat_id=chat_id, text=f"Рисую: {prompt}...", reply_to_message_id=reply_to)
        image_url = await generate_image(prompt)
        file_path = f"/tmp/image-{datetime.now().timestamp()}.png"
        if download_image(image_url, file_path):
            with open(file_path, 'rb') as img:
                await context.bot.send_photo(chat_id=chat_id, photo=img, caption=f"Вот изображение: {prompt}", reply_to_message_id=reply_to)
            os.remove(file_path)
        else:
            await context.bot.send_message(chat_id=chat_id, text="Не удалось загрузить изображение.")
    else:
        response = await generate_text(text)
        await context.bot.send_message(chat_id=chat_id, text=response, reply_to_message_id=reply_to)

# Запуск приложения
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
