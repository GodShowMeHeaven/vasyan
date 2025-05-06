import os
import requests
import asyncio
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, MessageHandler, CallbackContext, filters
from telegram.error import BadRequest
from openai import OpenAI
from dotenv import load_dotenv
import random

# Загружаем переменные из .env
load_dotenv()

# Токены из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Ограниченная история сообщений
MAX_TOKENS = 16000

class LimitedList(list):
    def __init__(self, limit):
        self.limit = limit
        super().__init__()

    def append(self, item):
        super().append(item)
        self._check_limit()

    def extend(self, iterable):
        super().extend(iterable)
        self._check_limit()

    def _check_limit(self):
        while len(self) > self.limit:
            self.pop(0)

conversation_history = LimitedList(limit=MAX_TOKENS)

# Отслеживание активности пользователей
messages_tracking = {}

def update_user_activity(user_id):
    messages_tracking[user_id] = datetime.now()

def get_recent_active_users():
    six_hours_ago = datetime.now() - timedelta(hours=6)
    return [user_id for user_id, timestamp in messages_tracking.items() if timestamp > six_hours_ago]

# Генерация текста
async def generate_text(prompt):
    history = conversation_history + [{"role": "user", "content": prompt}]
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
    user_id = update.message.from_user.id

    # Обновляем активность пользователя
    update_user_activity(user_id)

    bot_names = ["Васян", "васян", "Васян,", "васян,", "васяна,", "Васяна,", "@GPTforGroups_bot"]
    generate_pic = ["нарисуй", "сгенерируй", "изобрази", "покажи"]
    generate_random = ["выбери", "кто сегодня", "кто у нас", "выбираем"]

    # Проверяем, есть ли обращение к боту
    is_bot_mentioned = any(bot_name in text for bot_name in bot_names)
    is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id

    if not (is_bot_mentioned or is_reply_to_bot):
        return

    # Удаляем имя бота из текста, если оно есть
    for bot_name in bot_names:
        text = text.replace(bot_name, "").strip()

    # Обработка генерации изображения
    if any(pic in text.lower() for pic in generate_pic):
        prompt = text.lower()
        for pic in generate_pic:
            prompt = prompt.split(pic, 1)[-1].strip()
        await context.bot.send_message(chat_id=chat_id, text=f"Рисую: {prompt}...", reply_to_message_id=reply_to)
        image_url = await generate_image(prompt)
        file_path = f"/tmp/image-{datetime.now().timestamp()}.png"
        if download_image(image_url, file_path):
            with open(file_path, 'rb') as img:
                await context.bot.send_photo(chat_id=chat_id, photo=img, caption=f"Вот изображение: {prompt}", reply_to_message_id=reply_to)
            os.remove(file_path)
        else:
            await context.bot.send_message(chat_id=chat_id, text="Не удалось загрузить изображение.")
        return

    # Обработка рандомизации пользователей
    if any(rand in text.lower() for rand in generate_random):
        prompt = text.lower()
        for rand in generate_random:
            prompt = prompt.replace(rand, "").strip()
        recent_active_users = get_recent_active_users()
        if recent_active_users:
            random_user_id = random.choice(recent_active_users)
            try:
                random_member = await context.bot.get_chat_member(chat_id, random_user_id)
                message_to_send = f"Ты сегодня {prompt}. Вот случайный пользователь: @{random_member.user.username}"
                await context.bot.send_message(chat_id=chat_id, text=message_to_send, reply_to_message_id=reply_to)
            except BadRequest:
                await context.bot.send_message(chat_id=chat_id, text="Не удалось найти пользователя.", reply_to_message_id=reply_to)
        else:
            await context.bot.send_message(chat_id=chat_id, text="Нет активных пользователей за последние 6 часов.", reply_to_message_id=reply_to)
        return

    # Обработка текстового ответа
    response = await generate_text(text)
    await context.bot.send_message(chat_id=chat_id, text=response, reply_to_message_id=reply_to)

# Запуск приложения
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()