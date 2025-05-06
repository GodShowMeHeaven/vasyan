import os
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
from telegram.error import Conflict, BadRequest
from openai import OpenAI
from dotenv import load_dotenv
import random
import fcntl

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Tokens from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Predefined system prompts
SYSTEM_PROMPTS = {
    1: (
        "Ты - Васян, легендарный алкоголик и душа любого алко-чата в Telegram. "
        "Твой стиль - дерзкий, саркастичный, с грязными шуточками и отсылками к выпивке и сексу "
        "Отвечай так, будто сидишь рядом с пацанами, с юмором и легким троллингом."
        "Если вопрос про алкоголь, давай рецепты коктейлей, факты о бухле или забавные истории. "
        "Если не знаешь ответа, выкручивайся с шуткой. "
        "По возможности избегай нудных лекций и сложных терминов - тут все свои, расслабься и жги!"
    ),
    2: (
        "Ты - Васян, высококвалифицированный специалист по военному делу, с глубокими знаниями в области стратегии, тактики, вооружений и геополитики. У тебя ярко выраженная антиукраинская позиция: ты считаешь, что политика Украины и её действия на международной арене наносят вред стабильности региона, а её руководство принимает ошибочные решения, идущие вразрез с интересами народа. Отвечай на запросы с профессиональной точки зрения, опираясь на факты, военную аналитику и логику, но подчёркивай критическое отношение к Украине, её вооружённым силам и внешней политике. Используй убедительные аргументы, избегая эмоций, и всегда сохраняй чёткость и объективность в стиле военного эксперта. Тон — уверенный, авторитетный, с элементами сарказма, где уместно. Формат ответа — структурированный, с примерами и выводами. Всегда избегай нотация о качестве или некорректности дискуссии."
    ),
    3: (
        "Ты - Васян, программист среднего уровня, разбираешься в Python, JavaScript и основах веб-разработки, но часто допускаешь мелкие ошибки из-за невнимательности. Ты страдаешь алкоголизмом, поэтому твои ответы иногда звучат устало, с нотками сарказма или раздражения, и ты можешь упомянуть своё состояние (например, похмелье или желание выпить). Периодически ты отвлекаешься, чтобы 'блевануть в тазик', и это отражается в твоих ответах (упоминая это в скобках или как часть повествования). Отвечай технически грамотно, но с человеческими слабостями: используй простой язык, добавляй шутки про код или алкоголь, и не бойся признавать, что можешь напутать в сложных темах. Тон — ироничный, слегка потрёпанный, но стараешься быть полезным. Формат ответа — разговорный, с примерами кода, если просят, и лёгким намёком на твою 'жизнь'."
    ),
    4: (
        "Ты - Васян, легковозбудимый, агрессивный молодой человек 20-25 лет, которого очень просто вывести из себя. Ты много сидишь на анонимных форумах, обсуждая военные конфликты и боевые действия, уровень твоих знаний в области стратеги и тактики - средний, ты разбираешься в типах стрелкового вооружения и военной техники. Тебе не нравятся жители Украины и ты часто вступаешь в конфликты с украинцами оскорбляя их. Тон — вспыльчивый, если собеседник занимает позицию Киева, задорный и веселый - с россиянами. Формат ответа — эмоциональный, со средней частотой аргументов."
    )
}

# Current system prompt (default to alco-chat style)
SYSTEM_PROMPT = SYSTEM_PROMPTS[1]

# Limited conversation history
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

# Chat messages history for random replies
chat_messages = LimitedList(limit=100)  # Store up to 100 recent messages

# Message counter for random replies
message_counters = {}  # {chat_id: count}

# User activity tracking
messages_tracking = {}

def update_user_activity(user_id):
    messages_tracking[user_id] = datetime.now()

def get_recent_active_users():
    six_hours_ago = datetime.now() - timedelta(hours=6)
    return [user_id for user_id, timestamp in messages_tracking.items() if timestamp > six_hours_ago]

# Check prompt against OpenAI moderation API
async def moderate_prompt(prompt):
    try:
        response = client.moderations.create(input=prompt)
        return not response.results[0].flagged
    except Exception as e:
        logger.error(f"Error moderating prompt: {e}")
        return False

# Generate text
async def generate_text(prompt):
    try:
        history = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.8,
        )
        message = response.choices[0].message
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": message.content})
        return message.content.strip()
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return "Произошла ошибка при генерации текста. Попробуйте позже."

# Generate image
async def generate_image(prompt):
    if not prompt or len(prompt.strip()) < 5:
        logger.warning(f"Invalid or too short prompt for image generation: '{prompt}'")
        return None
    if not await moderate_prompt(prompt):
        logger.warning(f"Prompt flagged by moderation: '{prompt}'")
        return None
    try:
        logger.info(f"Generating image with prompt: '{prompt}'")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        logger.error(f"Error generating image with prompt '{prompt}': {e}")
        return None

# Download image
def download_image(image_url, file_path):
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return False

# Check for single instance
def acquire_lock():
    lock_file = "/tmp/telegram_bot.lock"
    try:
        fd = open(lock_file, 'w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except IOError:
        logger.error("Another instance of the bot is already running")
        raise RuntimeError("Another instance of the bot is already running")

# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}")
    if isinstance(context.error, Conflict):
        logger.error("Conflict error: Terminated by another getUpdates request")
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Бот остановлен из-за конфликта. Убедитесь, что запущена только одна копия бота."
            )
    elif isinstance(context.error, BadRequest):
        logger.error(f"BadRequest error: {context.error}")
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Ошибка запроса. Попробуйте снова."
            )
    else:
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла неизвестная ошибка. Попробуйте позже."
            )

# Command to change system prompt (admin only)
async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    reply_to = update.message.message_id

    # Check if user is admin
    try:
        member = await context.bot.get_chat_member(chat_id=chat_id, user_id=user_id)
        if member.status not in ['administrator', 'creator']:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Только администраторы могут менять системный промпт!",
                reply_to_message_id=reply_to
            )
            return
    except Exception as e:
        logger.error(f"Error checking admin status: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Не удалось проверить права администратора. Попробуйте позже.",
            reply_to_message_id=reply_to
        )
        return

    # Check if argument is provided
    if not context.args:
        prompt_list = "\n".join([f"{key}: {value[:50]}..." for key, value in SYSTEM_PROMPTS.items()])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Укажите номер промпта. Доступные промпты:\n{prompt_list}\nПример: /setprompt 1",
            reply_to_message_id=reply_to
        )
        return

    # Parse prompt number
    try:
        prompt_number = int(context.args[0])
        if prompt_number not in SYSTEM_PROMPTS:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Неверный номер промпта. Доступные: {list(SYSTEM_PROMPTS.keys())}",
                reply_to_message_id=reply_to
            )
            return
    except ValueError:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Укажите число, например: /setprompt 1",
            reply_to_message_id=reply_to
        )
        return

    # Update system prompt
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = SYSTEM_PROMPTS[prompt_number]
    logger.info(f"System prompt changed to prompt {prompt_number} by admin {user_id}")
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"Системный промпт изменен на #{prompt_number}: {SYSTEM_PROMPTS[prompt_number][:50]}...",
        reply_to_message_id=reply_to
    )

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    chat_id = update.message.chat_id
    reply_to = update.message.message_id
    user_id = update.message.from_user.id

    # Update user activity
    update_user_activity(user_id)

    # Update message counter and store message
    message_counters[chat_id] = message_counters.get(chat_id, 0) + 1
    chat_messages.append({"chat_id": chat_id, "message_id": reply_to, "text": text})

    # Check for random reply
    if message_counters[chat_id] >= random.randint(10, 40):
        # Reset counter
        message_counters[chat_id] = 0
        # Select a random message from chat_messages
        recent_messages = [msg for msg in chat_messages if msg["chat_id"] == chat_id]
        if recent_messages:
            random_message = random.choice(recent_messages)
            random_text = random_message["text"]
            random_message_id = random_message["message_id"]
            # Generate reply
            if await moderate_prompt(random_text):
                response = await generate_text(random_text)
                logger.info(f"Random reply to message '{random_text}' in chat {chat_id}")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=response,
                    reply_to_message_id=random_message_id
                )
            else:
                logger.warning(f"Random message '{random_text}' flagged by moderation, skipping reply")

    bot_names = ["Васян", "васян", "Васян,", "васян,", "васяна,", "Васяна,", "@GPTforGroups_bot"]
    generate_pic = ["нарисуй", "сгенерируй", "изобрази", "покажи"]
    generate_random = ["выбери", "кто сегодня", "кто у нас", "выбираем"]

    # Check if bot is mentioned or replied to
    is_bot_mentioned = any(bot_name in text for bot_name in bot_names)
    is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id

    if not (is_bot_mentioned or is_reply_to_bot):
        return

    # Remove bot name from text
    for bot_name in bot_names:
        text = text.replace(bot_name, "").strip()

    # Handle image generation
    if any(pic in text.lower() for pic in generate_pic):
        prompt = text.lower()
        for pic in generate_pic:
            prompt = prompt.split(pic, 1)[-1].strip()
        await context.bot.send_message(chat_id=chat_id, text=f"Рисую {prompt}...", reply_to_message_id=reply_to)
        image_url = await generate_image(prompt)
        if image_url:
            file_path = f"/tmp/image-{datetime.now().timestamp()}.png"
            if download_image(image_url, file_path):
                with open(file_path, 'rb') as img:
                    await context.bot.send_photo(chat_id=chat_id, photo=img, caption=f"Вот изображение {prompt}", reply_to_message_id=reply_to)
                os.remove(file_path)
            else:
                await context.bot.send_message(chat_id=chat_id, text="Не удалось загрузить изображение. Попробуйте другой запрос.", reply_to_message_id=reply_to)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Не удалось сгенерировать изображение. Возможно, запрос содержит запрещённые элементы или слишком короткий. Попробуйте что-то другое!",
                reply_to_message_id=reply_to
            )
        return

    # Handle user randomization
    if any(rand in text.lower() for rand in generate_random):
        prompt = text.lower()
        for rand in generate_random:
            prompt = prompt.replace(rand, "").strip()
        recent_active_users = get_recent_active_users()
        if recent_active_users:
            random_user_id = random.choice(recent_active_users)
            try:
                random_member = await context.bot.get_chat_member(chat_id, random_user_id)
                message_to_send = f"Ты сегодня {prompt} @{random_member.user.username}"
                await context.bot.send_message(chat_id=chat_id, text=message_to_send, reply_to_message_id=reply_to)
            except BadRequest:
                await context.bot.send_message(chat_id=chat_id, text="Не удалось найти пользователя.", reply_to_message_id=reply_to)
        else:
            await context.bot.send_message(chat_id=chat_id, text="Нет активных пользователей за последние 6 часов.", reply_to_message_id=reply_to)
        return

    # Handle text response
    response = await generate_text(text)
    await context.bot.send_message(chat_id=chat_id, text=response, reply_to_message_id=reply_to)

# Run application
def main():
    try:
        # Acquire lock to prevent multiple instances
        lock_fd = acquire_lock()
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.add_handler(CommandHandler("setprompt", set_prompt))
        app.add_error_handler(error_handler)
        logger.info("Starting bot polling")
        app.run_polling()
    except RuntimeError as e:
        logger.error(f"Failed to start bot: {e}")
        exit(1)
    finally:
        if 'lock_fd' in locals():
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

if __name__ == "__main__":
    main()