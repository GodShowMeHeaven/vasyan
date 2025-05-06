import os
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
from telegram.error import Conflict, BadRequest, NetworkError
from openai import OpenAI
from dotenv import load_dotenv
import random
import fcntl
import feedparser
import emoji
import aiohttp
from aiohttp import ClientTimeout

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()
if not os.path.exists('.env'):
    logger.warning("Environment file (.env) not found, relying on Railway variables")

# Tokens from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate tokens
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN is not set")
    exit(1)
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    exit(1)

logger.info("Environment variables loaded successfully")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# News cache
NEWS_CACHE = {"summary": None, "timestamp": None}
CACHE_DURATION = timedelta(minutes=30)

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

# System prompts by chat
SYSTEM_PROMPTS_BY_CHAT = {}  # {chat_id: prompt}

# Conversation histories by chat
conversation_histories = {}  # {chat_id: LimitedList}

# Chat messages history for random replies and summaries
chat_messages = LimitedList(limit=100)  # Store up to 100 recent messages

# Message counter for random replies
message_counters = {}  # {chat_id: count}

# User activity tracking
messages_tracking = {}

def get_conversation_history(chat_id):
    if chat_id not in conversation_histories:
        conversation_histories[chat_id] = LimitedList(limit=MAX_TOKENS)
    return conversation_histories[chat_id]

def get_system_prompt(chat_id):
    if chat_id not in SYSTEM_PROMPTS_BY_CHAT:
        SYSTEM_PROMPTS_BY_CHAT[chat_id] = SYSTEM_PROMPTS[1]  # Default to alco-chat
    return SYSTEM_PROMPTS_BY_CHAT[chat_id]

def update_user_activity(user_id):
    messages_tracking[user_id] = datetime.now()

def get_recent_active_users():
    six_hours_ago = datetime.now() - timedelta(hours=6)
    return [user_id for user_id, timestamp in messages_tracking.items() if timestamp > six_hours_ago]

# Check if message contains only emoji
def is_only_emoji(text):
    return all(emoji.is_emoji(char) for char in text.strip())

# Check prompt against OpenAI moderation API
async def moderate_prompt(prompt, skip_moderation=False):
    if skip_moderation:
        logger.info("Skipping moderation for prompt")
        return True
    try:
        logger.info("Sending moderation request to OpenAI")
        response = client.moderations.create(input=prompt)
        flagged = response.results[0].flagged
        logger.info(f"Moderation result for prompt: flagged={flagged}")
        return not flagged
    except Exception as e:
        logger.error(f"Error moderating prompt: {e}", exc_info=True)
        return False

# Fetch news from RSS
async def fetch_news():
    global NEWS_CACHE
    current_time = datetime.now()

    # Check cache
    if NEWS_CACHE["summary"] and NEWS_CACHE["timestamp"]:
        if current_time - NEWS_CACHE["timestamp"] < CACHE_DURATION:
            logger.info("Returning cached news")
            return NEWS_CACHE["summary"]

    rss_urls = [
        "https://static.feed.rbc.ru/rbc/logical/footer/news.rss",
        "https://lenta.ru/rss/news"
    ]
    
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=15)) as session:
        for rss_url in rss_urls:
            try:
                logger.info(f"Fetching news from {rss_url}")
                async with session.get(rss_url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch RSS from {rss_url}: HTTP {response.status}")
                        continue
                    content = await response.text()
                
                logger.info(f"Parsing RSS feed from {rss_url}")
                feed = feedparser.parse(content)
                
                if not feed.entries:
                    logger.warning(f"No news entries found in {rss_url}")
                    continue
                
                # Get top 3 news entries
                articles = feed.entries[:3]
                news_summary = f"Последние новости ({rss_url.split('/')[2]}):\n\n"
                for i, article in enumerate(articles, 1):
                    title = article.get("title", "Без заголовка")
                    description = article.get("description", "Без описания")
                    description = description.replace("<p>", "").replace("</p>", "").strip()
                    news_summary += f"{i}. **{title}**\n{description}\n\n"
                
                # Moderate the news summary
                logger.info("Moderating news summary")
                if not await moderate_prompt(news_summary, skip_moderation=False):
                    logger.warning("News summary flagged by moderation")
                    return "Новости содержат запрещённый контент. Попробуйте другой запрос."
                
                # Update cache
                NEWS_CACHE["summary"] = news_summary.strip()
                NEWS_CACHE["timestamp"] = current_time
                logger.info(f"Successfully fetched news: {news_summary[:100]}...")
                return news_summary.strip()
            
            except Exception as e:
                logger.error(f"Error fetching news from {rss_url}: {e}", exc_info=True)
                continue
        
        logger.error("Failed to fetch news from all RSS sources")
        return "Не удалось получить новости. Попробуйте позже."

# Summarize chat activity
async def summarize_chat(chat_id):
    try:
        recent_messages = [msg for msg in chat_messages if msg["chat_id"] == chat_id]
        if not recent_messages:
            logger.warning(f"No valid messages found for chat {chat_id}")
            return "Недостаточно сообщений для анализа чата. Поболтайте побольше!"
        
        messages_text = "\n".join(
            f"@{msg['username']}: {msg['text']}" for msg in recent_messages
        )
        prompt = (
            "Ты - аналитик чата. Ниже приведены последние сообщения из Telegram-чата (до 100 сообщений, отфильтрованные: без сообщений ботов, картинок, эмодзи и короче 3 символов). "
            "Сделай краткую выжимку (3-5 предложений) о том, о чём говорили в чате, какие темы обсуждались, какой был настрой (например, весёлый, серьёзный). "
            "Не упоминай конкретных пользователей по именам, только общие темы и настроение. "
            "Вот сообщения:\n\n" + messages_text
        )
        
        if not await moderate_prompt(prompt):
            logger.warning(f"Chat summary prompt flagged by moderation in chat {chat_id}")
            return "Сообщения чата содержат запрещённый контент. Попробуйте другой запрос."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты - аналитик, делающий краткие и точные выжимки."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        
        if not await moderate_prompt(summary):
            logger.warning(f"Chat summary flagged by moderation in chat {chat_id}")
            return "Сводка чата содержит запрещённый контент. Попробуйте другой запрос."
        
        logger.info(f"Generated chat summary for chat {chat_id}: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing chat {chat_id}: {e}", exc_info=True)
        return "Произошла ошибка при анализе чата. Попробуйте позже."

# Generate text
async def generate_text(prompt, chat_id):
    try:
        history = [{"role": "system", "content": get_system_prompt(chat_id)}] + get_conversation_history(chat_id) + [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.8,
        )
        message = response.choices[0].message
        conversation_history = get_conversation_history(chat_id)
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": message.content})
        return message.content.strip()
    except Exception as e:
        logger.error(f"Error generating text in chat {chat_id}: {e}", exc_info=True)
        return "Произошла ошибка при генерации текста. Попробуйте позже."

# Generate image
async def generate_image(prompt):
    if not prompt or len(prompt.strip()) < 5:
        logger.warning(f"Invalid or too short prompt for image generation: '{prompt}'")
        return None
    if not await moderate_prompt(prompt):
        logger.warning(f"Prompt flagged by moderation")
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
        logger.error(f"Error generating image with prompt '{prompt}': {e}", exc_info=True)
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
        logger.error(f"Error downloading image: {e}", exc_info=True)
        return False

# Check for single instance
def acquire_lock():
    lock_file = "/tmp/telegram_bot.lock"
    try:
        fd = open(lock_file, 'w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.info(f"Lock acquired successfully for process {os.getpid()}")
        return fd
    except IOError as e:
        logger.error(f"Failed to acquire lock, another instance is running (PID: {os.getpid()}): {e}", exc_info=True)
        raise RuntimeError("Another instance of the bot is already running")

# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update is None:
        logger.error(f"Error with no update object: {context.error}", exc_info=True)
        return
    
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    logger.error(f"Update {update} in chat {chat_id} caused error: {context.error}", exc_info=True)
    
    if isinstance(context.error, Conflict):
        logger.error(f"Conflict error in chat {chat_id}: Terminated by another getUpdates request")
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Бот остановлен из-за конфликта. Убедитесь, что запущена только одна копия бота."
            )
    elif isinstance(context.error, BadRequest):
        logger.error(f"BadRequest error in chat {chat_id}: {context.error}")
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Ошибка запроса. Попробуйте снова."
            )
    elif isinstance(context.error, NetworkError):
        logger.error(f"Network error in chat {chat_id}: {context.error}")
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Ошибка сети. Проверьте подключение и попробуйте снова."
            )
    else:
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Произошла неизвестная ошибка. Попробуйте позже."
            )

# Command to change system prompt (admin only)
async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    reply_to = update.message.message_id

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
        logger.error(f"Error checking admin status in chat {chat_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Не удалось проверить права администратора. Попробуйте позже.",
            reply_to_message_id=reply_to
        )
        return

    if not context.args:
        prompt_list = "\n".join([f"{key}: {value[:50]}..." for key, value in SYSTEM_PROMPTS.items()])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Укажите номер промпта. Доступные промпты:\n{prompt_list}\nПример: /setprompt 1",
            reply_to_message_id=reply_to
        )
        return

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

    SYSTEM_PROMPTS_BY_CHAT[chat_id] = SYSTEM_PROMPTS[prompt_number]
    logger.info(f"System prompt changed to prompt {prompt_number} in chat {chat_id} by admin {user_id}")
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
    username = update.message.from_user.username or update.message.from_user.first_name

    update_user_activity(user_id)

    if (
        user_id != context.bot.id and
        not update.message.from_user.is_bot and
        update.message.photo is None and
        len(text) >= 3 and
        not is_only_emoji(text)
    ):
        chat_messages.append({
            "chat_id": chat_id,
            "message_id": reply_to,
            "text": text,
            "user_id": user_id,
            "username": username
        })
        message_counters[chat_id] = message_counters.get(chat_id, 0) + 1

    if message_counters.get(chat_id, 0) >= random.randint(10, 40):
        message_counters[chat_id] = 0
        recent_messages = [msg for msg in chat_messages if msg["chat_id"] == chat_id]
        if recent_messages:
            random_message = random.choice(recent_messages)
            random_text = random_message["text"]
            random_message_id = random_message["message_id"]
            if await moderate_prompt(random_text):
                response = await generate_text(random_text, chat_id)
                logger.info(f"Random reply to message '{random_text}' in chat {chat_id}")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=response,
                    reply_to_message_id=random_message_id
                )
            else:
                logger.warning(f"Random message '{random_text}' flagged by moderation in chat {chat_id}, skipping reply")

    bot_names = ["Васян", "васян", "Васян,", "васян,", "васяна,", "Васяна,", "@GPTforGroups_bot"]
    generate_pic = ["нарисуй", "сгенерируй", "изобрази", "покажи"]
    generate_random = ["кто сегодня", "кто у нас"]
    generate_news = ["последние новости", "новости", "что нового"]
    generate_summary = ["что с чатом"]

    is_bot_mentioned = any(bot_name in text for bot_name in bot_names)
    is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id

    if not (is_bot_mentioned or is_reply_to_bot):
        return

    for bot_name in bot_names:
        text = text.replace(bot_name, "").strip()

    if any(summary in text.lower() for summary in generate_summary):
        logger.info(f"Chat summary request in chat {chat_id}")
        await context.bot.send_message(chat_id=chat_id, text="Анализирую, что творится в чате...", reply_to_message_id=reply_to)
        summary = await summarize_chat(chat_id)
        await context.bot.send_message(chat_id=chat_id, text=summary, reply_to_message_id=reply_to)
        return

    if any(news in text.lower() for news in generate_news):
        logger.info(f"News request in chat {chat_id}: {text}")
        await context.bot.send_message(chat_id=chat_id, text="Собираю последние новости...", reply_to_message_id=reply_to)
        news_summary = await fetch_news()
        await context.bot.send_message(chat_id=chat_id, text=news_summary, reply_to_message_id=reply_to)
        return

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
                    await context.bot.send_message(chat_id=chat_id, text=f"Вот изображение {prompt}", reply_to_message_id=reply_to)
                    await context.bot.send_photo(chat_id=chat_id, photo=img, reply_to_message_id=reply_to)
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

    response = await generate_text(text, chat_id)
    await context.bot.send_message(chat_id=chat_id, text=response, reply_to_message_id=reply_to)

# Run application
def main():
    lock_fd = None
    try:
        logger.info("Acquiring lock")
        lock_fd = acquire_lock()
        logger.info("Checking network connectivity")
        try:
            response = requests.get("https://api.telegram.org", timeout=5)
            logger.info(f"Telegram API check: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to reach Telegram API: {e}", exc_info=True)
        try:
            response = requests.get("https://api.openai.com", timeout=5)
            logger.info(f"OpenAI API check: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to reach OpenAI API: {e}", exc_info=True)
        logger.info("Building Telegram application")
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        logger.info("Adding handlers")
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.add_handler(CommandHandler("setprompt", set_prompt))
        app.add_error_handler(error_handler)
        logger.info("Starting bot polling")
        app.run_polling()
    except RuntimeError as e:
        logger.error(f"Failed to start bot: {e}", exc_info=True)
        exit(1)
    except NetworkError as e:
        logger.error(f"Network error during initialization: {e}", exc_info=True)
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        exit(1)
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            logger.info("Lock released")

if __name__ == "__main__":
    logger.info("Starting bot")
    main()