"""
Telegram-бот с RAG-функциональностью на основе OpenAI API.

Этот бот умеет:
- Отвечать на вопросы с использованием базы знаний (RAG)
- Обрабатывать изображения и извлекать текст (Vision)
- Индексировать документы в векторную базу данных
- Логировать все операции
"""

import asyncio
import logging
from pathlib import Path
from typing import List
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode

from config import TELEGRAM_TOKEN, DOCS_PATH, LOG_LEVEL, LOG_FORMAT
from rag.pipeline import RAGPipeline

# ========== НАСТРОЙКА ЛОГИРОВАНИЯ ==========
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== ИНИЦИАЛИЗАЦИЯ БОТА ==========
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Инициализируем RAG-пайплайн
logger.info("Инициализация RAG Pipeline...")
rag_pipeline = RAGPipeline()
logger.info("RAG Pipeline готов к работе")

# ========== ПАМЯТЬ РАЗГОВОРОВ ==========
# Словарь для хранения истории сообщений каждого пользователя
# Структура: {user_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
conversation_history = {}  # Обычный диалог
rag_history = {}  # Отдельная история вопросов к базе
MAX_HISTORY_LENGTH = 10  # Максимум последних сообщений для контекста


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """
    Разбивает текст на части заданного размера.
    
    Args:
        text: Текст для разбиения
        max_chars: Максимальный размер одной части в символах
        
    Returns:
        Список частей текста
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Разбиваем по параграфам (двойной перенос строки)
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph_length = len(paragraph) + 2  # +2 для \n\n
        
        # Если параграф сам больше лимита, разбиваем его по предложениям
        if paragraph_length > max_chars:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                sentence_length = len(sentence) + 2
                
                if current_length + sentence_length > max_chars and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        
        # Если добавление параграфа превысит лимит, сохраняем текущий chunk
        elif current_length + paragraph_length > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length
    
    # Добавляем последний chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def load_documents_from_directory(directory: Path) -> tuple[List[str], List[str]]:
    """
    Загружает все текстовые документы из директории.
    Большие документы разбиваются на части (chunks).
    
    Args:
        directory: Путь к директории с документами
        
    Returns:
        Кортеж (тексты документов, имена файлов)
    """
    documents = []
    sources = []
    
    if not directory.exists():
        logger.warning(f"Директория {directory} не существует")
        return documents, sources
    
    # Ищем все .txt файлы
    for file_path in directory.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
                # Разбиваем большие документы на части
                chunks = chunk_text(text, max_chars=6000)
                
                if len(chunks) > 1:
                    logger.info(f"Загружен документ: {file_path.name} ({len(text)} символов) - разбит на {len(chunks)} частей")
                    for i, chunk in enumerate(chunks, 1):
                        documents.append(chunk)
                        sources.append(f"{file_path.name} (часть {i}/{len(chunks)})")
                else:
                    logger.info(f"Загружен документ: {file_path.name} ({len(text)} символов)")
                    documents.append(text)
                    sources.append(file_path.name)
                    
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
    
    logger.info(f"Всего загружено документов/частей: {len(documents)}")
    return documents, sources


def format_rag_answer(answer: str, sources: list) -> str:
    """Добавляет имена файлов из результатов поиска к ответу."""
    response = f"💡 Ответ:\n{answer}"
    if sources and not answer.startswith("❌"):
        response += "\n\n📚 Источники контекста: " + ", ".join(dict.fromkeys(sources))
    return response


async def send_long_message(message: Message, text: str, max_length: int = 4000):
    """
    Отправляет длинное сообщение, разбивая его на части если нужно.
    
    Telegram ограничивает длину сообщения 4096 символами.
    
    Args:
        message: Исходное сообщение для ответа
        text: Текст для отправки
        max_length: Максимальная длина одного сообщения
    """
    if len(text) <= max_length:
        await message.answer(text)
        return
    
    # Разбиваем на части
    parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    for part in parts:
        await message.answer(part)
        await asyncio.sleep(0.5)  # Небольшая задержка между сообщениями


# ========== ОБРАБОТЧИКИ КОМАНД ==========

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start - приветствие и информация о боте.
    """
    logger.info(f"Пользователь {message.from_user.id} запустил бота")
    
    welcome_text = """
🤖 <b>Добро пожаловать в RAG-бота!</b>

Я интеллектуальный ассистент с доступом к базе знаний.
Я могу помочь вам найти информацию и ответить на вопросы.

<b>Мои возможности:</b>
📚 Поиск информации в базе знаний (RAG)
🖼 Обработка изображений и извлечение текста
💬 Ответы на вопросы с использованием контекста

<b>Доступные команды:</b>
/start - Показать это сообщение
/help - Подробная справка
/ask &lt;вопрос&gt; - Задать вопрос с поиском в базе знаний
/ingest - Перезагрузить базу знаний (только для администраторов)
/stats - Показать статистику системы

<b>Как пользоваться:</b>
• Просто напишите вопрос для обычного ответа ИИ без поиска по базе
• Отправьте изображение с текстом, и я его обработаю
• Для поиска по базе с источниками используйте /ask &lt;вопрос&gt;

Powered by OpenAI GPT-4 + FAISS
"""
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """
    Обработчик команды /help - подробная справка.
    """
    logger.info(f"Пользователь {message.from_user.id} запросил справку")
    
    help_text = """
📖 <b>Подробная справка по использованию бота</b>

<b>1️⃣ RAG-запросы (Retrieval-Augmented Generation)</b>
Для поиска по базе знаний используйте /ask &lt;вопрос&gt;. Обычные сообщения получают ответ ИИ без поиска по базе и без источников. Истории режимов раздельные.

Примеры запросов:
• "/ask Что такое RAG и как он работает?"
• "/ask Как применяется RAG в поддержке клиентов?"
• "/ask Расскажи о преимуществах RAG в HR"

<b>2️⃣ Обработка изображений</b>
Отправьте мне изображение (фото, скриншот, скан документа), и я:
• Извлеку текст с изображения
• Могу ответить на вопросы по этому тексту
• Найду связанную информацию в базе знаний

Пример: отправьте фото документа с подписью "Что это значит?"

<b>3️⃣ Команды</b>
/ask &lt;вопрос&gt; - Явный RAG-запрос
Пример: /ask Как работает векторный поиск?

/ingest - Перезагрузка базы знаний
Эта команда переиндексирует все документы. Используется при обновлении базы знаний.

/stats - Статистика системы
Показывает информацию о загруженных документах, используемых моделях и состоянии индекса.

<b>4️⃣ Технические детали</b>
🔹 Модель чата: GPT-4o-mini
🔹 Модель vision: GPT-4o-mini
🔹 Эмбеддинги: text-embedding-3-small
🔹 Векторная БД: FAISS
🔹 Количество релевантных документов: 3

<b>Примеры эффективного использования:</b>
✅ "Объясни как RAG помогает в HR-процессах"
✅ "Какие метрики эффективности RAG в поддержке?"
✅ "В чем преимущества использования RAG?"

❌ "Привет" (слишком общий запрос)
❌ "???" (неясный запрос)

Если у вас есть вопросы - просто задайте их! 😊
"""
    await message.answer(help_text, parse_mode=ParseMode.HTML)


@dp.message(Command("ask"))
async def cmd_ask(message: Message):
    """
    Обработчик команды /ask - RAG-запрос с явным указанием.
    """
    # Извлекаем вопрос из команды
    query = message.text[5:].strip()  # Убираем "/ask "
    
    if not query:
        await message.answer(
            "❌ Пожалуйста, укажите вопрос после команды.\n"
            "Пример: /ask Что такое RAG?"
        )
        return
    
    logger.info(f"RAG-запрос от пользователя {message.from_user.id}: {query}")
    
    # Отправляем сообщение о начале обработки
    processing_msg = await message.answer("🔍 Ищу информацию в базе знаний...")
    
    try:
        # Выполняем RAG-запрос
        user_id = message.from_user.id
        history = rag_history.setdefault(user_id, [])
        result = rag_pipeline.query_with_history(query, history)
        if not result['answer'].startswith("❌"):
            history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": result['answer']},
            ])
            rag_history[user_id] = history[-MAX_HISTORY_LENGTH * 2:]
        
        # Формируем ответ с источниками
        response_text = format_rag_answer(result['answer'], result.get('sources', []))
        
        # Удаляем сообщение о обработке
        await processing_msg.delete()
        
        # Отправляем ответ
        await send_long_message(message, response_text)
        
        logger.info(f"Ответ отправлен пользователю {message.from_user.id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await processing_msg.delete()
        await message.answer(f"❌ Произошла ошибка: {str(e)}")


@dp.message(Command("ingest"))
async def cmd_ingest(message: Message):
    """
    Обработчик команды /ingest - индексация документов.
    
    Загружает все документы из директории data/docs и индексирует их в FAISS.
    """
    logger.info(f"Запрос на индексацию от пользователя {message.from_user.id}")
    
    # Отправляем сообщение о начале процесса
    await message.answer("📥 Начинаю индексацию документов...")
    
    try:
        # Загружаем документы из директории
        documents, sources = load_documents_from_directory(DOCS_PATH)
        
        if not documents:
            await message.answer(
                f"❌ Документы не найдены в директории {DOCS_PATH}\n"
                "Пожалуйста, добавьте .txt файлы с документами."
            )
            return
        
        await message.answer(f"📄 Найдено чанков: {len(documents)}\nНачинаю обработку...")
        
        # Индексируем документы
        success = rag_pipeline.index_documents(documents, sources)
        
        if success:
            stats = rag_pipeline.get_stats()
            response = (
                "✅ <b>Индексация завершена успешно!</b>\n\n"
                f"📊 Статистика:\n"
                f"• Чанки: {stats['total_documents']}\n"
                f"• Векторов: {stats['total_vectors']}\n"
                f"• Размерность: {stats['dimension']}\n\n"
                f"Бот готов к работе! Задавайте вопросы. 💬"
            )
            await message.answer(response, parse_mode=ParseMode.HTML)
            logger.info("Индексация завершена успешно")
        else:
            await message.answer("❌ Ошибка при индексации документов. См. логи для деталей.")
            
    except Exception as e:
        logger.error(f"Ошибка при индексации: {e}")
        await message.answer(f"❌ Произошла ошибка при индексации: {str(e)}")


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """
    Обработчик команды /stats - показывает статистику системы.
    """
    logger.info(f"Запрос статистики от пользователя {message.from_user.id}")
    
    stats = rag_pipeline.get_stats()
    user_id = message.from_user.id
    
    status_emoji = "✅" if stats['is_loaded'] else "❌"
    status_text = "Загружена" if stats['is_loaded'] else "Не загружена"
    
    # Статистика истории для пользователя
    history_count = (len(conversation_history.get(user_id, [])) + len(rag_history.get(user_id, []))) // 2  # Делим на 2 (пары вопрос-ответ)
    
    stats_text = f"""
📊 <b>Статистика RAG-системы</b>

<b>Состояние базы знаний:</b>
{status_emoji} {status_text}

<b>Данные:</b>
• Чанки: {stats['total_documents']}
• Векторов: {stats['total_vectors']}
• Размерность: {stats['dimension']}

<b>Модели:</b>
• Чат: {stats['chat_model']}
• Vision: {stats['vision_model']}
• Эмбеддинги: {stats['embed_model']}

<b>Файлы:</b>
• Индекс: {'✅' if stats['index_exists'] else '❌'}
• Метаданные: {'✅' if stats['metadata_exists'] else '❌'}

<b>Ваш контекст:</b>
• Сообщений в истории: {history_count}
"""
    
    await message.answer(stats_text, parse_mode=ParseMode.HTML)


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    """
    Обработчик команды /clear - очищает историю разговора пользователя.
    """
    user_id = message.from_user.id
    logger.info(f"Очистка истории для пользователя {user_id}")
    
    if user_id in conversation_history or user_id in rag_history:
        messages_count = (len(conversation_history.get(user_id, [])) + len(rag_history.get(user_id, []))) // 2
        conversation_history[user_id] = []
        rag_history[user_id] = []
        await message.answer(
            f"🧹 <b>История очищена!</b>\n\n"
            f"Удалено {messages_count} сообщений из контекста.\n"
            f"Начинаем разговор с чистого листа!",
            parse_mode=ParseMode.HTML
        )
    else:
        await message.answer(
            "ℹ️ История пустая.\n\n"
            "У вас еще нет сохраненных сообщений в контексте."
        )


# ========== ОБРАБОТЧИКИ СООБЩЕНИЙ ==========

@dp.message(F.photo)
async def handle_photo(message: Message):
    """
    Обработчик фотографий - извлекает текст с изображения.
    """
    logger.info(f"Получено изображение от пользователя {message.from_user.id}")
    
    # Отправляем сообщение о начале обработки
    processing_msg = await message.answer("🖼 Обрабатываю изображение...")
    
    try:
        # Получаем файл изображения
        photo = message.photo[-1]  # Берем самое большое разрешение
        file = await bot.get_file(photo.file_id)
        
        # Формируем URL для доступа к файлу
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"
        
        # Извлекаем текст пользователя если есть
        user_query = message.caption if message.caption else None
        
        # Обрабатываем изображение через vision
        result = rag_pipeline.process_image(file_url, user_query)
        
        if result.get('error'):
            await processing_msg.delete()
            await message.answer(f"❌ Ошибка при обработке изображения: {result['error']}")
            return
        
        # Формируем ответ с источниками
        response_text = "📄 Текст с изображения:\n"
        response_text += result['extracted_text'] + "\n\n"
        
        # Добавляем обычный ответ ИИ на вопрос к изображению
        if result.get('image_answer'):
            response_text += result['image_answer']
        
        # Удаляем сообщение о обработке
        await processing_msg.delete()
        
        # Отправляем результат
        await send_long_message(message, response_text)
        
        logger.info(f"Изображение обработано для пользователя {message.from_user.id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await processing_msg.delete()
        await message.answer(f"❌ Произошла ошибка: {str(e)}")


@dp.message(F.text)
async def handle_text(message: Message):
    """
    Обработчик обычного диалога с ИИ без поиска по базе знаний.
    """
    query = message.text
    user_id = message.from_user.id
    
    # Игнорируем команды (они обрабатываются отдельно)
    if query.startswith('/'):
        return
    
    logger.info(f"Текстовый запрос от пользователя {user_id}: {query}")
    
    # Отправляем сообщение о начале обработки
    processing_msg = await message.answer("🔍 Ищу информацию...")
    
    try:
        # Инициализируем историю для нового пользователя
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        # Обычный диалог с отдельной историей, без поиска по базе
        result = rag_pipeline.chat_with_history(query, conversation_history[user_id])
        
        # Сохраняем в историю ЧИСТЫЕ вопросы и ответы (без RAG промптов)
        conversation_history[user_id].append({"role": "user", "content": query})
        conversation_history[user_id].append({"role": "assistant", "content": result['answer']})
        
        # Ограничиваем размер истории
        if len(conversation_history[user_id]) > MAX_HISTORY_LENGTH * 2:
            # Оставляем только последние MAX_HISTORY_LENGTH пар (вопрос-ответ)
            conversation_history[user_id] = conversation_history[user_id][-(MAX_HISTORY_LENGTH * 2):]
            logger.info(f"История обрезана до {MAX_HISTORY_LENGTH} пар сообщений")
        
        # Формируем ответ с источниками
        response_text = result['answer']
        
        # Удаляем сообщение о обработке
        await processing_msg.delete()
        
        # Отправляем ответ
        await send_long_message(message, response_text)
        
        logger.info(f"Ответ отправлен пользователю {user_id} (история: {len(conversation_history[user_id])//2} сообщений)")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await processing_msg.delete()
        await message.answer(f"❌ Произошла ошибка: {str(e)}")


# ========== ЗАПУСК БОТА ==========

async def main():
    """
    Главная функция запуска бота.
    """
    logger.info("=" * 50)
    logger.info("Запуск Telegram-бота с RAG (OpenAI)")
    logger.info("=" * 50)
    
    # Проверяем состояние RAG-пайплайна
    stats = rag_pipeline.get_stats()
    if stats['is_loaded']:
        logger.info(f"✅ База знаний загружена: {stats['total_documents']} документов")
    else:
        logger.warning("⚠️ База знаний не загружена. Выполните команду /ingest")
    
    # Запускаем polling
    logger.info("Бот запущен и готов к работе!")
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

