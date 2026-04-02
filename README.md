# Book Q&A Telegram Bot

Telegram bot for document-grounded Q&A over uploaded PDF and TXT books.

## Overview

This project lets a user upload a book and ask questions about its content in chat.
The bot extracts text, chunks it, embeds the chunks, retrieves the most relevant parts,
and returns a concise answer.

## Features

- Upload PDF or TXT files directly in Telegram
- Extract and chunk book content for retrieval
- Embedding-based similarity search with `sentence-transformers`
- Short, context-based answers from top relevant chunks
- Basic commands for loading, summary, and help

## Tech Stack

- Python
- `python-telegram-bot`
- `sentence-transformers`
- `PyPDF2`
- `torch`

## Project Structure

- `telegram_bot.py`: Telegram bot handlers and command flow
- `book_qa.py`: document processing, chunking, embeddings, and retrieval
- `.env.example`: environment template

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
```

4. Run:

```bash
python telegram_bot.py
```

## Usage

1. Start the bot and run `/load_book`
2. Upload a PDF or TXT file
3. Ask questions in chat

Available commands:

- `/start`
- `/help`
- `/load_book`
- `/summary`

## Notes

- Answers are retrieval-based and limited by extracted text quality.
- Scanned PDFs without selectable text may not work well.
- Embeddings and temporary files are stored locally.
