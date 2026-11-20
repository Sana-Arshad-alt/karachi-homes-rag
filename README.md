# Karachi Homes Realty: RAG Assistant

A chatbot for a (fictional) Karachi real estate agency. Customers ask in **English or Roman Urdu**. The bot answers from company documents and from real property listings, and refers the customer to a human agent when the answer is not in the data.

## What it does

- Answers questions about commission, office timings, payment plans, societies and the buying process from the knowledge documents.
- Searches **1,829 Karachi listings** by society, type, bedrooms and budget, and shows each result as a property card.
- Understands short follow-ups ("r rental ka?", "yes plz") by rewriting them into a full question.
- Does not guess: discounts, negotiation and anything missing from the data go to a human agent.
- Login / sign up, a separate chat history per user, and browser voice input (speak your question) with optional read-aloud.

## How a question is answered

```
question -> rewrite (follow-ups only) -> analyze (intent, filters, English search query)
         -> search (knowledge docs OR listings with filters) -> answer from the retrieved text only
```

Price maths (for example "8 crore" to rupees) is done in Python, not by the LLM. Repeated questions are served from a cache.

## Data (please read)

| Data | Source |
|---|---|
| Property listings | **Real**: Zameen.com listings from the Kaggle dataset "Zameen.com Property Data Pakistan 2023" (Karachi rows, cleaned) |
| FAQ, societies, payment plans, company policy | **AI-generated demo documents** (fees, plans and policies are fictional) |
| Test questions | Written by us |

Property type (house / flat / plot) is inferred from the listing name and bedrooms, because the dataset has no type column. Prices are a 2023 snapshot.

## Tech

Python 3.12, ChromaDB (vector search), Groq (`openai/gpt-oss-20b` for understanding the question, `openai/gpt-oss-120b` for the answer), FastAPI, plain HTML/JS front end, SQLite for users and chat history.

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and put your Groq key in it.
2. Download the Kaggle CSV and place it at `data/raw/archive/zameen.csv` (see `data/raw/PUT_ZAMEEN_CSV_HERE.txt`).
3. Build the data and the database:
```
python clean_data.py
python ingest.py
```
4. Start the app and open http://127.0.0.1:8000
```
python api.py
```

Optional: `python eval_rag.py` runs 34 test questions and prints score, tokens and time.

## Results

On our own 34-question test set the bot passed 34/34 in the last complete run, and average tokens per question went from about 1,224 to about 1,004 (shorter prompts, 3 knowledge chunks instead of 4, a cache, and listings shown as cards instead of generated text). The property cards, strict "under / above" price filters, login and voice features were added after that run, so re-run `python eval_rag.py` to refresh the numbers. A smaller model for answers was tried and dropped because accuracy fell to 94%.

## Security notes (demo level)

Passwords are stored as salted PBKDF2 hashes, session tokens are stored hashed, the cookie is HttpOnly, and login and chat are rate limited. There is no HTTPS, e-mail verification or password reset, so this is not production ready.

## Limitations

- The listings are a snapshot, not live data. A scheduled refresh is a planned next step.
- The 34 test questions were written by the same team that wrote the documents, so the score is optimistic.
- Very short Roman Urdu messages can be answered in English.
- Voice input uses the browser's speech recognition (Chrome or Edge); audio may be processed by the browser vendor's service. Read-aloud works best for English answers.
- The Groq free tier has usage limits.

## Credits

UI inspired by "Find your home" by Ronas IT and the colour palette of tubik (Dribbble).
