import os
import re
import json
import time
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Add it to the .env file.")

llm = Groq(api_key=api_key)
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")       # filters + rewrite
ANSWER_MODEL = os.getenv("GROQ_ANSWER_MODEL", "openai/gpt-oss-120b")  # final answer

db = chromadb.PersistentClient(path="chroma_db")
knowledge = db.get_collection("knowledge")
properties = db.get_collection("properties")

VALID_SOCIETIES = ["DHA", "Bahria Town", "Askari 5", "Scheme 33",
                   "Gulistan-e-Jauhar", "North Nazimabad", "Malir"]
VALID_TYPES = ["House", "Flat", "Plot"]

N_KNOWLEDGE = int(os.getenv("N_KNOWLEDGE", "3"))    # knowledge chunks sent to the LLM
N_PROPERTIES = int(os.getenv("N_PROPERTIES", "5"))  # property listings sent to the LLM
USE_CACHE = os.getenv("USE_CACHE", "1") == "1"
REWRITE_MAX_WORDS = 9  # longer messages are assumed to be complete questions
cache = {}  # standalone question (lowercase) -> reply

ANALYZE_PROMPT = f"""Analyze a question for a Karachi real estate agency. Return ONLY a JSON object:
{{"intent": "property_search" or "general",
 "society": one of {VALID_SOCIETIES} or null,
 "type": one of {VALID_TYPES} or null,
 "bedrooms": integer or null,
 "max_price_crore": number or null,
 "min_price_crore": number or null,
 "search_query": ALWAYS fill this. A clear English search query that keeps the key names (society, topic); translate Urdu or Roman Urdu fully}}
Rules: use "property_search" only when the user wants to find specific properties by area, size, bedrooms or budget; fees, discounts, policies, timings, installments, documents and societies are "general". The word "ghar" means House. Set a price field only for an explicit limit ("under", "above"); "a 6 crore house" is not a limit. Use null for anything not stated."""

ANSWER_PROMPT = """You are the assistant of Karachi Homes Realty, a Karachi real estate agency.
Rules:
1. Use ONLY the context. Never invent prices, fees, rules or properties.
2. If the context lacks the answer, say you do not have it and offer a human agent.
3. For properties, list matches briefly (area, size, bedrooms, price). If the context is empty, say none were found and suggest changing the budget or area.
4. Reply in the language of the user's own message (the "User wrote" line if present, otherwise the question): English -> English, Roman Urdu -> Roman Urdu. Never use Urdu or Arabic script.
5. Be short. Start with Yes or No ONLY when the question can be answered with yes or no; never for how much, what, which or how questions.
6. Write prices as "Rs" (for example Rs 1.7 crore), never the symbol for Indian rupees.
7. Discounts, negotiation and final prices are handled by a human agent: say so and offer to connect. Otherwise do not end replies with an agent offer.
8. This is demo data. Give no legal or tax advice."""

history = []  # list of (user_message, bot_reply)
last_stats = {}
calls = {}  # per-call token usage: name -> (prompt_tokens, output_tokens)


def rewrite(question, hist=None):
    """Turn a short follow-up into a standalone question. Returns (text, tokens)."""
    hist = history if hist is None else hist
    # Only short messages can be follow-ups, so skip the extra LLM call otherwise
    if not hist or len(question.split()) > REWRITE_MAX_WORDS:
        return question, 0
    recent = "\n".join(f"User: {q}\nBot: {a[:300]}" for q, a in hist[-3:])
    try:
        res = llm.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Rewrite the user's latest message as ONE complete standalone question or "
                    "request in English, using the chat history. The user may write Roman Urdu "
                    "with short forms: 'r' or 'or' = aur (and), 'kb' = kab (when), 'ha' = hai, "
                    "'ka/ki/ke' = of. A message like 'aur X ka?' or 'and X?' asks the same thing "
                    "as the previous question, but about X. Short replies such as 'yes', 'ok', "
                    "'please' accept the bot's last offer: rewrite them as a request for exactly "
                    "what the bot offered. If the message is already complete, return it "
                    "unchanged. Output only the rewritten text.\n"
                    "Example 1: History: User: What is the commission? Bot: 1 percent from the "
                    "buyer and seller; for rentals one month's rent. Message: r rental ka? "
                    "Output: What is the commission for rentals?\n"
                    "Example 2: History: User: office timings? Bot: Monday to Saturday 10 to 7. "
                    "Message: aur Sunday? Output: Is the office open on Sunday?")},
                {"role": "user", "content": f"Chat history:\n{recent}\n\nLatest message: {question}"},
            ],
            temperature=0,
            max_tokens=600,
            extra_body={"reasoning_effort": "low"},
        )
        text = (res.choices[0].message.content or "").strip()
        calls["rewrite"] = (res.usage.prompt_tokens, res.usage.completion_tokens)
        if not text:
            print("  (rewrite returned empty text)")
            return question, 0
        return text, res.usage.total_tokens
    except Exception as e:
        print("  (rewrite failed):", e)
        return question, 0


def analyze(question):
    """Ask the fast model for intent and search filters."""
    try:
        res = llm.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
            extra_body={"reasoning_effort": "low"},
        )
        text = res.choices[0].message.content or ""
        calls["analyze"] = (res.usage.prompt_tokens, res.usage.completion_tokens)
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return data, res.usage.total_tokens
    except Exception as e:
        print("  (analysis failed, using general search):", e)
        return {"intent": "general"}, 0


def build_where(f):
    """Turn the LLM filters into a Chroma 'where' filter. Price math is done here in code."""
    conds = []
    if f.get("society") in VALID_SOCIETIES:
        conds.append({"society": f["society"]})
    if f.get("type") in VALID_TYPES:
        conds.append({"type": f["type"]})
    if isinstance(f.get("bedrooms"), (int, float)):
        conds.append({"bedrooms": int(f["bedrooms"])})
    if isinstance(f.get("max_price_crore"), (int, float)):
        conds.append({"price": {"$lt": int(f["max_price_crore"] * 10_000_000)}})
    if isinstance(f.get("min_price_crore"), (int, float)):
        conds.append({"price": {"$gt": int(f["min_price_crore"] * 10_000_000)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


CARD_RE = re.compile(
    r"^(?:House|Flat|Plot) in (.+?), Karachi\. Size (\d+) sq yd \((\d+) sq ft\)"
    r"(?:, (\d+) bedrooms, (\d+) bathrooms)?\. Price .+\.$"
)


def fmt_price(p):
    if p >= 10_000_000:
        return f"{p / 10_000_000:g} crore"
    return f"{p / 100_000:g} lakh"


def make_cards(docs, metas):
    """Turn property search results into structured data for the chat page."""
    cards = []
    for doc, meta in zip(docs, metas):
        card = {
            "type": meta.get("type"),
            "society": meta.get("society"),
            "area": meta.get("society"),
            "bedrooms": meta.get("bedrooms"),
            "baths": None,
            "size_sqyd": meta.get("size_sqyd"),
            "size_sqft": None,
            "price_text": fmt_price(meta["price"]) if meta.get("price") else "",
        }
        m = CARD_RE.match(doc.strip())
        if m:
            card["area"] = m.group(1)
            card["size_sqft"] = int(m.group(3))
            card["baths"] = int(m.group(5)) if m.group(5) else None
        cards.append(card)
    return cards


def retrieve(question, analysis):
    """Search the right collection. Returns (documents, metadatas)."""
    query = analysis.get("search_query") or question
    if analysis.get("intent") == "property_search":
        res = properties.query(
            query_texts=[query],
            n_results=N_PROPERTIES,
            where=build_where(analysis),
        )
    else:
        res = knowledge.query(query_texts=[query], n_results=N_KNOWLEDGE)
    return res["documents"][0], res["metadatas"][0]


def answer(question, chunks, original=None, with_cards=0):
    """Write the final answer from the retrieved chunks only."""
    context = "\n\n".join(chunks) if chunks else "(empty)"
    extra = f"\nUser wrote: {original}" if original and original != question else ""
    if with_cards:
        extra += (f"\nNote: exactly {with_cards} properties are shown to the user as cards under "
                  f"your reply. Do not list them. Write 1 or 2 short sentences saying you found "
                  f"{with_cards} matching properties and inviting the user to refine the search.")
    res = llm.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}" + extra},
        ],
        temperature=0.2,
        max_tokens=1000,
        extra_body={"reasoning_effort": "low"},
    )
    calls["answer"] = (res.usage.prompt_tokens, res.usage.completion_tokens)
    text = res.choices[0].message.content or "(empty reply, please try again)"
    return text, res.usage.total_tokens


def ask(question, hist=None):
    start = time.time()
    hist = history if hist is None else hist
    calls.clear()
    question = question.strip()
    standalone, t0 = rewrite(question, hist)

    key = f"{question.lower()}|{standalone.lower()}"
    if USE_CACHE and key in cache:
        reply, cards = cache[key]
        hist.append((question, reply))
        last_stats.update(tokens=t0, filters="", standalone=standalone, intent="cache",
                          search="", seconds=round(time.time() - start, 1), cached=True,
                          cards=cards)
        print(f"  [cache hit | standalone={standalone!r} | tokens={t0}"
              f" | time={time.time() - start:.1f}s]")
        return reply

    analysis, t1 = analyze(standalone)
    chunks, metas = retrieve(standalone, analysis)
    is_property = analysis.get("intent") == "property_search" and bool(chunks)
    cards = make_cards(chunks, metas) if is_property else []
    reply, t2 = answer(standalone, chunks, question, with_cards=len(cards))
    if USE_CACHE:
        cache[key] = (reply, cards)
    hist.append((question, reply))
    elapsed = time.time() - start
    total = t0 + t1 + t2
    last_stats.update(tokens=total, filters=str(build_where(analysis)), standalone=standalone,
                      intent=analysis.get("intent"), search=analysis.get("search_query") or "",
                      seconds=round(elapsed, 1), cached=False, cards=cards)
    print(f"  [standalone={standalone!r} | intent={analysis.get('intent')}"
          f" | search={analysis.get('search_query')!r}"
          f" | filters={build_where(analysis)} | chunks={len(chunks)}"
          f" | tokens={total} | time={elapsed:.1f}s]")
    print("  tokens (prompt, output):", dict(calls))
    return reply


if __name__ == "__main__":
    print("Karachi Homes Realty assistant (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        print("\nBot:", ask(q))
