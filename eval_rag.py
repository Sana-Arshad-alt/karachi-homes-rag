import sys
import time
import statistics

sys.stdout.reconfigure(encoding="utf-8")

import rag

rag.USE_CACHE = False  # measure the real LLM path, not cached answers

# (question, keyword groups, filter strings)
# The answer passes if, for EVERY group, at least one keyword appears in it.
# For property questions we check the search filters instead.
TESTS = [
    ("How much commission do you charge?", [("1 percent", "1%", "1 %")], []),
    ("Is the office open on Sunday?", [("closed",), ("appointment",)], []),
    ("What time does the office open?", [("10",)], []),
    ("What is the down payment for a Bahria Town house?", [("30",)], []),
    ("Which plan has a 20 percent down payment?", [("askari", "plan d")], []),
    ("Can I get a bank mortgage?", [("bank",), ("eligib", "approval", "policy")], []),
    ("What is the difference between DHA and Bahria Town?",
     [("older", "established"), ("super highway", "installment", "affordable", "lower")], []),
    ("Is Bahria Town far from the city?", [("super highway", "outside", "longer")], []),
    ("What is token money?", [("advance", "reserve")], []),
    ("What is an NOC?", [("no objection",)], []),
    ("How many square feet is 120 sq yd?", [("1,080", "1080")], []),
    ("How many rupees is 4.5 crore?", [("45,000,000", "45000000", "45 million")], []),
    ("What happens if I miss an installment?", [("surcharge", "late", "cancel")], []),
    ("Do you charge for site visits?", [("free",)], []),
    ("Can I reschedule a site visit?", [("3 hours", "three hours")], []),
    ("Do I get my token money back if documents fail?", [("refund",)], []),
    ("What is the rental commission?", [("one month", "1 month")], []),
    ("Do you guarantee profit on property?", [("guarantee",), ("no", "not", "cannot")], []),
    ("Can I get a discount on a 6 crore house?", [("agent",)], []),
    ("Do you give legal advice?", [("legal",), ("not", "no ")], []),
    ("What will DHA prices be next year?",
     [("don't have", "do not have", "not have", "cannot", "can't", "no information"), ("agent",)], []),
    ("Which societies do you cover?", [("dha",), ("bahria",), ("scheme 33",)], []),
    ("Do you handle rentals?", [("yes",)], []),
    ("Bahria Town mein installment plan kya hai?", [("30",), ("25",)], []),
    ("Sunday ko office khula hai?", [("band", "closed"), ("appointment",)], []),
    ("Commission kitna lagta hai?", [("1 percent", "1%", "1 %")], []),
    ("Site visit kaise book hoti hai?", [("whatsapp", "website", "phone", "24")], []),
    ("Bank se loan mil sakta hai?", [("bank",), ("eligib", "approval", "policy", "income")], []),
    ("Token money wapas milta hai?", [("refund", "wapas")], []),
    # property questions: check the filters the bot built
    ("5 bedroom house in DHA under 8 crore", [], ["DHA", "'bedrooms': 5", "80000000"]),
    ("Show me flats in Bahria Town under 1 crore", [], ["Bahria Town", "Flat", "10000000"]),
    ("3 bedroom houses in Scheme 33", [], ["Scheme 33", "'bedrooms': 3", "House"]),
    ("Plots in DHA above 10 crore", [], ["DHA", "Plot", "100000000"]),
    ("DHA mein 10 crore se kam ke ghar dikhao", [], ["DHA", "House", "100000000"]),
]

def normalize(text):
    """Lowercase and replace special spaces and quotes with plain ones."""
    text = text.lower()
    for bad, good in {"\u202f": " ", "\u00a0": " ", "\u2011": "-",
                      "\u2019": "'", "\u2018": "'"}.items():
        text = text.replace(bad, good)
    return text


passed = 0
errors = 0
total_tokens = 0
total_time = 0.0
times = []
stopped_early = False

for i, (q, groups, filter_checks) in enumerate(TESTS, 1):
    rag.history.clear()
    rag.last_stats.clear()

    reply = None
    for attempt in range(3):
        try:
            start = time.time()
            reply = rag.ask(q)
            secs = time.time() - start
            break
        except Exception as e:
            err = str(e)
            print(f"\n{i:02d} ERROR (attempt {attempt + 1}) | {q} | {err[:250]}")
            low_err = err.lower()
            if "per day" in low_err:
                stopped_early = True  # daily limit: waiting a few seconds will not help
                break
            if "429" in err or "rate" in low_err or "limit" in low_err:
                time.sleep(25 * (attempt + 1))
            else:
                break
    if stopped_early:
        errors += len(TESTS) - i + 1
        print("\nDaily Groq limit reached. Stopping. Run again after the limit resets.")
        break
    if reply is None:
        errors += 1
        continue

    low = normalize(reply)
    answer_ok = all(any(k in low for k in g) for g in groups)
    filters_ok = all(s in rag.last_stats.get("filters", "") for s in filter_checks)
    ok = answer_ok and filters_ok and "\u20b9" not in reply

    passed += ok
    total_tokens += rag.last_stats.get("tokens", 0)
    total_time += secs
    times.append(secs)
    print(f"\n{i:02d} {'PASS' if ok else 'FAIL'} | {q}")
    if not ok:
        print("   reply:", reply[:300].replace("\n", " "))
    time.sleep(2)  # stay under the free-tier rate limit

n = len(TESTS)
answered = n - errors
print("\n" + "=" * 60)
if answered:
    print(f"Score: {passed}/{answered} answered = {passed / answered:.0%}")
    print(f"Average tokens per question: {total_tokens / answered:.0f}")
    print(f"Average time per question: {total_time / answered:.1f}s")
    print(f"Median time per question: {statistics.median(times):.1f}s")
print(f"Questions that failed with an error (not scored): {errors} of {n}")
if errors:
    print("WARNING: this run is NOT a valid score. Fix the errors above and run again.")
