import os
import glob
import pandas as pd
import chromadb

client = chromadb.PersistentClient(path="chroma_db")

# Recreate both collections from scratch on every run
for name in ["knowledge", "properties"]:
    try:
        client.delete_collection(name)
    except Exception:
        pass

knowledge = client.create_collection("knowledge", metadata={"hnsw:space": "cosine"})
properties = client.create_collection("properties", metadata={"hnsw:space": "cosine"})


def chunk_markdown(path):
    """Split a markdown file into chunks: one chunk per '###' section,
    prefixed with the '##' heading above it."""
    name = os.path.basename(path)
    chunks = []
    h2 = ""
    buf = []

    def flush():
        body = "\n".join(buf).strip()
        if len(body) > 60:
            chunks.append(f"[{name}] {h2}\n{body}")

    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# ") or line.startswith("NOTE:"):
            continue  # skip the file title and the demo disclaimer
        if line.startswith("## "):
            flush()
            buf = []
            h2 = line[3:].strip()
        elif line.startswith("### "):
            flush()
            buf = [line[4:].strip()]
        else:
            buf.append(line)
    flush()
    return chunks


# 1. Knowledge files -> "knowledge" collection
ids, docs, metas = [], [], []
for path in sorted(glob.glob("data/knowledge/*.md")):
    name = os.path.basename(path)
    for i, chunk in enumerate(chunk_markdown(path)):
        ids.append(f"{name}-{i}")
        docs.append(chunk)
        metas.append({"source": name})
knowledge.add(ids=ids, documents=docs, metadatas=metas)
print("Knowledge chunks:", knowledge.count())

# 2. Properties -> "properties" collection (with filterable metadata)
df = pd.read_csv("data/clean/properties.csv")
properties.add(
    ids=df["id"].tolist(),
    documents=df["description"].tolist(),
    metadatas=[
        {
            "society": r["society"],
            "type": r["type"],
            "bedrooms": int(r["bedrooms"]),
            "size_sqyd": int(r["size_sqyd"]),
            "price": int(r["price"]),
        }
        for _, r in df.iterrows()
    ],
)
print("Property documents:", properties.count())
print("Done.")
