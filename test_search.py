import chromadb

client = chromadb.PersistentClient(path="chroma_db")
knowledge = client.get_collection("knowledge")
properties = client.get_collection("properties")


def show(title, res):
    print("\n" + "=" * 70)
    print("Q:", title)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n  [{meta.get('source', 'property')}]  distance={dist:.3f}")
        print("  ", doc[:200].replace("\n", " "))


for q in ["How much commission do you charge?", "Is the office open on Sunday?",
          "Bahria Town installment plan", "What is the difference between DHA and Bahria Town?"]:
    show(q, knowledge.query(query_texts=[q], n_results=3))

res = properties.query(
    query_texts=["5 bedroom house in DHA"], n_results=5,
    where={"$and": [{"society": "DHA"}, {"type": "House"}, {"bedrooms": 5}, {"price": {"$lt": 80_000_000}}]},
)
show("5 bedroom house in DHA under 8 crore (with filters)", res)
