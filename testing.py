from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("LangChain is awesome.")
print(embedding[:5])  # print first 5 numbers
