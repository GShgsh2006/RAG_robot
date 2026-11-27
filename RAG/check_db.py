import chromadb

client = chromadb.PersistentClient(path='hdwj/chroma_db')
coll = client.get_collection('psychology_knowledge')
print('count:', coll.count())
docs = coll.get()
print('keys:', list(docs.keys()))
if docs['metadatas']:
    print('metadatas sample:', docs['metadatas'][:5])
else:
    print('metadatas: empty')