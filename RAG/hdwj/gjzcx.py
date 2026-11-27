# 关键字查询

from rank_bm25 import BM25Okapi
import jieba
class BM25:
    def __init__(self,documents: list[str]):
       Semantic_Chunk=documents

       self.Semantic_Chunk=Semantic_Chunk

       self.documents_doc=[(

           list(jieba.cut(doc)))
           for doc in Semantic_Chunk
       ]
       self.BM25=BM25Okapi(self.documents_doc)


    def kh(self,query:str,fanh:int=3):
        query_jb=list(jieba.cut(query))
        query_ss=list((self.BM25.get_scores(query_jb)))

        gjz_zpx=[]
        for idx,(doc,score) in enumerate(zip(self.Semantic_Chunk,query_ss)):
            gjz_zpx.append((idx, score,doc ))

        gjz_px=sorted(gjz_zpx,key=lambda x:x[1],reverse=True)
        return gjz_px[:fanh]






