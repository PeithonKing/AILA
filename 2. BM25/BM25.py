from rank_bm25 import BM25Okapi, BM25Plus, BM25L
from newap import average_precision
import json
import numpy as np
import matplotlib.pyplot as plt
import time

class BM25:
    def __init__(self, docs, queries):
        """
        D: dictionary of documents
        Q: dictionary of query terms
        k1, k2, b: parameters for the BM25 formula
        
        Example: 
        docs = {"C1": ['masud', 'khan', 'v', 'state', 'uttar', ......],
             "C2": ['indian', 'oil', 'corpor', 'v', 'nepc', ......],
             "C3": ['gurpal', 'singh', 'v', 'state', 'punjab', ......],
             ...
             ...
             "C2912": ['dharangadhara', 'chemic', 'work', 'limit', 'v', ......],
             "C2913": ['central', 'bank', 'india', 'v', 'sethumadhavan', ......],
             "C2914": ['som', 'raj', 'soma', 'v', 'state', ......],
            }
        
        queries = {"AILA_Q1": ['appel', 'februari', 'appoint', 'offic', 'grade', ......],
             "AILA_Q2": ['appel', 'us', 'examin', 'prime', 'wit', ......],
             "AILA_Q3": ['appeal', 'aris', 'judgment', 'learn', 'singl', ......],
             ...
             ...
             "AILA_Q48": ['whether', 'sanction', 'requir', 'initi', 'crimin', ......],
             "AILA_Q49": ['appel', 'patwari', 'work', 'villag', 'v1', ......],
             "AILA_Q50": ['peculiar', 'featur', 'appeal', 'special', 'leav', ......],
            }
        """
        self.D = docs
        self.Q = queries
        self.N = len(docs.keys())
        self.avdl = sum([len(d) for d in docs.values()]) / self.N

    def doc_part(self, doc, word, k1, b):
        """BM25 Part 1: This is the part of the BM25 formula that is dependent on the document."""
        fi = doc.count(word)
        dl = len(doc)
        return ((k1+1)*fi)/(fi+k1*(1-b+b*dl/self.avdl))

    def query_part(self, query, word, k2):
        """BM25 Part2: This is the part of the BM25 formula that is dependent on the query."""
        qfi = query.count(word)
        return ((k2+1)*qfi)/(qfi+k2)
    
    def ni(self, word):
        """Returns the number of documents that contain the word."""
        r = 0
        for doc in self.D.values():
            if word in doc:
                r += 1
        return r
    
    def df(self, word):
        """BM25 Part 3: This is the last part of the BM25 formula that is independent of the document or the query."""
        n = self.ni(word)
        val = (self.N - n + 0.5) / (n + 0.5)
        return np.log(val)

    def get_scores(self, k1=0.25, k2=1.2, b=0.75):
        """Returns a 2D list of scores for each query."""
        scores = []
        for query in self.Q.values():
            doc_score = []
            for doc in self.D.values():
                d = 0
                for word in query:
                    d += self.doc_part(doc, word, k1, b) * self.query_part(query, word, k2) * self.df(word)
                    print(1, end="", flush = True)
                doc_score.append(d)
                print("\ndoc_score: ", d)
            scores.append(doc_score)
        return scores


def namestr(obj, namespace = globals()):
    """Return the name of the variable, obj is stored in."""
    return [name for name in namespace if namespace[name] is obj][0]

def print_json(query, n = 3, m = 5, k=6):
    n = 3
    print(f"{namestr(query)} = "+"{\n", end="")  # start of the json
    l = sorted(list(query.keys()),
            key=lambda x: int(x[k:]))
    for QID in l[:n]:
        print('\t"'+QID+'":', query[QID][:m], "\b\b, ......],")
    for i in range(2): print("\t...")
    for QID in l[-n:]:
        print('\t"'+QID+'":', query[QID][:m], "\b\b, ......],")
    print("}")  # end of the json

if __name__ == "__main__":
    loc = "../refining_seriously/"
    
    # "cases.json" has the query and the doc_id of the relevant documents
    with open(loc+"cases.json") as f:
        prior_cases = json.load(f)
    # print_json(prior_cases, k=1)

    # "Query_doc.json" has all the queries (X)
    with open(loc+"Query_doc.json") as f:
        query = json.load(f)
    # print_json(query)

    # "answers.json" has the relevant documents (Y)
    with open(loc+"answers.json") as f:
        answers = json.load(f)
    # print_json(answers, 3, 1)


    model = BM25(prior_cases, query)
    print("started")
    scores = model.get_scores()