import json, os, time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

class BM25:
    def __init__(self, docs, queries, cache = "cache/", load = True):
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
        
        If load is True, first it will look for a file in the directory <cache> which has the
        nameEnd = f"{len(docs)}_{len(queries)}.csv" if it finds then it uses it, else makes a new one.
        Else it directly makes a new one.
        
        Attributes:
            docs:       dict
            queries:    dict
            N:          int
            avdl:       int
            cache:      str
            nameEnd:    str
            D:          pandas Dataframe
            Q:          pandas Dataframe
            I:          list
        """
        self.docs = docs
        self.queries = queries
        self.N = len(docs)
        self.avdl = sum([len(d) for d in docs.values()]) / self.N
        self.cache = cache
        self.nameEnd = f"{len(docs)}_{len(queries)}.csv"  # start will be with either "d_" or "q_"
        
        # Checking if we have that file:
        has = False
        if load:
            files = os.listdir(cache)
            for l in files:
                if l.endswith(self.nameEnd):
                    has = True
                    break

        if has:
            print("Loading from cache...")
            self.D = pd.read_csv(cache + "d_" + self.nameEnd)
            self.Q = pd.read_csv(cache + "q_" + self.nameEnd)
            self.Is = self.Q.columns.tolist()[1:]
            
        else:
            print("Previous file not found...")
            self.vect(docs, queries)
        
    def vect(self, docs, queries):
        print("Catching vectorised fis and qfis for faster calculations (takes ~2 mins)")
        a = []
        for i in queries.values(): a += i
        Is = list(set(a))
        self.Is = Is
        
        print("Starting docs... ", end = "")
        doc = {}
        for name, cont in docs.items():
            doc[name] = {"dl": len(cont)}
            for i in Is:
                doc[name][i] = cont.count(i)
        doc = pd.DataFrame(doc).T
        print(doc)
        doc.to_csv(self.cache + "d_" + self.nameEnd)
        print("done")
        
        print("Starting queries... ", end = "")
        query = {}
        for name, cont in queries.items():
            query[name] = {}
            for i in Is:
                query[name][i] = cont.count(i)
        query = pd.DataFrame(query).T
        print(query)
        query.to_csv(self.cache + "q_" + self.nameEnd)
        print("done")
        
        self.D = doc
        self.Q = query

    def doc_part(self, k1, b):
        """BM25 Part 1: This is the part of the BM25 formula that is dependent on the document."""
        fi = self.D.loc[:, self.Is[0]:].to_numpy(dtype=np.int32)
        k = np.array(self.D.loc[:, "dl"]).reshape((len(self.D), 1))
        # print("k:", k.shape)
        # print("o:", o.shape)
        dl = k @ np.ones((1, len(self.Is)))
        # print("OK")
        return ((k1+1)*fi)/(fi+k1*(1-b+b*dl/self.avdl))

    def query_part(self, k2):
        """BM25 Part2: This is the part of the BM25 formula that is dependent on the query."""
        qfi = self.Q.loc[:, self.Is[0]:].to_numpy(dtype=np.int32)
        return ((k2+1)*qfi)/(qfi+k2)
    
    def idf(self):
        """BM25 Part 3: This is the last part of the BM25 formula that is independent of the document or the query."""
        fi = self.D.loc[:, self.Is[0]:].to_numpy(dtype=np.int32)
        k = np.count_nonzero(fi, axis = 0).reshape((1, len(self.Is)))
        ni = np.ones((len(self.D), 1)) @ k
        val = (self.N-ni+0.5)/(ni+0.5)
        return np.log(val)

    def get_scores(self, k1=0.25, k2=1.2, b=0.75):
        """Returns a 2D list of scores for each query."""
        p1 = self.doc_part(k1, b)*self.idf()
        p2 = self.query_part(k2).T
        return (p1 @ p2).T


# Some faltu functions
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
    
    # IMPORTING THE DATA:
    #   "cases.json" has the query and the doc_id of the relevant documents
    with open(loc+"cases.json") as f:
        prior_cases = json.load(f)
    # print_json(prior_cases, k=1)

    #   "Query_doc.json" has all the queries (X)
    with open(loc+"Query_doc.json") as f:
        query = json.load(f)
    # print_json(query)

    #   "answers.json" has the relevant documents (Y)
    with open(loc+"answers.json") as f:
        answers = json.load(f)
    # print_json(answers, 3, 1)

    model = BM25(prior_cases, query)
    
    t0 = time.time()
    scores = model.get_scores()
    print(f"took {time.time()-t0} seconds")
    
    print(scores.shape)
    print(scores)