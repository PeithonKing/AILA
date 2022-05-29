def average_precision(result, relevant):
    """
    Example:
    result = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]
              C1  [C2]  [C3]  C4  [C5]  C6       # [x] just means x is relevant doc
                    3     1         2
               
    relevant = ["C3", "C5", "C2"]
    """
    R = {}
    result = list(result)
    d = sorted(result, reverse=True)
    for i, s in enumerate(d):
        R[f"C{result.index(s)+1}"] = i+1
    ans = 0
    for i, r in enumerate(relevant):
        print(f"{i+1}/{R[r]} = {(i+1)/R[r]}")
        ans += (i+1)/R[r]
    print(f"getting AP: {ans} / {len(relevant)} = {ans / len(relevant)}\n")
    return ans / len(relevant)

def MAP(scores, answers):
    """
    Example:
    scores = [[0.7, 0.9, 0.8, 1.0, 0.2, 0.81],
              [0.47, 0.49, 0.48, 1.40, 0.42, 0.84],
              ...
              ...
              ... 50 such lines]
    answers = {
    	"AILA_Q1": ['C14', ......],
    	"AILA_Q2": ['C27', ......],
    	"AILA_Q3": ['C1', ......],
    	...
    	...
    	"AILA_Q48": ['C82', ......],
    	"AILA_Q49": ['C174', ......],
    	"AILA_Q50": ['C27', ......],
    }
    """
    n = len(scores)
    ans = 0
    for i in range(n):
        ans += average_precision(scores[i], answers[f"AILA_Q{i+1}"])
    return ans/n

if __name__ == "__main__":
    import json, os, time
    import numpy as np
    import pandas as pd
    from BM25_2 import BM25
    # result = {"C1": 0.7,  # rank = 5
    #           "C2": 0.9,  # rank = 2
    #           "C3": 0.8,  # rank = 4
    #           "C4": 1.0,  # rank = 1
    #           "C5": 0.2,  # rank = 6
    #           "C6": 0.81} # rank = 3
    # result = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]
    # relevant = ["C3", "C5", "C2"]
    # print(average_precision(result, relevant))
    
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
    # print(scores)
    print(MAP(scores.tolist(), answers)*100)