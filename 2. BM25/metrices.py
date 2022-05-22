def rank(a, val):
    try: return sorted(a, reverse=True).index(val)+1
    except ValueError: return None

def average_precision(result, relevant):
    """
    Example:
    result = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]
              C1  [C2]  [C3]  C4  [C5]  C6       # [x] just means x is relevant doc
                    3     1         2

    relevant = ["C3", "C5", "C2"]
    """
    # print(result[:5], "...")
    # print(relevant,)
    AP = 0
    for GivenRankM1, DocName in enumerate(relevant):
        DocNum = int(DocName[1:])
        # PredRank = rank(result, result[DocNum-1])
        PredRank = sorted(result, reverse=True).index(result[DocNum-1])+1
        AP += (GivenRankM1+1)/PredRank
        # print(f"{GivenRankM1+1}/{PredRank} = {(GivenRankM1+1)/PredRank}")
    # print(f"getting AP: {AP} / {len(relevant)} = {AP/len(relevant)}\n")
    return AP/len(relevant)

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

if __name__=="__main__":
    # import json
    # from BM25_2 import BM25
    # import time
    print(average_precision([0.7, 0.9, 0.8, 1.0, 0.2, 0.81], ["C3", "C5", "C2"]))
    

    # # a = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]

    # # print(rank(a, 0.2))


    # loc = "../refining_seriously/"
    # with open(loc+"cases.json") as f:
    #     prior_cases = json.load(f)
    # with open(loc+"Query_doc.json") as f:
    #     query = json.load(f)
    # with open(loc+"answers.json") as f:
    #     answers = json.load(f)

    # model = BM25(prior_cases, query)

    # t0 = time.time()
    # score = model.get_scores()
    # print(f"Time taken for score = {time.time()-t0}")


    # print(score.shape)
    # print(len(answers))


    # t0 = time.time()
    # a = MAP(score, answers)
    # print(f"Time taken = {time.time()-t0}")
    # print(a)