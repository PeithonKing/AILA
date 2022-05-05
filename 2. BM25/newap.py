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
    for i, s in enumerate(sorted(result, reverse=True)):
        R[f"C{result.index(s)+1}"] = i+1
    ans = 0
    for i, r in enumerate(relevant):
        ans += (i+1)/R[r]
    return ans / len(relevant)

def MAP(scores, answers):
    """
    Example:
    scores = [[0.7, 0.9, 0.8, 1.0, 0.2, 0.81],
              [0.47, 0.49, 0.48, 1.40, 0.42, 0.84]]
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
        ans += average_precision(scores[i], answers[i])
    return ans/n

# if __name__ == "__main__":
#     # result = {"C1": 0.7,  # rank = 5
#     #           "C2": 0.9,  # rank = 2
#     #           "C3": 0.8,  # rank = 4
#     #           "C4": 1.0,  # rank = 1
#     #           "C5": 0.2,  # rank = 6
#     #           "C6": 0.81} # rank = 3
#     result = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]
#     relevant = ["C3", "C5", "C2"]
#     print(average_precision(result, relevant))