def average_precision(result, relevant):
    '''
    Example:
    result = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]
              C1  [C2]  [C3]  C4  [C5]  C6
                    3     1         2       
               
    relevant = ["C3", "C5", "C2"]
    '''
    R = {}
    result = list(result)
    for i, s in enumerate(sorted(result, reverse=True)):
        R[f"C{result.index(s)+1}"] = i+1
    ans = 0
    for i, r in enumerate(relevant):
        ans += (i+1)/R[r]
    return ans / len(relevant)

if __name__ == "__main__":
    # result = {"C1": 0.7,  # rank = 5
    #           "C2": 0.9,  # rank = 2
    #           "C3": 0.8,  # rank = 4
    #           "C4": 1.0,  # rank = 1
    #           "C5": 0.2,  # rank = 6
    #           "C6": 0.81} # rank = 3
    result = [0.7, 0.9, 0.8, 1.0, 0.2, 0.81]
    relevant = ["C3", "C5", "C2"]
    print(average_precision(result, relevant))