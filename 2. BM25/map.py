def average_precision(result, relevant):
    '''
    Example:
    result = {"C5": 0.7,
              "C17": 0.9,
              "C6": 0.8,
              "C8": 1.0,
              "C89": 0.2,
              "C101": 0.81}
    relevant = ["C5", "C6", "C7"]
    '''
    # Sort the result according to the score,
    # now from a dictionary, result becomes a list
    result = sorted(result.keys(),
                    key=lambda x: result[x],
                    reverse=True)
    
    # Calculating the precision values
    precision = [((i+1)/(result.index(relevant[i])+1))
                     for i in range(len(relevant))
                     if relevant[i] in result]

    return sum(precision)/len(relevant)

if __name__ == "__main__":
    result = {"C5": 0.7,
              "C17": 0.9,
              "C6": 0.8,
              "C8": 1.0,
              "C89": 0.2,
              "C101": 0.81}
    relevant = ["C5", "C6", "C7"]
    print(map(result, relevant)) 