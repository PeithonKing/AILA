import translators as ts
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# import os
import time
import pandas as pd
from tqdm import tqdm
tokenizer = nltk.RegexpTokenizer(r"\w+")
ps = PorterStemmer()
ts.google("I love Elephants!", "en", "bn")
langs = list(ts._google.language_map.keys())
bad_langs = ['ay', 'bm', 'doi', 'dv', 'ee', 'lus', 'mni-Mtei'] + ["en"]
for x in bad_langs:
    langs.remove(x)

loc = "../refining_seriously/"
def namestr(obj, namespace = globals()):
    return [name for name in namespace if namespace[name] is obj][0]

def print_json(query, n = 3, m = 5, k=6):
    n = 3
    print(f"{namestr(query)} = "+"{\n", end="")  # start of the json
    l = sorted(list(query.keys()),
            key=lambda x: int(x[k:]))
    for QID in l[:n]:
        print('\t"'+QID+'":', query[QID][:m], "\b\b, ......],")
    for i in range(2):
        print("\t...")
    for QID in l[-n:]:
        print('\t"'+QID+'":', query[QID][:m], "\b\b, ......],")
    print("}")  # end of the json

def process(string,
            tokenizer = nltk.RegexpTokenizer(r"\w+"),
            ps = PorterStemmer(),
            stopwords = stopwords.words('english')):
    '''
    - A function to process a string and return a list of tokens.
    - We tokenize the string, remove stopwords and numbers, and
        finally stem the tokens to keep them in a list.
    - This function will be used in all cases uniformly so that 
        we can compare "APPLES WITH APPLES".
    '''
    string = tokenizer.tokenize(string.lower()) # tokenize
    tokens = [ps.stem(fl) for fl in string # stem tokens
                if not fl.isnumeric() and # remove numbers
                fl not in stopwords] # remove stopwords
    return tokens # takes string as input and returns a list

def augment(string, toLang, threshold = 4900):
    string = string.split("\n")
    text = []

    for line in string:
        if len(line)>threshold:
            text += line.split(".")
        elif line != "":
            text.append(line)

    ans = []
    for tex in tqdm(text):
        oth = ts.google(tex, "en", toLang)
        eng = ts.google(oth, toLang, "en")
        ans.append(eng)

    return ". ".join(ans)

# with open(loc+"cases.json") as f:
#     prior_cases = json.load(f)
# with open(loc+"Query_doc.json") as f:
#     query = json.load(f)
# with open(loc+"answers.json") as f:
#     answers = json.load(f)

# Opening the file to know which name to use while saving as json to avoid conflicts
with open("log/name.txt") as f: SaveName = f.read() + ".json"
# Update the same file to mark that it has taken the previous name
with open("log/name.txt", "w") as f: f.write(str(int(SaveName[:-5])+1))

with open(f"log/outputs/{SaveName}", "w") as f:
	json.dump({}, f)

while True:
    with open("log/todo.txt") as f:
        todo = f.read().split()
    with open("log/doing.txt") as f:
        doing = f.read().split()
    
    do = None
    # Find the document to work upon
    for i in todo:
        if i not in doing:
            do = i
            print(f"doing {do}")
            break
    if do == None:
        print("completed! YaY!")
        break
    
    # Write it in the register so that no one else takes it up
    with open("log/doing.txt") as f:
        w = f.read() + " " + do
    with open("log/doing.txt", "w") as f:
        f.write(w)
    
    
    with open(f"log/outputs/{SaveName}") as f:
        docs = json.load(f)
    
    with open(f"../Object_casedocs/{do}.txt") as f:
        document = f.read()
     
    docName = do+".000"
    docs[docName] = process(document)
    for lang in langs[:10]:
        docName = "C" + str(float(docName[1:])+0.001)
        try:
            docs[docName] = process(augment(document, lang))
        except:
            pass
    
    with open(f"log/outputs/{SaveName}", "w") as f:
        json.dump(docs, f)
    
    with open("log/doing.txt") as f:
        w = f.read().split()
    w.remove(do)
    with open("log/doing.txt", "w") as f:
        f.write(" ".join(w))
    
    with open("log/todo.txt") as f:
        w = f.read().split()
    w.remove(do)
    with open("log/todo.txt", "w") as f:
        f.write(" ".join(w))
    
    print("\n")