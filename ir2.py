# simple corrected matching code
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))
doc_dict = [
    ["d1", "Welcome to hotel heaven such a lovely place"],
    ["d2", "She is buying a stairway to heaven"],
    ["d3", "Don't make it bad"],
    ["d4", "Take me to the heaven"]
]

def normalize_and_filter(text):
    # tokenize
    tokens = word_tokenize(text)
    # lowercase, remove punctuation-only tokens, remove stopwords
    filtered = [t.lower() for t in tokens if t.lower() not in stop_words and any(ch.isalnum() for ch in t)]
    return set(filtered)

doc_sets = {key: normalize_and_filter(val) for key, val in doc_dict}

query = input("Enter query: ").strip()
query_set = normalize_and_filter(query)

doc_with_best_match = []   # any overlap
doc_with_exact_match = []  # query contained entirely in doc

for key, docset in doc_sets.items():
    if len(query_set & docset) > 0:        # any common token
        doc_with_best_match.append(key)
    if query_set.issubset(docset):         # all query tokens are in doc
        doc_with_exact_match.append(key)

print("Best match (any query token present):", doc_with_best_match)
print("Exact match (all query tokens present):", doc_with_exact_match)
