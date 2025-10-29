import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Sample text
text = "This is a simple example to demonstrate text preprocessing, including stop word removal!! %and stemming."

# Tokenize the text
words = word_tokenize(text)
print("After Tokenization:", words)
# Convert to lowercase
words = [word.lower() for word in words]
print("After Lowercasing:", words)
# Remove Punctuations
words = [word for word in words if word not in string.punctuation]
print("After Removing the Punctuations :",words)
# Remove stop words
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]
print("After Stop Word Removal:", words)
# Perform stemming
stemmer = PorterStemmer()
words = [stemmer.stem(word) for word in words]
print("After Stemming:", words)

