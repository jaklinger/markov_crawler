'''
'''

from bs4 import BeautifulSoup
from collections import Counter
from fuzzywuzzy import process as fuzzy_process
import re
import requests
from nltk.corpus import stopwords

class PageScorer:
    ''''''
    def __init__(self):
        r = requests.get("http://search.ucas.com/subject/fulllist")
        words = []
        soup = BeautifulSoup(r.text,"lxml")
        for a in soup.find_all("a"):
            if "href" not in a.attrs:
                continue
            if not a.attrs["href"].startswith("/subject/"):
                continue
            words += tokenize_alphanum(a.text)
        self.word_sig = WordSignificance(words)

    def get_score(self,text):
        ''''''
        score = sum(self.word_sig[word] for word in tokenize_alphanum(text))
        #if score > 5:
        #    for word in tokenize_alphanum(text):
        #        print(word,self.word_sig[word])
        return score

class WordSignificance(Counter):
    ''' Inverse counter with fuzzy matching. Assigns a score to a word defined as:
    
    [fuzzy_score^n] x [count] x [base_score]
    
    where count is the word count from the training set,
    fuzzy_score is the highest fuzzy ratio of the test word and the training set,
    n is a damping parameter, by to which to reduce the significance of fuzzy matching
    '''
    def __init__(self,training_words,damping=3,max_count=10,
                 stops=set(stopwords.words("english"))):
        # Count the occurences of lower case'd words
        super().__init__(training_words)
        # Convert the count to a score
        for _word,_count in self.items():
            if _count > max_count:
                _count = max_count
            self[_word] = _count
        for _s in stops:
            if _s in self:
                del self[_s]
        # Also assign the fuzzy wuzzy process extractOne function
        self.extractOne = fuzzy_process.extractOne
        self.damping = damping
        
    def __getitem__(self,key):
        '''Modified getitem to apply lower case and fuzzy matching to a test set'''
        key = key.lower()
        # If an exact match exists
        if key in self.keys():
            _match = key
            _score = 100
        # Apply fuzzy matching (note: score in %, so divide by 100)
        else:
            _match,_score = self.extractOne(key,self.keys())
            # Normalise the score to word lengths, to account for
            # small words fitting inside big ones
            _norm = len(key)/len(_match)
            if _norm > 1:
                _norm = 1/_norm
            _score = _score * _norm
        return super().__getitem__(_match) * (_score/100)**self.damping


_stops = set(stopwords.words("english"))
''''''
def tokenize_alphanum(text):
    words = list(filter(('').__ne__, re.split('[^a-zA-Z\d]',text)))
    words = [_w.lower() for _w in words if _w not in _stops]
    return words
