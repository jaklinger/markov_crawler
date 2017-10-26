import networkx as nx
from page_scorer import tokenize_alphanum
from page_scorer import WordSignificance

class GraphScorer(nx.Graph):
    def __init__(self,tokenized_sentences):
        super().__init__()
        
        words = []
        for _words in tokenized_sentences:
            self.add_edges(_words)            
            words += _words
            #print(_words)
        self.word_sig = WordSignificance(words)
        
        self.total = 0
        self.n_flush = 0
        self.cumulative_score = 1
        self.cumulator = False


    def two_grams(self,words):
        for i in range (0,len(words)-1):
            yield tuple(words[i:i+2])

    def add_edges(self,words):
        for edge in self.two_grams(words):
            super().add_edge(*edge)
        
    def flush_scores(self,reset=False):
        if reset:
            self.n_flush = 0
            self.total = 0
        else:
            if self.cumulative_score > 0:
                self.n_flush += 1
                #print("\t\tflushing",self.cumulative_score)
                self.total += self.cumulative_score
        self.cumulative_score = 0
        self.cumulator = False
    
    def get_score(self,text,reset=True):
        self.flush_scores(reset=reset)
        
        last_word = None
        for word in tokenize_alphanum(text):
            _score = self.word_sig[word]
            
            if word not in self.nodes():
                self.flush_scores()
                self.cumulative_score = _score
                self.flush_scores()
            else:
                if last_word not in self.nodes():
                    self.cumulative_score = _score
                else:
                    self.cumulator = word in self.neighbors(last_word)
                    if self.cumulator:
                        #print(last_word,word,"--->",self.cumulative_score,"*",_score)                        
                        self.cumulative_score *= _score
                    else:
                        self.flush_scores()
                        self.cumulative_score = _score
            last_word = word
        self.flush_scores()
        if self.n_flush > 0:
            return self.total/self.n_flush
        return 0

# Print out test score for a page
def test(site):
    from markov_crawler import text_from_soup
    import numpy as np
    from collections import Counter

    g.flush_scores(reset=True)
    r = requests.get(site)
    soup = BeautifulSoup(r.text,"lxml")
    scores = {text:g.get_score(text)
              for text in text_from_soup(soup)}

    print(site,len(scores),np.mean([s for _,s in scores.items()]))
    print(Counter(scores).most_common()[0:10])
    print()
    
    
if __name__ == "__main__":

    import requests
    from bs4 import BeautifulSoup
    
    r = requests.get("http://search.ucas.com/subject/fulllist")
    tokenized_sentences = []
    soup = BeautifulSoup(r.text,"lxml")
    for a in soup.find_all("a"):
        if "href" not in a.attrs:
            continue
        if not a.attrs["href"].startswith("/subject/"):
            continue
        tokenized_sentences.append(tokenize_alphanum(a.text))
    
    g = GraphScorer(tokenized_sentences)
    #print(g.get_score("business studies"))
    #print(g.get_score("business chocolate studies"))
    #print(g.get_score("joel and klinger"))
    # print(g.get_score("business and joel studies"))
    #print(g.get_score("french and islamic studies"))
    # print(g.get_score("french band islamic studies"))

    #print(g.get_score("Academic Departments"))
    #import sys
    #sys.exit()

    test("https://www.masdar.ac.ae/")
    test("https://www.masdar.ac.ae/join-us/work-with-us")    
    test("https://www.masdar.ac.ae/research-education/degree-offerings")

    test("https://www.manchester.ac.uk/")    
    test("http://www.manchester.ac.uk/study/experience/accommodation")
    test("http://www.manchester.ac.uk/study/undergraduate/courses/2018/")
    #print(g.neighbors("research"))
