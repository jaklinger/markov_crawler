'''
TODO: Usage instructions
TODO: Setup instructions, including required table names, such as top_url etc
'''

# new idea for threshold, perhaps give bonus related to score?
# Work out how to drop without hacking
# Need to do proper analysis of course pages vs non course pages to optimise scorer
# Add extra universities (get these from someone)

from bs4 import BeautifulSoup
from collections import Counter
import logging
import numpy as np
#from page_scorer import PageScorer
from page_scorer import tokenize_alphanum
import random
import requests
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy.sql import text as sql_text
from sqlalchemy.sql.expression import select as sql_select
import sys
import time
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.parse import urldefrag

from graph_scorer import GraphScorer
from bs4.element import Comment
import threading

'''Tag text as visible'''
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

'''Extract set of all visible text from soup'''
def text_from_soup(soup):
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    texts = set(t for t in visible_texts if t.strip() != "")
    return texts
#return u" ".join(t.strip() for t in visible_texts)

class KeyValueTables(dict):
    '''A collection of _KeyValueTable objects'''    
    def __init__(self):
        self.metadatas = []

    '''Override setitem to generate the _KeyValueTable'''
    def __setitem__(self,table_name,kwargs):
        # Append metadata to list of metadatas and kwargs
        if kwargs["conn"] is not None:
            metadata = MetaData(bind=kwargs["conn"].engine)
            kwargs["metadata"] = metadata
            if metadata not in self.metadatas:
                self.metadatas.append(metadata)
        # Instantiate the key value table from the kwargs
        _kvt = _KeyValueTable(table_name,**kwargs)
        # and then assign it to this key=table_name
        super().__setitem__(table_name,_kvt)

    '''Wrapper to execute all metadata create all statements'''
    def create_all(self):
        for m in self.metadatas:
            m.create_all()
            
    ''' Wrapper to execute all conn close statements'''
    def close(self):
        for _,table in self.items():            
            if (table.conn is not None) and not table.conn.closed:
                table.conn.close()
                
class _KeyValueTable:
    '''
    A wrapper around sqlalchemy Table to generate a 2-column table 
    of the form "key : score", intended for recording scores of
    keywords for the markov chain procedure. Creates the table if none 
    exists. Also features a handy select_all and insert function.
    '''
    def __init__(self,table_name,conn,key,value,metadata=None):
        if conn is not None:
            self.table = Table(table_name,metadata,
                               Column(key,Text,nullable=False),
                               Column(value,Float,nullable=False),)        
        self.key = key
        self.value = value
        self.table_name = table_name
        self.conn = conn

    '''Get all scores for this table in a dictionary of the form {"word":"score"}'''        
    def select_all(self):
        if self.conn is None:
            return {}
        
        # SELECT key,value FROM table_name
        results = self.conn.execute(sql_select([sql_text(self.key),
                                                sql_text(self.value)],
                                               from_obj=sql_text(self.table_name)))
        # Fetch results in chunks, and store in 'scores'
        scores = {}
        max_fetch = 1000
        _all_results = results.fetchmany(max_fetch)
        while len(_all_results) > 0:            
            for _word,_score in _all_results:
                scores[_word] = _score
            _all_results = results.fetchmany(max_fetch)
        return scores

    '''Insert new scores (dict) into the table in the form {"word","score"}'''
    def insert(self,new_scores):
        if self.conn is None:
            return        
        _scores = [{self.key:_k,self.value:float(_v)} for _k,_v in new_scores.items()]
        self.conn.execute(self.table.insert(),_scores)

class MarkovCrawler:
    '''
    Crawl through pages from a given top URL with the aim of finding a target page.
    The choice of each each subsequent URL is performed using Monte Carlo 
    selection based on a score assigned to the URL, which should
    reflect the probablity for a given URL to contain the target page.
    Each web page is also given a score, which should reflect a measure of 
    confidence that the page is the target page.
    '''
    def __init__(self,top_url=None,max_depth=4,page_scorer=None,threshold=None,
                 label="test",is_dummy=False,engine_url=None,log_file=None):
        # Crawler settings
        self.alpha = 1.0 # Initial word score
        self.epsilon = 0.1 # Successful word score
        self.beta = 1.0 # Unsuccessful word increment
        self.max_depth = max_depth # Maximum search depth
        self.threshold = threshold # Successful score threshold 

        # Other attributes
        self.open_transactions = [] # List of open SQL transactions
        self.url_scores = {} # Scores assigned to each URL
        self.active_words = [] # A list of words in the current crawling session
        self.label = label # A suffix for the temporary SQL table names created 
                           # whilst crawling

        # dummy mode: this is intended if the user only wants a hook
        # to read or write to a KeyValueTable, without wanting to perform
        # any web crawling.
        self.is_dummy = is_dummy 

        # Get the logger
        self.logger = logging.getLogger(__name__)
        if log_file is not None:
            handler = logging.FileHandler(log_file,mode="w")
            self.logger = logging.getLogger(label)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.getLogger(__name__).level)
            self.logger.propagate = False
        
        # Assign a page scoring function, that take one argument (a string)
        # which it analyses and returns a score
        self.page_scorer = page_scorer 

        # Create tables if they don't already exist
        self.conn = self.connect(engine_url)
        self.conn_tmp = self.connect(engine_url,"_tmp")

        # Generate the KeyValueTables
        self.tables = KeyValueTables()
        self.tables["word_scores"] = dict(key="word",value="score",conn=self.conn)
        self.tables["page_scores"] = dict(key="page",value="score",conn=self.conn)
        if not is_dummy:
            self.tables["page_scores_"+label] = dict(key="page",value="score",conn=self.conn_tmp)
            self.tables["word_scores_"+label] = dict(key="word",value="score",conn=self.conn_tmp)
        self.tables.create_all()

        # Get word scores
        self.scores = self.tables["word_scores"].select_all()
        self.pages = self.tables["page_scores"].select_all()
        self.logger.debug("Got %d word scores from the database.",(len(self.scores)))
        self.logger.debug("Got %d page scores from the database.",(len(self.pages)))

        # For optimisiation
        self.raw_score = {}
        
        # If a top URL has been specified, otherwise select a random top URL
        # from a table called "top_urls"
        if top_url is not None:
            self.top_url = top_url
        elif self.conn is not None:
            # SELECT url FROM top_urls;
            results = self.conn.execute(sql_select([sql_text("url")],
                                                   from_obj=sql_text("top_urls")))
            self.top_url = random.choice([_r["url"] for _r in results.fetchall()])
        else:
            raise RuntimeError("Neither top url nor connection URL has been set")
        self.current_url = self.top_url
            
        # Record the top URL stub, in order to identify external sites
        parse_result = urlparse(self.top_url)
        self.top_url_stub = ".".join(parse_result.netloc.split(".")[-3:])

        # If page scores exist, then set the threshold to 75% of the highest score
        # TODO arbitrary limit
        if len(self.pages) > 0 and self.top_url in self.pages:
            _page,max_score = Counter(self.pages).most_common(1)[0]
            homepage_score = self.pages[self.top_url]
            _threshold = 0.75 * max_score / homepage_score
            if _threshold > self.threshold:
                self.threshold = _threshold
                self.logger.debug("Set the threshold to %s which corresponds to %s.",
                                  _threshold,_page)
                

    '''
    Convenience method to connect to a DB via a sql_alchemy engine,
    whilst also keeping track of open transactions.
    '''
    def connect(self,engine_url,extension=""):
        if engine_url == None:
            return None
            
        engine = create_engine(engine_url+extension+"?charset=utf8")
        conn = engine.connect()
        if not self.is_dummy:
            self.open_transactions.append(conn.begin())
        return conn

    '''The "main" crawling method'''        
    def crawl(self,url=None,depth=1):

        # Preparation for this crawl:
        # 1) (Re)initialise url_scores
        self.url_scores = {}
        # 2) Break out if reached max_depth
        if depth > self.max_depth:
            self.logger.debug("Breaking out")
            return False    
        # 3) Set the url if None
        if url is None:
            url = self.top_url        
        # 4) Set the current URL
        if not url.startswith("http"):
            url = urljoin(self.current_url,url)
        self.current_url = url.rstrip("/")
        self.logger.debug("Trying %s",(self.current_url))
        
        # Get this page's HTML
        try:
            r = requests.get(self.current_url)
            r.raise_for_status()
            # Update the URL (in case of forwarding)
            self.current_url = r.url
        except (requests.exceptions.HTTPError,
                requests.exceptions.InvalidSchema,
                requests.exceptions.ConnectionError) as err:
            self.logger.warning("Got error %s for URL %s",str(type(err)),
                                                           self.current_url)
            return False
        
        # Get the request text (not using r.text since
        # PDF and binary files hang on r.text call)
        try:
            html = "\n".join(x.decode("utf-8") for x in r.iter_lines())
        except UnicodeDecodeError:
            self.logger.warning("Ignoring suspicious path %s",(self.current_url))
            return False
        soup = BeautifulSoup(html,"lxml")

        # Accrue the page score, unless previously done
        # (otherwise the page score can be acquired through self.pages[url]
        page_previously_done = (url in self.pages)
        if page_previously_done:
            self.logger.debug("\tAlready found this URL before")
        
        # Iterate through links on the page to calculate the score
        page_score = 0
        anchors = soup.find_all("a")
        for a in anchors:
            # If not a link, skip
            if "href" not in a.attrs:
                continue            
            # Ignore '#' fragments or internal links
            link_url = urldefrag(a["href"])[0].rstrip("/")
            if not link_url:
                continue
            if link_url == self.current_url or link_url.strip() == '':
                continue
            # Ignore external links
            parse_result = urlparse(link_url)
            if parse_result.netloc != '':
                if self.top_url_stub not in parse_result.netloc:
                    continue
            if link_url.startswith("mailto"):
                continue
                
            # Calculate the 'Markov score' (a relative weight used for 
            # Monte Carlo selection) and page score for this URL and link text
            link_text = a.text
            mkv_score = self.calculate_markov_score(link_text)
            if mkv_score > 0:
                self.url_scores[link_url] = mkv_score
            # Don't reprocess previously visited pages
            # if not page_previously_done:
            #     # Normalise the score to the number of links per page
            #     _score = self.page_scorer(link_text)
            #     #self.logger.debug("\t\tGot page score %0.2f / %0.2f",_score,np.sqrt(len(anchors)))
            #     page_score += _score / np.sqrt(len(anchors))
        
        if not page_previously_done:
            self.raw_score[url] = 0
            # page score = mean(score per word on page)
            texts = text_from_soup(soup)
            first_time = True
            for text in texts:
                _page_score = self.page_scorer(text,reset=first_time)
                page_score += _page_score
                self.raw_score[url] += _page_score
                first_time = False
            if len(texts) > 0:
                page_score = page_score/len(texts)
            else:
                page_score = 0
            
        # Record the non-normalised score
        if not page_previously_done:
            self.pages[url] = page_score
        else:
            page_score = self.pages[url]
            
        # Normalise the score to the home page
        normalised_score = page_score/self.pages[self.top_url]
        self.logger.debug("\t\t ===> Got a normalised page score of %0.2f from %0.2f.",normalised_score,page_score)
        if normalised_score >= self.threshold:
            self.logger.info("\t\t%s is over threshold.",(self.current_url))
            # Reduce the score of successful words
            for word in self.active_words:
                self.scores[word] = self.epsilon
            return True

        if len(self.url_scores) == 0:
            return False        
        return self.crawl(url=self.select_url(),depth=depth+1)

    '''
    Initialise any active word scores, then reset active words.
    Should be called before every iteration of "crawl(...)"
    '''
    def prepare(self):
        for word in self.active_words:
            self.scores[word] += self.beta
        # Reset
        self.active_words = []

    '''
    Return a score based on the average weight of 
    words within the link text.
    '''
    def calculate_markov_score(self,link_text):
        total_score = 0
        words = tokenize_alphanum(link_text)
        # If there are no words, ignore
        if len(words) == 0:
            return 0
        # Sum the score for each words
        for word in words:
            # Ignore any numeric-containing word:
            if any(char.isdigit() for char in word):
                continue
            # Append to active words
            self.active_words.append(word)
            # If the word has not been previously found, then
            # inititalise to <alpha>
            if word not in self.scores:
                self.scores[word] = self.alpha
            total_score += self.scores[word]
        return total_score/len(words)

    '''
    Monte Carlo selection of URLs based on weights calculated in
    `calculate_markov_score`.
    '''
    def select_url(self):
        #return "https://www.masdar.ac.ae/research-education/degree-offerings"
        #return "http://www.manchester.ac.uk/study/undergraduate/courses/2018/"
        #return "https://www.research.manchester.ac.uk/portal/en/facultiesandschools/search.html"
        urls=[]
        weights=[]
        for k,v in self.url_scores.items():
            weights.append(1/v)
            urls.append(k)        
        return np.random.choice(urls,p=np.array(weights)/sum(weights))

    '''Save database and close connection on exit'''
    def __exit__(self, exc_type, exc_value, traceback):
        # In the case of a clean exit:
        if exc_type == None:
            # If in regular mode, then insert values
            if not self.is_dummy and len(self.tables) > 0:
                self.tables["page_scores_"+self.label].insert(self.pages)
                self.tables["word_scores_"+self.label].insert(self.scores)
            # Then commit
            for trans in self.open_transactions:
                trans.commit()
        # Otherwise, rollback
        else:
            for trans in self.open_transactions:
                trans.rollback()                
        self.tables.close()

    '''Method for use with "with" statement'''
    def __enter__(self):
        return self
    
''''''
def run(logging_level=logging.INFO,**kwargs):
    
    # Set the logging level
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.basicConfig(stream=sys.stderr,level=logging_level)
    
    # Instantiate the crawler
    with MarkovCrawler(**kwargs) as crawler:
        n_tries = 0
        finished = False
        while not finished:
            crawler.prepare()
            finished = crawler.crawl()
            n_tries += 1
            if n_tries == 5:
                break


''''''
def reduce_scores(num_total_jobs,engine_url):

    logging.info("Clearing out old tables")
    engine = create_engine(engine_url+"?charset=utf8")
    conn = engine.connect()
    conn.execute("DROP TABLE IF EXISTS page_scores;")
    conn.execute("DROP TABLE IF EXISTS word_scores;")
    conn.close()

    # Reduce the results
    logging.info("Going to generate the new tables")
    with MarkovCrawler(is_dummy=True,engine_url=engine_url) as crawler:        
        for i in range(num_total_jobs):
            crawler.tables["word_scores_"+str(i)] = dict(key="word",value="score",
                                                         conn=crawler.conn_tmp)
            crawler.tables["page_scores_"+str(i)] = dict(key="page",value="score",
                                                         conn=crawler.conn_tmp)
        crawler.tables.create_all()

        logging.info("consolidating")
        scores = {}
        for table_name,table in crawler.tables.items():
            if not table_name.startswith("word_scores_"):
                continue            
            _scores = table.select_all()
            for word,score in _scores.items():
                if word not in scores:
                    scores[word] = []
                scores[word].append(score)

        pages = {}
        for table_name,table in crawler.tables.items():
            if not table_name.startswith("page_scores_"):
                continue            
            _scores = table.select_all()
            for page,score in _scores.items():
                if page not in pages:
                    pages[page] = score

        logging.info("aggregating")
        for word,score_list in scores.items():
            if min(score_list) == crawler.epsilon:
                scores[word] = crawler.epsilon
            else:
                scores[word] = np.mean(score_list)
                
        logging.info("committing")
        crawler.tables["word_scores"].insert(scores)
        crawler.tables["page_scores"].insert(pages)

    #
    engine = create_engine(engine_url+"_tmp?charset=utf8")
    conn = engine.connect()
    for i in range(num_total_jobs):
        conn.execute("DROP TABLE IF EXISTS page_scores_"+str(i)+";")
        conn.execute("DROP TABLE IF EXISTS word_scores_"+str(i)+";")
    conn.close()        

''''''
def distributed_run(num_total_jobs,max_threads,**kwargs):

    logging.basicConfig(stream=sys.stderr,level=kwargs["logging_level"])
    threads = []
    for i in range(num_total_jobs):
        kwargs["label"] = str(i)
        kwargs["log_file"] = "logs/log"+str(i)+".out"
        
        if len(threads) == max_threads:
            logging.info("waiting for previous thread")
            threads[0].join()
            threads.pop(0)

        logging.info("launching thread %s",(i))
        t = threading.Thread(target=run,kwargs=kwargs)
        t.start()
        time.sleep(5)
        threads.append(t)

    logging.info("finishing threads")
    for t in threads:
        t.join()
        
    reduce_scores(num_total_jobs,kwargs["engine_url"])
    
#________________________
if __name__ == "__main__":

    with open('db.config') as f:
        engine_url = f.read()
    
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

    distributed_run(num_total_jobs=10,  # 100
                    max_threads=4,     # 4
                    max_depth=4,       # 4
                    page_scorer=g.get_score,
                    threshold=1.5,     # ?
                    engine_url=engine_url,
                    logging_level=logging.DEBUG)
    
    # run(logging_level=logging.DEBUG,
    #     top_url="http://www.manchester.ac.uk/",
    #     #top_url="https://www.masdar.ac.ae/",
    #     page_scorer=g.get_score,
    #     max_depth=4,
    #     threshold=2.1,
    #     label="test",
    #     engine_url=engine_url)

