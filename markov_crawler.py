'''
TODO: Usage instructions
TODO: Setup instructions, including required table names, such as top_url etc
'''

# Fix normalised score (currently forced select_url to manchester)
# Work out how to drop without hacking
# Need to do proper analysis of course pages vs non course pages to optimise scorer
# Add extra universities (get these from someone)

from bs4 import BeautifulSoup
import logging
import numpy as np
from page_scorer import PageScorer
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

class KeyValueTables(dict):
    '''A collection of _KeyValueTable objects'''    
    def __init__(self):
        self.metadatas = []

    '''Override setitem to generate the _KeyValueTable'''
    def __setitem__(self,table_name,kwargs):
        # Append metadata to list of metadatas and kwargs
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
            if not table.conn.closed:
                table.conn.close()

class _KeyValueTable:
    '''
    A wrapper around sqlalchemy Table to generate a 2-column table 
    of the form "key : score", intended for recording scores of
    keywords for the markov chain procedure. Creates the table if none 
    exists. Also features a handy select_all and insert function.
    '''
    def __init__(self,table_name,conn,metadata,key,value):        
        self.table = Table(table_name,metadata,
                           Column(key,Text,nullable=False),
                           Column(value,Float,nullable=False),)
        self.key = key
        self.value = value
        self.table_name = table_name
        self.conn = conn

    '''Get all scores for this table in a dictionary of the form {"word":"score"}'''        
    def select_all(self):
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
                 label="test",is_dummy=False,engine_url=None):
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
              
        # Assign a page scoring function, that take one argument (a string)
        # which it analyses and returns a score
        self.page_scorer = page_scorer 

        # Create tables if they don't already exist
        self.conn = self.connect(engine_url)
        self.conn_tmp = self.connect(engine_url+"_tmp")

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
        logging.debug("Got %d word scores from the database.",(len(self.scores)))
        logging.debug("Got %d page scores from the database.",(len(self.pages)))

        # If a top URL has been specified, otherwise select a random top URL
        # from a table called "top_urls"
        if top_url is not None:
            self.top_url = top_url
        else:
            # SELECT url FROM top_urls;
            results = self.conn.execute(sql_select([sql_text("url")],
                                                   from_obj=sql_text("top_urls")))
            self.top_url = random.choice([_r["url"] for _r in results.fetchall()])
                    
        # Record the top URL stub, in order to identify external sites
        parse_result = urlparse(self.top_url)
        self.top_url_stub = ".".join(parse_result.netloc.split(".")[-3:])

    '''
    Convenience method to connect to a DB via a sql_alchemy engine,
    whilst also keeping track of open transactions.
    '''
    def connect(self,engine_url):
        engine = create_engine(engine_url)
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
            logging.debug("Breaking out")
            return False    
        # 3) Set the url if None
        if url is None:
            url = self.top_url        
        # 4) Set the current URL
        if not url.startswith("http"):
            url = urljoin(self.current_url,url)
        self.current_url = url.rstrip("/")
        logging.debug("Trying %s",(self.current_url))

        # Get this page's HTML
        try:
            r = requests.get(self.current_url)
            r.raise_for_status()
        except (requests.exceptions.HTTPError,
                requests.exceptions.InvalidSchema) as err:
            logging.warning("Got error %s for URL %s",(str(type(err)),
                                                       self.current_url))
            return False
        soup = BeautifulSoup(r.text,"lxml")

        # Accrue the page score, unless previously done
        # (otherwise the page score can be acquired through self.pages[url]
        page_previously_done = (url in self.pages)

        # Iterate through links on the page to calculate the score
        page_score = 0
        anchors = soup.find_all("a")
        for a in anchors:
            # If not a link, skip
            if "href" not in a.attrs:
                continue
            # Ignore '#' fragments or internal links
            link_url = urldefrag(a["href"])[0].rstrip("/")
            if link_url == self.current_url or link_url.strip() == '':
                continue
            # Ignore external links
            parse_result = urlparse(link_url)
            if parse_result.netloc != '':
                if self.top_url_stub not in parse_result.netloc:
                    continue
            # Calculate the 'Markov score' (a relative weight used for 
            # Monte Carlo selection) and page score for this URL and link text
            link_text = a.text
            mkv_score = self.calculate_markov_score(link_url+" "+link_text)
            if mkv_score is not None:
                self.url_scores[link_url] = mkv_score
            # Don't reprocess previously visited pages
            if not page_previously_done:
                # Normalise the score to the number of links per page
                page_score += self.page_scorer(link_text) / np.sqrt(len(anchors))

        # Record the non-normalised score
        if not page_previously_done:
            self.pages[url] = page_score
        else:
            page_score = self.pages[url]
        
        # Normalise the score to the home page
        normalised_score = page_score/self.pages[self.top_url]
        logging.debug("\t\tGot a normalised page score of %0.2f.",(normalised_score))
        if normalised_score >= self.threshold:
            logging.info("\t\t%s is over threshold.",(self.current_url))
            # Reduce the score of successful words
            for word in self.active_words:
                self.scores[word] = self.epsilon
            return True
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
            return None
        # Sum the score for each words
        for word in words:
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
        #return "http://www.manchester.ac.uk/study/undergraduate/courses/2018/"
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
            if not self.is_dummy:
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
    logging.basicConfig(stream=sys.stderr, level=logging_level)

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
    engine = create_engine(engine_url)
    conn = engine.connect()
    conn.execute("DROP TABLE IF EXISTS page_scores;")
    conn.execute("DROP TABLE IF EXISTS word_scores;")
    conn.close()

    # Reduce the results
    logging.info("Going to generate the new tables")
    with MarkovCrawler(is_dummy=True) as crawler:        
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
    engine = create_engine(engine_url+"_tmp")
    conn = engine.connect()
    for i in range(num_total_jobs):
        conn.execute("DROP TABLE IF EXISTS page_scores_"+str(i)+";")
        conn.execute("DROP TABLE IF EXISTS word_scores_"+str(i)+";")
    conn.close()        

''''''
def distributed_run(num_total_jobs,max_threads,**kwargs):
    threads = []
    for i in range(num_total_jobs):
        kwargs["label"] = str(i)
        
        if len(threads) == max_threads:
            logging.info("waiting for previous thread")
            threads[0].join()
            threads.pop(0)
        logging.info("launching thread",i)
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
    
    # distributed_run(num_total_jobs=4,max_threads=4,
    #                 top_url="http://www.manchester.ac.uk/",max_depth=4,
    #                 page_scorer=PageScorer().get_score,threshold=1.3,
    #                 label="test",engine_url=engine_url)

    run(logging_level=logging.DEBUG,
        top_url="http://www.manchester.ac.uk/",
        page_scorer=PageScorer().get_score,
        max_depth=4,
        threshold=1.3,
        label="test",
        engine_url=engine_url)
