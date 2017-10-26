import nltk
import numpy as np
from collections import Counter
import re 
import pandas as pd
#nltk.data.path = ["D:/Grenfell_Anika/Joel/nltk_data/"]
from nltk.stem.wordnet import WordNetLemmatizer

#___________________ 
class Compounder:
    '''Extracts commonly associated words by extracting recursively smaller common n-grams,
    where the "commoness" of an n-gram is defined by it's frequency with respect to the mean
    number of occurences, and the standard deviation of occurences. The <threshold> parameter
    controls the minimum number of standard deviations required for an n-gram to pass the
    selection, whilst the <max_context> defines the maximum size of the n-grams which are
    returned to the user.
    '''
    #___________________ 
    def __init__(self,max_context=3,threshold=6,reverse=False):
        self.start_context = 10
        self.max_context = max_context
        self.stops = nltk.corpus.stopwords.words('english')
        self.threshold = threshold
        self.data = []
        self.compounds = []
        self.reverse = reverse
        self.wnl = WordNetLemmatizer()
    #___________________
    def is_noun(self,word):
        return nltk.pos_tag([word])[0][1][:2] == "NN"
    #___________________ 
    def _extract_subsentences(self,sentences):
        '''Split up sentences sub-sentences, based on any non-alphanum'''
        _sentences = []
        for _sentence in sentences:
            if pd.isnull(_sentence):
                continue
            _sub_sentences = [x.rstrip(" ").lstrip(" ").lower() 
                              for x in re.split('[^a-zA-Z\d\s]',_sentence)]
            if _sub_sentences != []:
                _sentences += _sub_sentences
        return _sentences
    #___________________ 
    def process_sentences(self,sentences,stops=[],synonyms={},stemmer=None):
        '''Iteratively extract compounds from sentences'''
        _sentences = self._extract_subsentences(sentences)
        print("Extracted",len(_sentences),"sentences")
        # Iterate over context range
        the_range = np.arange(self.start_context,0,-1)
        if self.reverse:
            the_range = np.arange(1,self.start_context+1,1)
        for _context in the_range:
            # Remove any previous compounds from the sentence
            for _c in self.compounds:
                _sentences = [_s.replace(" ".join(_c),"") for _s in _sentences
                              if not pd.isnull(_s)]
            # Get compounds for this context
            self.compounds += self._process_sentences(_sentences,_context,
                                                      stops,synonyms,stemmer)
        # Select the compound words within the required context
        self.compounds = [_c for _c in self.compounds if len(_c) <= self.max_context]
    #___________________ 
    def _process_sentences(self,sentences,context,stops,synonyms,stemmer):
        '''Extract compounds from sentences with a given context'''
        compounds = []
        for words in self._preprocess(sentences,stops,synonyms,stemmer):
            # Get all compounds in this sentence
            for _compound in nltk.ngrams(words,context):
                # If this is a 1-gram, ignore stopwords and numbers
                if all(word in self.stops or word.isdigit()
                       for word in _compound):
                        continue
                # Append compound
                compounds.append(_compound)
        
        # Count compounds
        counts = Counter(compounds)
        count_values = [x for x in counts.values()]
        # Calculate the mean and std
        mean = np.mean(count_values)
        std = np.std(count_values)
        
        # Calculate the threshold data for this context 
        # This is purely for analytical purposes
        for i in np.arange(1,10.5,0.5):
            total = sum(1 for _compound,_count in counts.items()
                        if _count > mean + i*std)
            self.data.append(dict(threshold=i,context=context,total=total))
        # Filter out the compounds passing the threshold condition
        for _compound,_count in counts.items():            
            if _count <= mean + self.threshold*std:
                compounds = list(filter((_compound).__ne__, compounds))
        # Return unique compounds
        return set(compounds)
    #___________________
    def _preprocess(self,sentences,stops,synonyms,stemmer):
        _sentences = []
        if stemmer is None:
            # Blank argument for lemmatizer
            stem = lambda x,_ : x
        else:
            stem = stemmer.stem
            
            
        for _sentence in sentences:
            # Split sentences into words
            for k,v in synonyms.items():
                if k in _sentence:
                    _sentence = re.sub(r"\b"+k+r"\b",v,_sentence)
            words = nltk.word_tokenize(_sentence)
            words = [stem(w,self.is_noun(w))
                     for w in words 
                     if w not in stops and not w.isdigit()]
            _sentences.append(words)
        return _sentences
    #___________________ 
    def print_sorted_compounds(self):
        '''Method for printing compounds'''
        c = [(" ".join(c)) for c in self.compounds]
        for _c in sorted(c):
            print(_c)

#___________________
# NOTE, MOVE THIS FUNCTION OUT OF THIS MODULE, MORE RELATED TO THE DATA
# CLEANING PIPELINE
'''From the bottom up, find cells with value <upfill_value>, and iteratively
increase replace with the first cell above which does not have the value
<upfill_value>. For example if 3 consecutive column cells contain the values:

    Hello, World!
    as above
    as above
    
Then setting upfill_value="as above" will result in

    Hello, World!
    Hello, World!
    Hello, World!
'''
def upfill(column,upfill_value):
    upfill_indexes = []
    # Iterate backwards through column values
    for _idx,_value in enumerate(reversed(column.values)):
        # If a non-upfill-value is found, then flush the indexes
        if _value != upfill_value:
            # If there are any upfill indexes to flush
            if upfill_indexes != []:
                # For each index, set the cell value to the non-upfill-value
                for upfill_idx in upfill_indexes:
                    column.values[upfill_idx] = _value
                # Empty the indexes
                upfill_indexes=[]
            # Don't append indexes for non-upfill-values
            continue
        # This is an upfill-value, so record the index
        # Note: reverse iterating so true index = N - idx - 1
        upfill_indexes.append(len(column.values)-_idx-1)
    return column.values

#___________________
# Example of how to run
if __name__ == "__main__":
    columns = ["Building information", "Updates – immediate response",
               "Community Engagement","Remedial action"]
    
    # "Upfill" the data for values of "see above" and "as above"
    # NOTE: this is data cleaning, and probably best done prior to running this
    # module, but I haven't got around to that yet
    df = pd.read_csv("casework-tracker-with-google-places.tsv",sep="\t")
    df[columns] = df[columns].apply(upfill,upfill_value="as above")
    df[columns] = df[columns].apply(upfill,upfill_value="see above")
    
    # Get all sentences, and drop duplicates
    sentences = []
    for col_name in columns:
        sentences += list(df[col_name].drop_duplicates().values)
    
    # Instantiate the compounder
    print("------------\nRunning top-down compounder\n")
    comper = compounder(max_context=8,threshold=6)
    comper.process_sentences(sentences)
    comper.print_sorted_compounds()
    
    # Instantiate a reverse compounder
    print("------------\nRunning bottom-down compounder\n")
    comper_r = compounder(max_context=8,threshold=6,reverse=True)
    comper_r.process_sentences(sentences)
    comper_r.print_sorted_compounds()
