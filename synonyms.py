"""
Requirements: Python 3.6+, NLTK
Setup: Install NLTK and download the WordNet corpus

pip install nltk
python -m nltk.downloader wordnet
"""

from nltk.corpus import wordnet

class SortResults:
    Semantic = 1,
    AlphaNum = 2

def get_synonyms(word:str, sort:SortResults=SortResults.Semantic, num:int=-1) -> list:
    syns = list()
    for synset in wordnet.synsets(word):
	    for lemma in synset.lemmas(): # Lemmas for different contexts of the word (e.g. verb, noun, etc.)
		    syns.append(lemma.name()) # Combine all the synonyms for all lemmas
    
    if num == -1: num = len(syns)
    if sort == SortResults.AlphaNum: syns = sorted(syns)
    return syns[:num]