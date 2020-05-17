import pandas as pd
import spacy
import matplotlib.pyplot as plt

from spacy.lang.en import English
from dateutil.parser import parse

from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

nlp = English()
nlp = spacy.load('en_core_web_sm')

mw = set(open('merriam_webster.txt', 'r', encoding='utf-8').read().splitlines())
chv = set(pd.read_csv('CHV_concepts_terms_flatfile_20110204.tsv', sep='\t', dtype='unicode').iloc[: , 1])
df = pd.read_csv('covid19.tsv', sep='\t')

def is_date(string, fuzzy=False):
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def data_trend():
    dates = {}
    for row in df.itertuples():
        doc = nlp(row.tweet)
        dt = str(row.collected_at)
        if not is_date(dt): continue
        count = 0
        for token in doc:
            if token.lower_ == 'covid19' or token.lower_ == 'covid' or token.lower_ == 'covid-19' or token.lower_ == '#covid19':
                count += 1
        if dt in dates.keys():
            dates[dt] += count
        else:
            dates[dt] = count

    plt.bar(range(len(dates)), list(dates.values()), align='center')
    plt.xticks(range(len(dates)), list(dates.keys()), rotation=90)
    plt.show()

@labeling_function()
def is_medical_mw(x):
    tweet = nlp(x.tweet)
    tokens = set([t.text for t in tweet])
    hit = mw & tokens
    return MEDICAL if len(hit) > 0 else NONMEDICAL

@labeling_function()
def is_medical_chv(x):
    tweet = nlp(x.tweet)
    tokens = set([t.text for t in tweet])
    hit = chv & tokens
    return MEDICAL if len(hit) > 0 else NONMEDICAL

NONMEDICAL = 0
MEDICAL = 1

def label_medical():
    lfs = [is_medical_mw, is_medical_chv]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    lfa = LFAnalysis(L=L_train, lfs=lfs)
    print (lfa.lf_summary())

label_medical()
data_trend()