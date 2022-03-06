# IMPORT PACKAGES
import pandas as pd
import nltk
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
import ast
import dask.dataframe as dd
from dask.multiprocessing import get
import spacy
from gensim import corpora, models
from gensim.utils import effective_n_jobs
try:
    nlp = spacy.load("en")
except OSError:
    nlp = spacy.load("en_core_web_sm")

    
# CREATE FUNCTIONS
def word_tokenize(word_list, model=nlp, MAX_LEN=1500000):
    
    tokenized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list]) 
    # since we're only tokenizing, I remove RAM intensive operations and increase max text size

    model.max_length = MAX_LEN
    doc = model(word_list, disable=["parser", "tagger", "ner", "lemmatizer"])
    
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized


def normalizeTokens(word_list, extra_stop=[], model=nlp, lemma=True, MAX_LEN=1500000):
    #We can use a generator here as we just need to iterate over it
    normalized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list]) 

    # since we're only normalizing, I remove RAM intensive operations and increase max text size

    model.max_length = MAX_LEN
    doc = model(word_list.lower(), disable=["parser", "tagger", "ner"])

    if len(extra_stop) > 0:
        for stopword in extra_stop:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

    # we check if we want lemmas or not earlier to avoid checking every time we loop
    if lemma:
        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
                normalized.append(str(w.lemma_))
    else:
        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
                normalized.append(str(w.text.strip()))

    return normalized
    

def ngram_tagger(tokens):
    n = len(tokens)
    i = 0
    tokens_q = []
    tokens_qt = []
    tokens_qtb = []
    
    # quadgrams
    while i < n:
        words = '_'.join(tokens[i:i+4])
        if words in quadgrams:
            tokens_q.append(words)
            i += 4
        else:
            tokens_q.append(tokens[i])
            i += 1
    
    # trigrams
    n = len(tokens_q)
    i = 0
    while i < n:
        words = '_'.join(tokens_q[i:i+3])
        if words in trigrams:
            tokens_qt.append(words)
            i += 3
        else:
            tokens_qt.append(tokens_q[i])
            i += 1
    
    # bigrams
    n = len(tokens_qt)
    i = 0
    while i < n:
        words = '_'.join(tokens_qt[i:i+2])
        if words in bigrams:
            tokens_qtb.append(words)
            i += 2
        else:
            tokens_qtb.append(tokens_qt[i])
            i += 1
    
    return tokens_qtb
    
print('Reading in data, splitting...')
env = pd.read_csv('../Data/Environmental-Discourse/env.csv', index_col=0)
env = env.sample(3000, random_state=4151995)
env, validation = train_test_split(env, test_size=0.5, random_state=3291995)

print('Saving split pickles...')
pd.to_pickle('../Data/Environmental-Discourse/env_0.pkl')
pd.to_pickle('../Data/Environmental-Discourse/env_validation.pkl')

print('Creating n-gram lists...')
quadgrams = [('intergovernmental', 'panel', 'climate', 'change'),
             ('natural', 'resources', 'defense', 'council'),
             ('coal', 'fired', 'power', 'plants'),
             ('national', 'oceanic', 'atmospheric', 'administration')]

tr = pd.read_csv('../Data/Environmental Discourse/trigrams.csv', converters={'Unnamed: 0': ast.literal_eval})
tr.columns = ['trigram', 'freq', 'tag']
trigrams = [t for t in tr[tr.tag == 1].trigram]

b = pd.read_csv('../Data/Environmental Discourse/bigrams.csv', converters={'Unnamed: 0': ast.literal_eval})
b.columns = ['bigram', 'freq', 'tag']
bigrams = [t for t in b[b.tag == 1].bigram]

quadgrams = ['_'.join(t) for t in quadgrams]
trigrams = ['_'.join(t) for t in trigrams]
bigrams = ['_'.join(t) for t in bigrams]

print('Tokeninzing...')
d_env = dd.from_pandas(env, npartitions=effective_n_jobs(-1))
d_env['tokens'] = d_env.text.map(lambda x: ngram_tagger(normalizeTokens(word_tokenize(x))), meta=('x', str))
env_tok = d_env.compute()

print('Creating dictionary, bow corpus, tfidf...')
dictionary = corpora.Dictionary([i for i in env_tok.tokens])
bow_corpus = [dictionary.doc2bow(text) for text in env_tok.tokens]
tfidf = models.TfidfModel(bow_corpus)

mask_07 = env_tok.year == 2007
mask_13 = env_tok.year == 2013
mask_19 = env_tok.year == 2019

bow_corpus_07 = [doc for i, doc in enumerate(bow_corpus) if mask_07.iloc[i]]
bow_corpus_13 = [doc for i, doc in enumerate(bow_corpus) if mask_13.iloc[i]]
bow_corpus_19 = [doc for i, doc in enumerate(bow_corpus) if mask_19.iloc[i]]

print('Running models...')
models_07, coherence_07 = compute_coherence_values(dictionary=dictionary, 
                                                   corpus=tfidf[bow_corpus_07], 
                                                   texts=env_tok.tokens,
                                                   start=4, limit=11, step=2)

models_13, coherence_13 = compute_coherence_values(dictionary=dictionary, 
                                                   corpus=tfidf[bow_corpus_13], 
                                                   texts=env_tok.tokens,
                                                   start=4, limit=11, step=2)

models_19, coherence_19 = compute_coherence_values(dictionary=dictionary, 
                                                   corpus=tfidf[bow_corpus_19], 
                                                   texts=env_tok.tokens,
                                                   start=4, limit=11, step=2)


print('Saving models, top words, and coherence scores...')
all_models = models_07 + models_13 + models_19
names = ['tm_{}_{}'.format(yr, tp) for yr in ['07', '13', '19'] for tp in ['04', '06', '08', '10']]

for model, filename in zip(all_models, names):
    topicsDict = {}
    for topicNum in range(model.num_topics):
        topicWords = [w for w, p in model.show_topic(topicNum)]
        topicsDict['Topic_{}'.format(topicNum)] = topicWords

    wordRanksDF = pd.DataFrame(topicsDict)
    wordRanksDF.to_csv('../Data/Environmental-Discourse/Single-Year-TMs/Top-Words/' + filename + '.pkl')
    model.save('../Data/Environmental-Discourse/Single-Year-TMs/Models/' + filename)

coherence = coherence_07 + coherence_13 + coherence_19
coh = pd.DataFrame({'model': names, 'coherence': coherence})
coh.to_pickle('../Data/Environmental-Discourse/Single-Year-TMS/coherence_scores.pkl')

print('Saving dictionary, bow corpus, tfidf...')
dictionary.save('../Data/Environmental-Discourse/Single-Year-TMs/dictionary')
gensim.corpora.MmCorpus.serialize('../Data/Environmental-Discourse/Single-Year-TMs/bow_corpus.mm', bow_corpus)
tfidf.save('../Data/Environmental-Discourse/Single-Year-TMs/tfidf')
    
