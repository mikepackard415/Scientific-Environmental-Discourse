# Import packages
import pandas as pd
from gensim import corpora, models
import ast

# Set path
path = 'Environmental-Discourse'

# Read in data
env = pd.read_csv('../Data/'+path+'/env_processed_tokens.csv', 
                  index_col=0, 
                  converters={'tokens': ast.literal_eval})
env['date'] = pd.to_datetime(env.date)
env['year'] = env.date.dt.year

dictionary = corpora.Dictionary.load('../Data/'+path+'/Full-TMs/dictionary')
corpus = corpora.MmCorpus('../Data/'+path+'/Single-Year-TMs/bow_corpus.mm')

# Set up model
docs_per_time_slice = list(env.groupby('year').agg({'year':'count'}).year)

# Fit model
ldaseq = models.ldaseqmodel.LdaSeqModel(corpus=corpus,
                                        id2word=dictionary, 
                                        time_slice=docs_per_time_slice, 
                                        num_topics=9)

model.save('../Data/' + path + '/Full-TMs/Models/dtm_09')
