import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from lda import LDA

print("Reading data...")
env = pd.read_csv('../Data/Environmental Discourse/env.csv', index_col=0).sample(1500, random_state=3291995)

def learn_topics(texts, topicnum):

    # Get vocabulary and word counts.  Use the top 10,000 most frequent
    # lowercase unigrams with at least 3 alphabetical, non-numeric characters,
    # punctuation treated as separators.
    print("Vectorizing...")
    CVzer = CountVectorizer(max_features=5000,
                            lowercase=True)
    doc_vcnts = CVzer.fit_transform(texts)
    vocabulary = CVzer.get_feature_names()

    # Learn topics.  Refresh conrols print frequency.
    print("LDA")
    lda_model = LDA(topicnum, n_iter=500, refresh=50) 
    doc_topic = lda_model.fit_transform(doc_vcnts)
    topic_word = lda_model.topic_word_

    return doc_topic, topic_word, vocabulary

doc_topic, topic_word, vocabulary = learn_topics(env.text, 100)

print(type(doc_topic))
print(doc_topic[0,:])
