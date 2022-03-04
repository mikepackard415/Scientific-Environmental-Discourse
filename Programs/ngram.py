import pandas as pd

env = pd.read_csv('../Data/Environmental Discourse/env.csv', index_col=0)

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

# Apply tokenization and normalization functions
env['tokenized_text'] = env['text'].apply(lambda x: word_tokenize(x))
env['normalized_tokens'] = env['tokenized_text'].apply(lambda x: normalizeTokens(x, lemma=False))
env.to_pickle('../Data/env.pkl')

env['bigrams'] = env['normalized_tokens'].apply(lambda x: [i for i in ngrams(x, 2)])
bigrams = pd.Series(env['bigrams'].sum()).value_counts().head(100)
bigram_df = pd.DataFrame({'bigram': bigrams})
bigram_df.to_csv('../Data/Environmental Discourse/bigrams.csv')

env['trigrams'] = env['normalized_tokens'].apply(lambda x: [i for i in ngrams(x, 3)])
trigrams = pd.Series(env['trigrams'].sum()).value_counts().head(100)
trigram_df = pd.DataFrame({'trigram': trigrams})
trigram_df.to_csv('../Data/Environmental Discourse/trigrams.csv')