import pandas as pd
from pmaw import PushshiftAPI
api = PushshiftAPI()

r_env = api.search_comments(subreddit="environment", limit=50000)
r_cc = api.search_comments(subreddit="climatechange", limit=50000)
r_sus = api.search_comments(subreddit="sustainability", limit=50000)
r_eco = api.search_comments(subreddit="ecology", limit=50000)

r_env_df = pd.DataFrame(r_env)
print(r_env_df.shape)

r_cc_df = pd.DataFrame(r_cc)
print(r_cc_df.shape)

r_sus_df = pd.DataFrame(r_sus)
print(r_sus_df.shape)

r_eco_df = pd.DataFrame(r_eco)
print(r_eco_df.shape)

r_env_df.to_pickle('../data/r_env.pkl')
r_sus_df.to_pickle('../data/r_sus.pkl')
r_cc_df.to_pickle('../data/r_cc.pkl')
r_eco_df.to_pickle('../data/r_eco.pkl')