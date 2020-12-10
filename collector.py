import os
import tweepy as tw
import pandas as pd
import sys
import time
from api_conf import *

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

lang_choice = input("Type a ISO language code. Example; 'en', 'es', 'tr'\nYour choice : ")

subject = input("Type the word/phrase you expect to see : ")
max_limit = int(input("Type the amount of tweets you need : "))
search_words = subject + " -filter:retweets"
tweets_list = []

tweets = tw.Cursor(api.search, q=search_words, tweet_mode="extended", lang=lang_choice).items()

print("Collecting Twitter data including '{}'".format(subject))

while True:
    
    try:
        tweet = tweets.next()
        tweets_list.append(tweet)
        if len(tweets_list)%50 == 0:
            print(len(tweets_list))
        if len(tweets_list) == max_limit:
            break
    except tw.TweepError:
        print("You have to wait 15 minutes!")
        for remaining in range(15, 0, -1):
            sys.stdout.write("\r")
            sys.stdout.write("{:2d} minutes remaining...".format(remaining))
            time.sleep(60) 
            sys.stdout.flush()
        ###time.sleep(60 * 15)     
        continue
    
    except StopIteration:
        break
    
print("Done! Total tweets count : {}".format(len(tweets_list)))

if len(tweets_list) < max_limit:
    print("Total amount of collected tweets are less than you expected!")

df = pd.DataFrame()
df["text"] = [tweet.full_text for tweet in tweets_list]
df["author"] = [tweet.user.screen_name for tweet in tweets_list]
df["fav_count"] = [tweet.favorite_count for tweet in tweets_list]

file_name = subject.strip()
file_name = file_name.replace(" ", "_")
df.to_excel(file_name+".xlsx")




