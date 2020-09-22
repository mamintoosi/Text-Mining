
import tweepy
import csv
consumer_key = '####' # کلید consumer شخصی شما
consumer_secret = '###' # شما consumer_secret
access_token = '###' # کد دسترسی شما
access_token_secret = '###'   # access_token_secret شما

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

f=open("tex.txt",'a')

for tweet in tweepy.Cursor(api.search,q="iran",count=10,text_mode='extended',
                           lang="en",
                           since="2018-01-01").items():
    print (tweet.created_at, tweet.text)
    f.write(tweet.text + '\n\n')
f.close()

# برای دریافت کد دسترسی خود به بخش Developers توئیتر مراجع کنید
