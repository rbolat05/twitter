'''


Katkılarından dolayı Mehmet Fatih Akça'ya teşekkürler.



'''


import pandas as pd
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

df = pd.read_excel("ist_soz_raw.xlsx", index_col = False)

def remove_qmark(df, column):
	#Tırnak işaretini siler
	df[column] = df[column].apply(lambda x: x.replace("'",""))		

def drop_duplicates(df):
	#text ve author sütunları aynı olan satırları siler, son satırı tutar
    df.drop_duplicates(subset = ["text","author"], keep = "last", inplace = True)

def normalise(df, column):
	#tüm harfleri küçük harfe dönüştürür
    lower = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRŞSTUÜVYZ", "abcçdefgğhıijklmnoöprşstuüvyz")
    df[column] = df[column].apply(lambda x: x.translate(lower))
    
def remove_digit(df, column):
	#sayıları siler
	df[column] = df[column].str.replace("\d","")

def remove_space(df, column):
	#boşlukları siler
    df[column] = df[column].apply(lambda x: x.replace("\n"," "))
    df[column] = df[column].apply(lambda x: x.replace("\t"," "))
    
def remove_hashtag(df, column):
	#hashag'leri siler
    df[column] =  df[column].apply(lambda x: re.sub(r"#\S+","", x))

def remove_url(df, column):
	#url'leri siler
    df[column] = df[column].apply(lambda x: re.sub(r"http\S+","", x))

def remove_mention(df, column):
	#bahsedilmeleri siler
    df[column] = df[column].apply(lambda x: re.sub(r"@\S+","", x))

def remove_emoji(df, column): 
	#tüm emoji ve benzerlerini siler
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
         
    df[column] = df[column].apply(lambda x: emoji_pattern.sub(r"", x))

def remove_punct(df, column):
	#noktalama işaretlerini siler
    df[column] = df[column].str.replace('[^\w\s]',' ')

def remove_stopwords(df, column):
	#nltk kütüphanesindeki Türkçe duraksama kelimelerini siler
    stop_words = stopwords.words("turkish")
    df[column] = df[column].apply(lambda x: " ".join(word for word in x.split() if not word in stop_words))
    
def drop_row(df, column):
	#boş kalan satırları siler
	df[column] = df[column].apply(lambda x: x.strip())
	df.drop(df[df.text == ""].index, inplace=True)
		
def remove_keyword(df, column):
	#aranılan kelimeyi/kelime grubunu siler
	df[column] =  df[column].apply(lambda x: re.sub(r"istanbul sözleşmesi\S+","", x))	
	df[column] =  df[column].apply(lambda x: x.replace("istanbul sözleşmesi",""))
			
remove_keyword(df, "text")
#df = df.drop("Unnamed: 0", axis = 1)
normalise(df, "text")
remove_digit(df, "text")
remove_qmark(df, "text")
remove_space(df, "text")
remove_hashtag(df, "text")
remove_url(df, "text")
remove_mention(df, "text")
remove_punct(df, "text")
remove_stopwords(df, "text")
remove_emoji(df, "text")
drop_duplicates(df)
drop_row(df, "text")

df.to_excel("istanbul_sozlesmesi_prep.xlsx", index = False) 

