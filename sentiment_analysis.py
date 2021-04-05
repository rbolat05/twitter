from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

#Turkish Bert ile duygu analizi
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer = tokenizer, model = model)

df = pd.read_excel("result.xlsx")  #model.py ile oluşturulan excel dosyasını çağırır

sent_label = []

for i in df["text"]:
    p=sa(i)
    if p[0]["label"] == "positive": #label olarak pozitif dönerse P, negatif dönerse N alır ve listeye ekler
    	sent_label.append("P") 
    else:
    	sent_label.append("N")
    
df["sent_label"] = sent_label

df.to_excel("result_sa.xlsx", index = False)
