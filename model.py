#gerekli kütüphanelerin hazırlanması
import pandas as pd 
import numpy as np
from snowballstemmer import TurkishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

###################################################################

turkStem= TurkishStemmer()

df = pd.read_excel("istanbul_sozlesmesi_prep.xlsx")
df.drop(["fav_count"], axis = 1, inplace = True)

#Bag of Words yöntemi ile sayısallaştırma yapar
cv = CountVectorizer()  
word_vector = cv.fit_transform(df["text"].apply(lambda x: " ".join([turkStem.stemWord(i) for i in x.split()])))

#etiketli olan 600 verilik kısmı eğitim ve test seti olarak kullanmak üzere X ve y değişkenlerine atar
X = word_vector[:600, : ]
y = df["category"].head(600)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

model_score = []
model_probas = []
###################################################################

#DecisionTreeClassifier
DecTree = DecisionTreeClassifier()  #model için nesne oluşturur
DecTree.fit(X_train, y_train)  #eğitim verilerini alır
DecTree_prediction = DecTree.predict(X_test)  #test verilerini tahmin eder
model_score.append(accuracy_score(y_test, DecTree_prediction))  #başarı oranını model_score listesine ekler
DecTree_prediction = DecTree.predict(word_vector)  #verinin tamamını tahmin eder ve değişkene atar

try:
	DecTree_prob = DecTree.predict_proba(word_vector) #modelden olasılıkları döndürmeyi dener, her modelde işe yaramaz
	model_probas.append(DecTree_prob)  #dönen olasılık matrisini model_probas listesine ekler
except:
	model_probas.append("n/a")  #olasılık dönmeyi desteklemiyorsa listeye n/a yazar

#RandomForestClassifier
RandFor = RandomForestClassifier()
RandFor.fit(X_train,y_train)
RandFor_prediction = RandFor.predict(X_test)
model_score.append(accuracy_score(y_test, RandFor_prediction))
RandFor_prediction=RandFor.predict(word_vector)

try:
	RandFor_prob = RandFor.predict_proba(word_vector)
	model_probas.append(RandFor_prob)
except:
	model_probas.append("n/a")
	
#MultiNominalNaiveBayes
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
MNB_prediction = MNB.predict(X_test)
model_score.append(accuracy_score(y_test, MNB_prediction))
MNB_prediction=MNB.predict(word_vector)

try:
	MNB_prob = MNB.predict_proba(word_vector)
	model_probas.append(MNB_prob)
except:
	model_probas.append("n/a")

#SupportVectorMachine
from sklearn import svm
SVMac=svm.SVC(probability = True)
SVMac.fit(X_train, y_train)
SVMac_prediction = SVMac.predict(X_test)
model_score.append(accuracy_score(y_test, SVMac_prediction))
SVMac_prediction=SVMac.predict(word_vector)
try:
	SVMac_prob = SVMac.predict_proba(word_vector)
	model_probas.append(SVMac_prob)
except:
	model_probas.append("n/a")

#LogisticRegression
LogicReg = LogisticRegression()
LogicReg.fit(X_train, y_train)
LogicReg_prediction = LogicReg.predict(X_test)
model_score.append(accuracy_score(y_test, LogicReg_prediction))
LogicReg_prediction = LogicReg.predict(word_vector)
try:
	LogicReg_prob = LogicReg.predict_proba(word_vector)
	model_probas.append(LogicReg_prob)
except:
	model_probas.append("n/a")

#XGBoostClassifier
xgboost = XGBClassifier()
xgboost.fit(X_train,y_train)
xgboost_prediction = xgboost.predict(X_test)
model_score.append(accuracy_score(y_test, xgboost_prediction))
xgboost_prediction = xgboost.predict(word_vector)
try:
	xgboost_prob = xgboost.predict_proba(word_vector)
	model_probas.append(xgboost_prob)
except:
	model_probas.append("n/a")

#Perceptron
perc = Perceptron()
perc.fit(X_train, y_train)
perc_prediction = perc.predict(X_test)	
model_score.append(accuracy_score(y_test, perc_prediction))
perc_prediction = perc.predict(word_vector)	
try:
	perc_prob = perc.predict_proba(word_vector)
	model_probas.append(perc_prob)
except:
	model_probas.append("n/a")

#K-NearestNeighbor
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
model_score.append(accuracy_score(y_test, knn_prediction))
knn_prediction = knn.predict(word_vector)
try:
	knn_prob = knn.predict_proba(word_vector)
	model_probas.append(knn_prob)
except:
	model_probas.append("n/a")

#yukarıdaki sırayla aynı olacak şekilde, modellerin isimlerini ve skorlarını tutan bir dataframe oluşturur
model_name = ["DEC-TREE", "RAND-FOR", "MULT-NB","SVM", "LOG-REG", "XGBOOST", "PERC", "KNN"]
model_scores = pd.DataFrame()
model_scores["ALGORITHM NAME"] = model_name
model_scores["ALGORITHM SCORE"] = model_score
model_scores.to_excel("model_scores.xlsx", index = False)


pred_list = [DecTree_prediction, RandFor_prediction, MNB_prediction, SVMac_prediction, LogicReg_prediction, xgboost_prediction, perc_prediction, knn_prediction]  #tüm modellere ait tahmin değişkenlerini listeye ekler
best_model_index = model_score.index(max(model_score))  #daha önce oluşturulan model_score listesindeki en yüksek değerin indisini bulur

best_model = pred_list[best_model_index]  #bu değerin hangi modele ait olduğunu bulur ve best_model değişkenine atar
df["category"][600:] = best_model[600:]  #kategori sütununun elle etiketlenmeyen kısmını en yüksek skorlu model ile doldurur
title_predict = str(model_name[best_model_index]).lower()+"_predict"  #dataframe'de sütun adı olarak kullanılmak üzere başlık oluşturur

best_probas = model_probas[best_model_index]  #yukarıdaki index yardımıyla en iyi modelin olasılık matrisini bulur
title_probas = str(model_name[best_model_index]).lower()+"_probas"  #olasılık sütununa yazılacak başlığı hazırlar
max_probas = [max(x) for x in best_probas]  #her satır için tahmin edilen değere ait olasılığı bulur

#######################################################################


#kullanılan tüm modellerin tüm tahminlerini ve biraz önce tamamlanmış olan kategori sütununu alarak dataframe oluşturur.
mlearning_set = pd.DataFrame()
mlearning_set["category"] = df["category"]

mlearning_set["DEC-TREE"] = DecTree_prediction
mlearning_set["RAND-FOR"] = RandFor_prediction
mlearning_set["MULT-NB"] = MNB_prediction
mlearning_set["SVM"] = SVMac_prediction
mlearning_set["LOG-REG"] = LogicReg_prediction
mlearning_set["XGBOOST"] = xgboost_prediction
mlearning_set["PERC"] = perc_prediction
mlearning_set["KNN"] = knn_prediction

mlearning_set.to_excel("machine_learning_set.xlsx", index = False)

#########################################################

#olasılık dönmeyi destekleyen modelleri kullanarak bütün veriye ait olasılıkları birleştirir
modelprobtable = [[] for x in range(len(df))]
for i in range(len(df)):
	for j in range(6):
		#modelproblist[i].append(DecTree_prob[i][j]) 
		modelprobtable[i].append(RandFor_prob[i][j]) 
		modelprobtable[i].append(MNB_prob[i][j]) 
		modelprobtable[i].append(SVMac_prob[i][j]) 
		modelprobtable[i].append(LogicReg_prob[i][j]) 
		modelprobtable[i].append(xgboost_prob[i][j]) 
		#modelproblist[i].append(knn_prob[i][j])
						
model_prob_file = pd.DataFrame(data = modelprobtable)
model_prob_file.to_excel("model_prob_file.xlsx", index = False)
#########################################################

df[title_predict] = best_model  #en yüksek skorlu modelin tahminlerini sütuna atar
df[title_probas] = max_probas  #en yüksek skorlu modelin olasılıklarını sütuna atar.
df.to_excel("result.xlsx", index = False)
