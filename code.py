import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#test_data = input("Write your review: ")


print("Loading data...")


data = pd.read_csv("/home/honza/programování/python/deeplearning/IMDB reviews/files/IMDB Dataset (shorter).csv")
test_data = ["I could almost wish this movie had not been made. Stan Laurel was dying, and it shows in his face, even more angular and gaunt than usual. A poor script, and inept supporting cast."]

labels = data["sentiment"].replace({"positive":1, "negative":0})
data = data.drop(["sentiment"],axis=1)

data = data.append({"review":test_data[0]},ignore_index=True)

data["review"] = [sentance.replace("<br />","") for sentance in test_data[0]]
data["review"] = [sentance.lower() for sentance in data["review"]]
data["review"] = [re.sub(r'[^\w\s]','',i) for i in data["review"]]


print("removing stop words...")

stop_word = set(stopwords.words("english"))

data['review'] = data["review"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_word)]))

data = data.drop([0],axis=1)



print("transforming words...")
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])


data['review'] = data["review"].apply(lemmatize_text)


from keras.preprocessing.text import one_hot

data["review"] = [one_hot(sentence,5000) for sentence in data["review"]]


data_list = []


for listt in data["review"]:
	data_list.append(listt)

data_list = pd.DataFrame(data_list)

data_list = data_list.fillna(0)

test_list = data_list[-1:]

data_list = data_list[:-1]



input_set = np.array(data_list).astype(float)

output_set = np.array(labels).astype(float)

from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding

print("building model...")
#680
model = Sequential([
	Embedding(input_dim=5000,output_dim=60,input_length=925),
	LSTM(256),
	Dropout(0.1)
	LSTM(256),
	Dense(100),
	Dense(1,activation="softmax"),
	])

print(model.summary())

model.compile(optimizer="adam",loss="binary_crossentropy")

print("starting training...")

model.fit(input_set,output_set,batch_size=100,epochs=5,verbose=1)

model.save("review_model.h5")

model = load_model("review_model.h5")

print(model.predict_classes(test_list))
