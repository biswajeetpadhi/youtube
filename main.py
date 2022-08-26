
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train)
print(test)

print(train.shape)
print(test.shape)

print(train.isnull().values.any())
print(test.isnull().values.any())

print(train.columns)
print(test.columns)

cat_type = train.category_id.value_counts()
print(cat_type)
print(cat_type.shape)

from nltk.corpus import stopwords
stop_word = set(stopwords.words("english"))

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

import re


def pre_processing1(text_item1, index1, column1):
    if type(text_item1) is not int:
        string1 = ""

        url_pattern1 = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
        text_item1 = re.sub(url_pattern1, " ", text_item1)

        email_pattern1 = r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'
        text_item1 = re.sub(email_pattern1, " ", text_item1)

        spe_char1 = '[^a-zA-Z0-9\n]'
        text_item1 = re.sub(spe_char1, " ", text_item1)

        mult_space1 = '\s+'
        text_item1 = re.sub(mult_space1, " ", text_item1)

        text_item1 = text_item1.lower()

        for word1 in text_item1.split():
            if not word1 in stop_word:
                word1 = stemmer.stem(word1)
                string1 += word1 + " "

        train[column1][index1] = string1


import re


def pre_processing2(text_item2, index2, column2):
    if type(text_item2) is not int:
        string2 = ""

        url_pattern2 = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
        text_item2 = re.sub(url_pattern2, " ", text_item2)

        email_pattern2 = r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'
        text_item2 = re.sub(email_pattern2, " ", text_item2)

        spe_char2 = '[^a-zA-Z0-9\n]'
        text_item2 = re.sub(spe_char2, " ", text_item2)

        mult_space2 = '\s+'
        text_item2 = re.sub(mult_space2, " ", text_item2)

        text_item2 = text_item2.lower()

        for word2 in text_item2.split():
            if not word2 in stop_word:
                word2 = stemmer.stem(word2)
                string2 += word2 + " "

        test[column2][index2] = string2


print(train.columns)
print(test.columns)

for index1, row1 in train.iterrows():
    if type(row1["description"]) is str:
        pre_processing1(row1["description"], index1, "description")


for index2, row2 in test.iterrows():
    if type(row2["description"]) is str:
        pre_processing2(row2["description"], index2, "description")


print(train.head(20))
print(test.head(20))

x_train = train["description"]
y_train = train["category_id"]

x_test = test["description"]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

x_train = vectorizer.fit_transform(x_train)
x_test= vectorizer.transform(x_test)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss = 'hinge', alpha = 0.01, class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(y_pred)