#import nltk
#nltk.download()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

import pandas as pd
import numpy as np

data = pd.read_csv(r'Facts.csv', delimiter=";")
data = pd.concat([data["span"], data["Labeler_1 (yes/no)"]], axis=1, keys=["span", "ans"])
for i, row in zip(range(len(data["ans"])), data["ans"]):
    data["ans"][i] = row.lower() == "yes"
#======================
data2 = pd.read_csv(r'Facts2.csv', delimiter=";")
data2 = pd.concat([data2["span"], data2["Labeler_1 (yes/no)"]], axis=1, keys=["span", "ans"])
for i, row in zip(range(len(data2["ans"])), data2["ans"]):
    data2["ans"][i] = row.lower() == "yes"

print("Объединили")
data = pd.concat([data, data2], ignore_index=True)
print(data)
print(len(data["span"]))

data.dropna()
# убрать дубликаты
print(len(data["span"]))
data.drop_duplicates(subset="span", inplace=True)
print(len(data["span"]))


#stemmer
for i, row in zip(range(len(data["span"])), data["span"]):
    print(stemmer.stem(row))
    data["span"][i] = stemmer.stem(row)





data = data.sample(frac=1).reset_index(drop=True)

test_percentage = 0.05
train_count = int(len(data) - len(data) * test_percentage)
train_x = data["span"][:train_count].tolist()
train_y = data["ans"][:train_count].tolist()
test_x = data["span"][train_count:].tolist()
test_y = data["ans"][train_count:].tolist()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])

text_clf = text_clf.fit(train_x, train_y)

predicted = text_clf.predict(test_x)
print("Базовая точность: ", np.mean(predicted == test_y))


print("\n")
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train_x, train_y)

print("gs:")
print("GS точность: ", gs_clf.best_score_)
print("Лучшие параметры: ", gs_clf.best_params_)
print("----")

predicted = gs_clf.predict(test_x)
print("GS точность на тестовой выборке: ", np.mean(predicted == test_y))
print("\n\n")

while True:
    s = input("Давай сюда текст для проверки: ")
    res = gs_clf.predict([s])
    score = gs_clf.predict_proba([s])
    print("Тут вроде что-то про встречи" if res[0] else "Про встречи тут не говорят")
    print("Probability: ", score)
    print()

