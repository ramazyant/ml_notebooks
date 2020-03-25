import pandas as pd
from sklearn.tree import DecisionTreeClassifier

file = pd.read_csv("train.csv")

data = file.loc[:, ["Pclass", "Fare", "Age", "Sex"]]

data["Sex"] = data["Sex"].map({"male" : 0, "female": 1})

survived = file["Survived"]

data = data.dropna()
survived = survived[data.index]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, survived)

feature_importances = pd.Series(clf.feature_importances_, index=data.columns).sort_values(ascending=False)
feature_importances

print (feature_importances)