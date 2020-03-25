import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('titanic.csv',index_col = 'Passenger_Id')

X = df.loc[:,['Pclass', 'Fare', 'Sex', 'Age']]
X['Sex'] = X['Sex'].map({'female' : 0, 'male' : 1})

y = df['Survived']

X = X.dropna()
y = y[X.index]

id3 = DecisionTreeClassifier(random_state= 241)
id3.fit(X, y)

f_i = pd.Series(id3.feature_importances_, index=X.columns).sort_values(ascending=False)

print(" ".join(f_i[:2].index.values))