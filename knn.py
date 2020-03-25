import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

df = pd.read_csv('/Users/armenramazan/Desktop/machine_learning/coursera/week_2/feature normalization/train.csv')

my_columns = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]
df.columns = my_columns

X = df.loc[:, df.columns != 'Class']
y = df['Class']

cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

def get_best_score (X, y, cv):
    best_score, best_k = -1, -1

    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors = k)
        score = cross_val_score(model, X, y, cv = cv).mean()

        if best_score < score:
            best_score, best_k = score, k

    return best_score, best_k

score_1, k_1 = get_best_score(X, y, cv)
print (f"{score_1:.2f}", k_1)

score_2, k_2 = get_best_score(scale(X), y, cv)
print (f"{score_2:.2f}", k_2)

