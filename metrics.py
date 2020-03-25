import numpy as np
from sklearn.preprocessing import scale
import sklearn.datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

df = sklearn.datasets.load_boston()

X = scale(df['data'])
y = df['target']

cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_score, best_p = None, None

for p in np.linspace(1, 10, num = 200):
    model = KNeighborsRegressor(p = p, n_neighbors=5, weights='distance')
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()

    if best_score == None or best_score < score:
        best_score = score
        best_p = p

print(best_p)