import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train = pd.read_csv("/Users/armenramazan/Desktop/machine_learning/coursera/week_2/feature normalization/test.csv", header=None)
test = pd.read_csv("/Users/armenramazan/Desktop/machine_learning/coursera/week_2/feature normalization/train.csv", header=None)

x_train = train.loc[:, 1:]
y_train = train[0]

x_test = test.loc[:, 1:]
y_test = test[0]

model = Perceptron(max_iter = 5, tol = None, random_state = 241)
model.fit(x_train, y_train)

acc_before = accuracy_score(y_test, model.predict(x_test))
acc_before

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train).astype(float)
x_test_scaled = scaler.transform(x_test).astype(float)

model.fit(x_train_scaled, y_train)

acc_after = accuracy_score(y_test, model.predict(x_test_scaled))
acc_after

acc_diff = acc_after - acc_before
print("{0:.3f}".format(acc_diff))

