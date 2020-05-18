import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


train = pd.read_csv("/Users/armenramazan/Desktop/machine_learning/coursera/week_4/LinReg/salary-train.csv")

def transform(text: pd.Series) -> pd.Series:
    return text.str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)

vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(transform(train["FullDescription"]))

train["LocationNormalized"].fillna("nan", inplace=True)
train["ContractTime"].fillna("nan", inplace=True)

dict_vec = DictVectorizer()
X_train_dict = dict_vec.fit_transform(train[["LocationNormalized", "ContractTime"]].to_dict("records"))

X_train = hstack([X_train_dict, X_train_text])

y_train = train["SalaryNormalized"]
model = Ridge(random_state=241, alpha=1)
model.fit(X_train, y_train)
### _________________ 
test = pd.read_csv("/Users/armenramazan/Desktop/machine_learning/coursera/week_4/LinReg/salary-train.csv")
X_test_text = vec.fit_transform(transform(train["FullDescription"]))

test["LocationNormalized"].fillna("nan", inplace=True)
test["ContractTime"].fillna("nan", inplace=True)

X_test_dict = dict_vec.fit_transform(test[["LocationNormalized", "ContractTime"]].to_dict("records"))
X_test = hstack([X_test_dict, X_test_text])

y_test = model.predict(X_test)
print(f"{y_test[0]:.2f} {y_test[1]:.2f}")