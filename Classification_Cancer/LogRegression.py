import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('cancer.csv')
print(len(df.columns))

df.head()

df.drop(['id'], axis=1, inplace=True)

df.isnull().sum()

x = df.iloc[:, 1:28].values
y = df.iloc[:, 0].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = LogisticRegression()


params = {'penalty': ['l1', 'l2', 'elasticnet'],
          'solver':['newton-cg', 'lbfgs', 'sag', 'saga']}

grid = GridSearchCV(estimator = clf,   
                    param_grid = params, 
                    cv = 20)


grid.fit(x_train, y_train)

pd.DataFrame(grid.cv_results_)
pd.DataFrame(grid.cv_results_)[['params','rank_test_score','mean_test_score']]
grid.best_params_

grid.best_score_

y_preds = grid.predict(x_test)

print(confusion_matrix(y_test, y_preds))