from utils import db_connect
engine = db_connect()

# your code here
train_data = pd.read_csv("https://raw.githubusercontent.com/pdeleonsoria/Decission-tree/main/data/processed/clean_train.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/pdeleonsoria/Decission-tree/main/data/processed/clean_test.csv")
train_data.head()

test_data.head()

#SEPARAR EN TRAIN Y TEST 
X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]


modelo = XGBClassifier(random_state = 42)
modelo.fit(X_train, y_train)

y_pred= modelo.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

param_grid = {"n_estimators": [100, 200, 300],"learning_rate": [0.01, 0.1, 0.2]}


grid_search = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)

modelo_opti = XGBClassifier(learning_rate= 0.2, n_estimators= 200, random_state = 42)
modelo_opti.fit(X_train, y_train)

y_pred= modelo.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

#GUARDAR

dump(modelo_opti, open("../models/xboostingmodel.sav", "wb"))
