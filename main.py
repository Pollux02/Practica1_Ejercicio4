import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut, KFold

# Paso 1: Cargar y dividir los datos
data = pd.read_csv('irisbin.csv')
X = data.iloc[:, :-3]  # Características
y = data.iloc[:, -3]   # Etiquetas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Entrenar el modelo
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Paso 3: Evaluar el modelo
y_pred = model.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Paso 4: Validación cruzada
# Leave-one-out
loo = LeaveOneOut()
scores = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print("Leave-one-out accuracy:", sum(scores)/len(scores))

# Leave-k-out (k=5)
kf = KFold(n_splits=5)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print("Leave-k-out accuracy (k=5):", sum(scores)/len(scores))
