# Importe as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Carregando um conjunto de dados de íris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, names=col_names)

# Usando todas as características neste caso
X = df.drop("class", axis=1)
y = df["class"]

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando histogramas para visualizar dados
df.hist()
plt.show()

# Aplicando o algoritmo ExtraTrees
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
accuracy = et.score(X_test, y_test)
print(f'Acurácia do modelo ExtraTrees: {accuracy * 100:.0f}%')
