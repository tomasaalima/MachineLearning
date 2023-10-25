# Importe a biblioteca Pandas
import pandas as pd

# Carregue a base de dados Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(url, header=None, names=col_names)

# Filtrar linhas e colunas
# Exemplo: Filtrar apenas as linhas onde a coluna 'sepal_length' é maior que 5
iris_filtrado = iris[iris['sepal_length'] > 5]

# Manipular linhas
# Exemplo: Reordenar as linhas de acordo com os valores na coluna 'sepal_width'
iris_ordenado = iris.sort_values(by='sepal_width')

# Manipular colunas
# Exemplo: Criar uma nova coluna 'petal_ratio' que é a razão entre 'petal_length' e 'petal_width'
iris['petal_ratio'] = iris['petal_length'] / iris['petal_width']

# Excluir colunas
# Exemplo: Excluir a coluna 'petal_ratio'
iris = iris.drop(columns=['petal_ratio'])

# Criar histogramas
import matplotlib.pyplot as plt

# Exemplo: Criar um histograma da coluna 'sepal_length'
plt.hist(iris['sepal_length'], bins=10, color='blue', alpha=0.7)
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Frequência')
plt.title('Histograma do Comprimento da Sépala')
plt.show()
