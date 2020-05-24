# Desafio-Titanic
# Este desafio foi replicado na plataforma Google Colab

# Importando as bibliotecas necessárias

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;

# Permitir visualizar todas as colunas

pd.options.display.max_columns = None

# Renderizar os gráficos diretamento no notebook
# %matplotlib inline


# Importando o arquivo CSV do Google Drive

from google.colab import drive
drive.mount('/content/drive')

# Lendo o arquivo CSV importado direto do Google Drive

test = pd.read_csv('/content/drive/My Drive/test.csv')
train = pd.read_csv('/content/drive/My Drive/train.csv')

# Verificando as dimensões do DataFrame

print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))

# Identificar o tipo de cada variável

display(train.dtypes)

# Ver as 5 primeiras entradas do conjunto de treino

display(train.head())

# Ver a porcentagem valores faltantes

(train.isnull().sum() / train.shape[0]).sort_values(ascending=False)

train.describe()

# Ver histograma das variáveis numéricas

train.hist(figsize=(10,8));

# Analisar a probabilidade de sobrevivência pelo Sexo

train[['Sex', 'Survived']].groupby(['Sex']).mean()

# Plotar os gráficos para Survived vs. Sex, Pclass e Embarked

fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(12,4))

sns.barplot(x='Sex', y='Survived', data=train, ax=axis1)
sns.barplot(x='Pclass', y='Survived', data=train, ax=axis2)
sns.barplot(x='Embarked', y='Survived', data=train, ax=axis3);

# Ver influência da idade na probabilidade de sobrevivência

age_survived = sns.FacetGrid(train, col='Survived')
age_survived.map(sns.distplot, 'Age')

# Plotar uma scatter matrix

columns=['Parch', 'SibSp', 'Age', 'Pclass']
pd.plotting.scatter_matrix(train[columns], figsize=(15, 10));

# Plotar o heatmap para as variáveis numéricas

sns.heatmap(train.corr(), cmap='coolwarm', fmt='.2f', linewidths=0.1,
            vmax=1.0, square=True, linecolor='white', annot=True);
           
           
train.describe(include=['O'])

# Salvar os índices dos datasets para recuperação posterior

train_idx = train.shape[0]
test_idx = test.shape[0]

# Salvar PassengerId para submissao ao Kaggle

passengerId = test['PassengerId']

# Extrair coluna 'Survived' e excluir ela do dataset treino

target = train.Survived.copy()
train.drop(['Survived'], axis=1, inplace=True)

# Concatenar treino e teste em um único DataFrame

df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Completar ou apagar valores faltantes nos datasets de treino e teste

df_merged.isnull().sum()

#Preenchendo valores de idade que faltam com a mediana
age_median = df_merged['Age'].median()
df_merged['Age'].fillna(age_median, inplace=True)

#Preenchendo valores de tarifa que faltam com a mediana
fare_median = df_merged['Fare'].median()
df_merged['Fare'].fillna(fare_median, inplace=True)

#Preenchendo valores de embarque que faltam com a mediana
embarked_top = df_merged['Embarked'].value_counts([0])
df_merged['Embarked'].fillna(embarked_top, inplace=True)

# Converter 'Sex' em 0 e 1

df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})

# Dummie variables para 'Embarked'

embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)

display(df_merged.head())

# Recuperando dataset de treino e teste

train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]

# Importando bibliotecas de ML

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Criação do modelo de regressão logística

lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(train, target)

# Teste de precisão do modelo

acc_logReg = round(lr_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Regressão Logística: {}".format(acc_logReg))
#Out: Precisão (acurácia) do modelo de Regressão Logística: 80.13

y_pred_lr = lr_model.predict(test)

submission = pd.DataFrame({
    "PassengerId": passengerId,
    "Survived": y_pred_lr
})

# Gerando arquivo CSV

submission.to_csv('./submission_lr.csv', index=False)

# Criação do modelo de árvore de decisão

tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(train, target)

# Teste de precisão do modelo

acc_tree = round(tree_model.score(train, target) * 100, 2)
print("Acurácia do modelo de Árvore de Decisão: {}".format(acc_tree))
#Out: Precisão (acurácia) do modelo de Árvore de Decisão: 82.72

y_pred_tree = tree_model.predict(test)

submission = pd.DataFrame({
    "PassengerId": passengerId,
    "Survived": y_pred_tree
})

# Gerando novo arquivo csv com a accuracy maior que a anterior

submission.to_csv('./submission_tree.csv', index=False)

# Declarando os valores das variáveis para mim e minha irmã

tiago_santos  = np.array([2, 0, 19, 0, 1, 32.2, 0, 0, 1]).reshape((1, -1))
beatriz_santos= np.array([2, 1, 23, 0, 1, 32.2, 0, 0, 1]).reshape((1, -1))

# Verificando se teríamos sobrevivido

print("Tiago Santos:\t{}".format(tree_model.predict(tiago_santos)[0]))
print("Beatriz Santos:\t{}".format(tree_model.predict(beatriz_santos)[0]))
Out: Tiago Santos:	0
    Beatriz Santos:	1
