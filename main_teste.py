from flask import Flask, request, jsonify

from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("raw.githubusercontent.com_alura-cursos_1576-mlops-machine-learning_aula-5_casas.csv")

colunas = ['tamanho', 'ano', 'preco', 'garagem']


X = df.drop('preco', axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

app = Flask('meu_app')


@app.route('/')
def home():
    return "Hello World"
    
@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)
    
"""@app.route('/cotacao/<int:tamanho>/<int:ano>/<int:garagem>')
def cotacao(tamanho, ano, garagem):
    preco = modelo.predict([[tamanho, ano, garagem]])
    return str(preco)"""
    
    
@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run()
#app.run(degub=True)