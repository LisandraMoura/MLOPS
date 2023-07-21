from textblob import TextBlob
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
import pickle


modelo = pickle.load(open('modelo.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)

app.config['BASIC_AUTH_USERNAME'] = 'AUAU'
app.config['BASIC_AUTH_PASSWORD'] = 'AUAU'

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return 'Hello World'

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return 'polaridade: {}'.format(polaridade)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True)
