from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import pickle, re
import sqlite3
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
app.json_provider_class = LazyJSONEncoder
app.json = LazyJSONEncoder(app)

DATABASE = 'database.db'

def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            file_data BLOB
        )
    ''')
    conn.commit()
    conn.close()
    
create_table()
    
swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Documentation for Sentiment Analysis'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Analisa Sentiment')
    },
    host=LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 4000
tokenizer = Tokenizer(num_words=max_features, split=' ',lower=True)
sentiment = ['negative', 'neutral', 'positive']

def lowercase(s):
    return s.lower()

def remove_punctuation(s):
    s = re.sub('((www\.[^\s]+)|(https?:\/\/[^\s]+)|(http?:\/\/[^\s]+)|(http[^\s]+))',' ', s) #menghapus semua URL
    s = re.sub(r'(?:\\x[A-Fa-f0-9]{2})+', r'', s) #menghapus \xDD substring
    s = re.sub('[^0-9a-zA-Z]+', ' ', s) #menghilangkan semua karakter yang bukan huruf atau angka dan menggantinya dengan spasi.
    s = re.sub(r'\brt\b',' ', s) #menghapus awalan rt
    s = re.sub('gue','saya', s) # Mengganti kata "gue" dengan kata "saya"
    s = re.sub(r'\d+', '', s) #menghapus semua angka
    s = re.sub(r'\buser\b',' ', s) #menghapus kata 'user'
    s = re.sub(r':', ' ', s) #menggantikan karakter : dengan spasi 
    s = re.sub(' +', ' ', s) #menggantikan satu atau lebih spasi berturut-turut dengan satu spasi 
    s = re.sub('\n',' ',s) #menggantikan karakter newline (\n) dengan spasi 
    s = re.sub(r'pic.twitter.com.[\w]+', ' ', s) #menghapus semua tautan Twitter (pic.twitter.com)
    s = s.strip() #menghilangkan whitespace di awal dan di akhir teks
    s = re.sub(r'‚Ä¶', '', s)
    return s

def cleansing(sent):
    string = lowercase(sent)
    string = remove_punctuation(string)
    return string


# Load file sequences lstm
file_lstm = open("LSTM/x_pad_sequences.pickle","rb")
feature_file_from_lstm = pickle.load(file_lstm)
file_lstm.close()
model_file_from_lstm = load_model("LSTM/model.h5")

# Load file sequences rnn
file_rnn = open("RNN/x_pad_sequences.pickle","rb")
feature_file_from_rnn = pickle.load(file_rnn)
file_rnn.close()
model_file_from_rnn = load_model("RNN/model.h5")

# Load file NN feature 
count_vect = pickle.load(open("NN/feature.p", "rb"))
model = pickle.load(open("NN/model.p", "rb"))

# Endpoint NN teks
@swag_from("docs/nn_text.yml", methods=['POST'])
@app.route('/nn_text', methods=['POST'])
def nn_text():
    try:
        # Get the original text from the request
        original_text = request.form.get('text')
        text = cleansing(original_text)  
        text_features = count_vect.transform([text])  
        prediction = model.predict(text_features)[0] 
        conn = sqlite3.connect(DATABASE) #result in the database
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasentiment (text, sentiment) VALUES (?, ?)", (text, prediction))
        conn.commit()
        conn.close()
        #the response JSON
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using MLP",
            "data": {
                "text": original_text,
                "sentiment": prediction
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e: 
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500 

# Endpoint nn file
@swag_from("docs/nn_file.yml", methods=['POST'])
@app.route("/nn_file", methods=['POST'])
def nn_file():
    try:
        file = request.files.get('upload_file')
        print(file)
        if not file:
            return jsonify({'error': 'No File Uploaded'}), 400
        pretext = file.read().decode('utf-8')
        text = cleansing(pretext)
        text = count_vect.transform([text]).toarray()
        prediction = model.predict(text)[0]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasentiment (text, sentiment) VALUES (?, ?)", (text, prediction))
        conn.commit()
        conn.close()
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using MLP",
            "data": {
                "text": pretext,
                "sentiment": prediction
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500
     
# Endpoint LSTM teks
@swag_from("docs/LSTM_text.yml", methods=['POST'])
@app.route("/LSTM_text", methods=['POST'])
def lstm_text():
    try:
        original_text = request.form.get('text')
        text = cleansing(original_text)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasentiment (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using LSTM",
            "data": {
                "text": original_text,
                "sentiment": get_sentiment
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

# Endpoint LSTM file
@swag_from("docs/LSTM_file.yml", methods=['POST'])
@app.route("/LSTM_file", methods=['POST'])
def lstm_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No File Uploaded'}), 400
        pretext = file.read().decode('utf-8')
        text = cleansing(pretext)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasentiment (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using LSTM",
            "data": {
                "text": pretext,
                "sentiment": get_sentiment
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "eroor": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500

# Endpoint RNN teks
@swag_from("docs/rnn_text.yml", methods=['POST'])
@app.route("/rnn_text", methods=['POST'])
def rnn_text():
    try:
        original_text = request.form.get('text')
        text = cleansing(original_text)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_file_from_rnn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasentiment (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using RNN",
            "data": {
                "text": original_text,
                "sentiment": get_sentiment
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "error": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500
    
# Endpoint RNN file
@swag_from("docs/rnn_file.yml", methods=['POST'])
@app.route("/rnn_file", methods=['POST'])
def rnn_file():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No File Uploaded'}), 400
        pretext = file.read().decode('utf-8')
        text = cleansing(pretext)
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_file_from_rnn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasentiment (text, sentiment) VALUES (?, ?)", (text, get_sentiment))
        conn.commit()
        conn.close()
        json_response = {
            "status_code": 200,
            "description": "Result of Sentiment Analysis using RNN",
            "data": {
                "text": pretext,
                "sentiment": get_sentiment
            },
        }
        response_data = jsonify(json_response)
        return response_data
    except Exception as e:
        error_response = {
            "status_code": 500,
            "eroor": str(e)
        }
        response_data = jsonify(error_response)
        return response_data, 500 

if __name__ == '__main__':
    app.run()
