#sidi baat no bakwas
from flask import Flask, jsonify, request
from flask_cors import CORS 
from brains import sumup

app = Flask(__name__)
CORS(app)

# GET all items
@app.route('/isup', methods=['GET'])
def chck():
    return "I'm up"

# Summarize text
@app.route('/summarize', methods=['POST'])
def sumup_api():

    queryPL = request.json
    return jsonify(sumup(queryPL.get('text'), queryPL.get('lang')))


if __name__ == '__main__':
    app.run(debug=True)
