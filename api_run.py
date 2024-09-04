from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Bonjour depuis l'API!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)