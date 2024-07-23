from flask import Flask, request, jsonify
import os
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    return jsonify({'Posture': 'proper'})

if __name__ == '__main__':
    app.run(debug=True)