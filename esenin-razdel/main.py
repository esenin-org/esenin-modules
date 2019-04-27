#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import logging

import os

import razdel

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    else:
        app.logger.exception(e)
    return jsonify(error=repr(e)), code

def razdel_tokenize(text):
    tokens = []
    for t in razdel.tokenize(text):
        tokens.append(t.text)

    return tokens

@app.route('/api/token', methods=['POST'])
def tokenize():
    text = request.json['text']
    return jsonify({"tokens": razdel_tokenize(text)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
