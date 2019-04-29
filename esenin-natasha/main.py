#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import logging

from natasha import NamesExtractor

import bisect

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

def natasha_named_entities(tokens):
    token_starts = []
    last_start = 0
    for token in tokens:
        token_starts.append(last_start)
        last_start += len(token) + 1

    text = " ".join(tokens)
    extractor = NamesExtractor()
    matches = extractor(text)
    entities = []
    for match in matches:
        app.logger.info(str(match))
        (start, finish) = match.span
        token_start = bisect.bisect_right(token_starts, start) - 1
        token_finish = bisect.bisect_left(token_starts, finish) - 1
        entities.append({"indexes": list(range(token_start, token_finish + 1)), "kind": "name"})

    return entities


@app.route('/api/ne', methods=['POST'])
def tokenize():
    tokens = request.json['tokens']
    return jsonify({"entities": natasha_named_entities(tokens)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
