#!/usr/bin/env python
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import logging
import artm
import artm.artm_model
import os
import sys
import uuid

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

@app.route('/api/fit', methods=['POST'])
def fit():
    batch_id = str(uuid.uuid4())
    app.logger.info("batch %s", batch_id)

    rjson = request.json
    terms = rjson['terms']
    topics_cnt = rjson['topics']

    batch = artm.messages.Batch()
    term_to_id = {}
    all_terms = []
    
    batch = artm.messages.Batch()
    batch.id = batch_id

    for i, doc in enumerate(terms):
        item = batch.item.add()
        item.id = i
        field = item.field.add()
        for term in doc:
            if not term in term_to_id:
                term_to_id[term] = len(all_terms)
                all_terms.append(term)
            field.token_id.append(term_to_id[term])
            field.token_count.append(1)
    
    for t in all_terms:
        batch.token.append(t)

    os.mkdir(batch_id)
    with open(os.path.join(batch_id, "batch.batch"), 'wb') as fout:
        fout.write(batch.SerializeToString())    

    app.logger.info("batch %s is created", batch_id) 

    dictionary = artm.Dictionary()
    dictionary.gather(batch_id)

    model_artm = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(topics_cnt)],
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary)],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15)],
                           show_progress_bars=False) 

    batch_vectorizer = artm.BatchVectorizer(data_path=batch_id, data_format="batches")

    model_artm.initialize(dictionary=dictionary)
    app.logger.info("model is starting to fit")
    model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)
    app.logger.info("mode was fitted")

    model_artm.save(os.path.join(batch_id, "model"))

    return jsonify({"id": batch_id})

@app.route('/api/topics', methods=['POST'])
def topics():
    term = request.json['term']
    batch_id = request.json['id']
    model_path = os.path.join(batch_id, "model")

    if not os.path.exists(model_path):
        return jsonify({"error": u"Unknown id: {}".format(batch_id)}), 500

    model_artm = artm.ARTM(num_topics=0)
    model_artm.load(model_path)

    phi = model_artm.get_phi()
    if not term in phi.index:
        return jsonify({"error": u"Unknown term: {}".format(term)}), 500

    phi_term = phi.loc[term].astype(float)
    phi_term /= phi_term.sum()

    return jsonify({"topics": list(phi_term)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
