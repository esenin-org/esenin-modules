#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import logging

import os
import ipywidgets as widgets
import tensorflow as tf
from IPython import display
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import spec_builder
from dragnn.python import load_dragnn_cc_impl  # This loads the actual op definitions
from dragnn.python import render_parse_tree_graphviz
from dragnn.python import visualization
from google.protobuf import text_format
from syntaxnet import load_parser_ops  # This loads the actual op definitions
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from tensorflow.python.platform import tf_logging as logging

def load_model(base_dir, master_spec_name, checkpoint_name):
    # Read the master spec
    master_spec = spec_pb2.MasterSpec()
    with open(os.path.join(base_dir, master_spec_name), "r") as f:
        text_format.Merge(f.read(), master_spec)
    spec_builder.complete_master_spec(master_spec, None, base_dir)
    logging.set_verbosity(logging.WARN)  # Turn off TensorFlow spam.

    # Initialize a graph
    graph = tf.Graph()
    with graph.as_default():
        hyperparam_config = spec_pb2.GridPoint()
        builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
        # This is the component that will annotate test sentences.
        annotator = builder.add_annotation(enable_tracing=True)
        builder.add_saver()  # "Savers" can save and load models; here, we're only going to load.

    sess = tf.Session(graph=graph)
    with graph.as_default():
        #sess.run(tf.global_variables_initializer())
        #sess.run('save/restore_all', {'save/Const:0': os.path.join(base_dir, checkpoint_name)})
        builder.saver.restore(sess, os.path.join(base_dir, checkpoint_name))
        
    def annotate_sentence(sentence):
        with graph.as_default():
            return sess.run([annotator['annotations'], annotator['traces']],
                            feed_dict={annotator['input_batch']: [sentence]})
    return annotate_sentence

segmenter_model = load_model("/models/Russian-SynTagRus/segmenter", "spec.textproto", "checkpoint")
parser_model = load_model("/models/Russian-SynTagRus", "parser_spec.textproto", "checkpoint")

def syntaxnet_tokenize(text):
    sentence = sentence_pb2.Sentence(
        text=text,
        token=[sentence_pb2.Token(word=text, start=-1, end=-1)]
    )

    # preprocess
    with tf.Session(graph=tf.Graph()) as tmp_session:
        char_input = gen_parser_ops.char_token_generator([sentence.SerializeToString()])
        preprocessed = tmp_session.run(char_input)[0]
    segmented, _ = segmenter_model(preprocessed)
    tokens = []

    for t in sentence_pb2.Sentence.FromString(segmented[0]).token:
        tokens.append(t.word)

    return tokens

def syntaxnet_sentence(tokens):
    pb_tokens = []
    last_start = 0
    for token in tokens:
        token_bytes = token.encode("utf8")
        pb_tokens.append(sentence_pb2.Token(
            word=token_bytes, start=last_start, end=last_start + len(token_bytes) - 1)
        )
        last_start = last_start + len(token_bytes) + 1

    annotations, traces = parser_model(sentence_pb2.Sentence(
        text=u" ".join(tokens).encode("utf8"),
        token=pb_tokens
    ).SerializeToString())
    assert len(annotations) == 1
    assert len(traces) == 1
    return sentence_pb2.Sentence.FromString(annotations[0])

def esenin_dtree(sentence):
    nodes = []
    for t in sentence.token:    
        nodes.append({"label": t.label, "parent": t.head})
    return jsonify({"nodes": nodes})

def esenin_pos(sentence):
    def parse_tag(tag):
        result_dict = {}

        def remove_prefix(prefix, s):
            if s.startswith(prefix):
                return s[len(prefix):]
            else:
                raise ValueError(s + " doesn't start with " + prefix)

        def remove_suffix(suffix, s):
            if s.endswith(suffix):
                return s[:-len(suffix)]
            else:
                raise ValueError(s + " doesn't end with " + suffix)

        tag = remove_prefix("attribute { ", tag)
        tag = remove_suffix(" } ", tag)

        for part in tag.split(" } attribute { "):
            part = remove_prefix('name: "', part)
            k, part = part.split('"', 1)
            part = remove_prefix(' value: "', part)
            v, part = part.split('"', 1)
            result_dict[k] = v
        
        return result_dict

    pos = []
    for t in sentence.token:        
        tag_dict = parse_tag(t.tag)
        pos.append(tag_dict['fPOS'])
    
    return jsonify({"pos": pos})

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

@app.route('/api/pos', methods=['POST'])
def pos():
    tokens = request.json['tokens']
    return esenin_pos(syntaxnet_sentence(tokens))

@app.route('/api/dtree', methods=['POST'])
def dtree():
    tokens = request.json['tokens']
    return esenin_dtree(syntaxnet_sentence(tokens))

@app.route('/api/tokenize', methods=['POST'])
def tokenize():
    text = request.json['text']
    return jsonify({"tokens": syntaxnet_tokenize(text)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
