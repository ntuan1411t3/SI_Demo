# By: Thuong Tran
# Refer to: https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/

import tensorflow as tf
from tensorflow.contrib import learn
from gensim.parsing.preprocessing import strip_tags,strip_multiple_whitespaces,strip_numeric,strip_short,stem_text,strip_punctuation
import gensim
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_string(s):
    filters = [strip_tags,strip_punctuation,strip_multiple_whitespaces,strip_numeric,strip_short,stem_text]
    s = gensim.utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

def preprocess_dict(x_test):
    for x_t in x_test:
        x= x_t
        break
    x_clean = x[x >0]
    x_new = np.zeros(x.shape[0], dtype= int)
    x_new[:x_clean.shape[0]] = x_clean
    return [x_new]

class ImportGraph():
  """  Importing and running isolated TF graph """
  def __init__(self, checkpoint_path, vocab_file, classes):
    # Create local graph and use it in the session
    self.session_config = tf.ConfigProto(
      #device_count = {'GPU': 0}
    )
    self.checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph, config=self.session_config)
    self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_file)
    self.classes = classes
    self.num_class = len(classes)
    with self.graph.as_default():
      # Import saved model from location 'loc' into local graph
      saver = tf.train.import_meta_graph(self.checkpoint_path + '.meta',  clear_devices=True)
      saver.restore(self.sess, self.checkpoint_path)
      # There are TWO options how to get activation operation:
      # # FROM SAVED COLLECTION:
      # self.activation = tf.get_collection('activation')[0]
      # BY NAME:
      # Get the placeholders from the graph by name
      self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
      self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
      # Tensors we want to evaluate
      self.scores = self.graph.get_operation_by_name("output/scores").outputs[0]


  def run(self, sentence):
    """ Running the activation operation previously imported """
    x_test = self.vocab_processor.transform([preprocess_string(sentence)])
    x_test = preprocess_dict(x_test)
    # Collect the predictions here
    predictions = self.sess.run(self.scores, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
    result = softmax(np.array(predictions[0]))
    # tmp = np.round(result,decimals=4)
    index = result.argsort()[::-1]
    responses = []
    for i in range(self.num_class):
        # print(self.classes[index[i]],tmp[index[i]])
        responses.append((self.classes[index[i]], result[index[i]]))
    return responses


