# By Thuong Tran based on Thanh Nguyen
import numpy as np
import tensorflow as tf
import cv2
import os


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


_IMG_SIZE = 227


class ImportCNNGraph():
    """  Importing and running isolated TF graph """

    def __init__(self, checkpoint_path, classes):
        # Create local graph and use it in the session
        self.session_config = tf.ConfigProto(
            #device_count = {'GPU': 0}
        )
        self.checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.session_config)

        self.classes = classes
        self.num_class = len(classes)
        with self.graph.as_default():
            #   print(self.checkpoint_path)
            saver = tf.train.import_meta_graph(
                self.checkpoint_path + '.meta',  clear_devices=True)
            saver.restore(self.sess, self.checkpoint_path)
            self.input_x = self.graph.get_operation_by_name(
                "data/Input").outputs[0]
            self.scores = self.graph.get_operation_by_name(
                "output/output").outputs[0]

    def run(self, image_file):
        """ Running the activation operation previously imported """
        img = cv2.imread(image_file)
        img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = np.array(img,dtype= float)
        # img = img.ravel()
        # img /= 255

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img, dtype=float)
        img = img.ravel()
        img /= 255

        predictions = self.sess.run(self.scores, {self.input_x: [img]})
        result = softmax(np.array(predictions[0]))
        tmp = np.round(result, decimals=4)
        index = result.argsort()[::-1]
        responses = []

        for i in range(self.num_class):
            print(self.classes[index[i]], tmp[index[i]])
            responses.append((self.classes[index[i]], result[index[i]]))
        return responses


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = CURRENT_PATH + '/models_all/image_cnn'
MODEL_2_DIR = os.path.join(MODEL_DIR, 'cifar-10-2')
MODEL_6_DIR = os.path.join(MODEL_DIR, 'cifar-10-6')

TWO_CLASSES = np.array(['Normal', 'Other'])
SIX_CLASSES = np.array(['Normal', 'BoxBox', 'NonBox',
                        'NonKeyword', 'EtcDoc', 'HandWritten'])

model_2 = ImportCNNGraph(MODEL_2_DIR, TWO_CLASSES)
# model_6 = ImportCNNGraph(MODEL_6_DIR, SIX_CLASSES)


def predictSI(image_file):
    return model_2.run(image_file)


if __name__ == '__main__':
    predictSI(
        '/Users/tuan.cao/PycharmProjects/first_project/si_3000/si_3000_1/2895.png')
