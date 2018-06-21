from __future__ import absolute_import
from pyspark.ml.linalg import VectorUDT, Vectors
import pickle
import os
import numpy as np

run_for_test = False
if run_for_test:
    import mlsql.python_fun
else:
    import python_fun


def predict(index, s):
    items = [i for i in s]
    modelPath = pickle.loads(items[1])[0] + "/model.h5"
    if not hasattr(os, "mlsql_models"):
        setattr(os, "mlsql_models", {})
    if modelPath not in os.mlsql_models:
        # import tensorflow as tf
        # from keras import backend as K
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        # session = tf.Session(config=config)
        # K.set_session(session)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Load Keras model %s, CUDA_VISIBLE_DEVICES:%s " % (modelPath, os.environ["CUDA_VISIBLE_DEVICES"]))
        from keras.models import load_model
        os.mlsql_models[modelPath] = load_model(modelPath)
    # here we can get train params
    trainParams = pickle.loads(items[2])[0]
    width = int(trainParams["fitParam.0.width"])
    height = int(trainParams["fitParam.0.height"])

    model = os.mlsql_models[modelPath]
    rawVector = pickle.loads(items[0])
    feature = VectorUDT().deserialize(rawVector).toArray()
    feature_final = np.reshape(feature, [1, width, height, 3])
    # y是一个numpy对象，是一个预测结果的数组。因为predict是支持批量预测的，所以是一个二维数组。
    y = model.predict(feature_final)
    return [VectorUDT().serialize(Vectors.dense(y.tolist()[0]))]


if run_for_test:
    import json

    model_path = '/tmp/__mlsql__/3242514c-4113-4105-bdc5-9987b28f9764/0'
    data_path = '/Users/allwefantasy/Downloads/data1/part-00000-03769d42-1948-499b-8d8f-4914562bcfc8-c000.json'

    with open(file=data_path) as f:
        for line in f.readlines():
            s = []
            wow = json.loads(line)['features']
            feature = Vectors.sparse(wow["size"], list(zip(wow["indices"], wow["values"])))
            s.insert(0, pickle.dumps(VectorUDT().serialize(feature)))
            s.insert(1, pickle.dumps([model_path]))
            s.insert(2, pickle.dumps([{"width": "100", "height": "100"}]))
            print(VectorUDT().deserialize(predict(1, s)[0]))

python_fun.udf(predict)
