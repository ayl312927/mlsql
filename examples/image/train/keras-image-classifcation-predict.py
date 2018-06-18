from __future__ import absolute_import
from pyspark.ml.linalg import VectorUDT, Vectors
import pickle
import os
import numpy as np
from keras.models import load_model

run_for_test = True
if run_for_test:
    import mlsql.python_fun
else:
    import python_fun


def predict(index, items):
    modelPath = pickle.loads(items[1])[0] + "/model.h5"

    if not hasattr(os, "mlsql_models"):
        setattr(os, "mlsql_models", {})
    if modelPath not in os.mlsql_models:
        print("Load Keras model %s" % modelPath)
        os.mlsql_models[modelPath] = load_model(modelPath)
    # here we can get train params
    trainParams = pickle.loads(items[2])[0]
    width = int(trainParams["width"])
    height = int(trainParams["height"])

    model = os.mlsql_models[modelPath]
    rawVector = pickle.loads(items[0])
    feature = VectorUDT().deserialize(rawVector).toArray()
    y = model.predict(np.reshape(feature, [width, height, 3]), batch_size=1)
    return [VectorUDT().serialize(Vectors.dense(y))]


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
