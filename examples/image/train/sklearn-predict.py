from __future__ import absolute_import
from pyspark.ml.linalg import VectorUDT, Vectors
import pickle
import os

run_for_test = False
if run_for_test:
    import mlsql.python_fun
else:
    import python_fun


def predict(index, items):
    modelPath = pickle.loads(items[1])[0] + "/model.pkl"

    if not hasattr(os, "mlsql_models"):
        setattr(os, "mlsql_models", {})
    if modelPath not in os.mlsql_models:
        print("Load sklearn model %s" % modelPath)
        os.mlsql_models[modelPath] = pickle.load(open(modelPath, "rb"))

    model = os.mlsql_models[modelPath]
    rawVector = pickle.loads(items[0])
    feature = VectorUDT().deserialize(rawVector)
    y = model.predict([feature.toArray()])
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
            print(VectorUDT().deserialize(predict(1, s)[0]))

python_fun.udf(predict)
