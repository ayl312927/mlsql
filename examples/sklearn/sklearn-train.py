from __future__ import absolute_import
import numpy as np
import os
import json
import sys
import pickle
import scipy.sparse as sp
import importlib
from pyspark.mllib.linalg import Vectors, SparseVector

if sys.version < '3':
    import cPickle as pickle

else:
    import pickle

    xrange = range

unicode = str

run_for_test = False
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
if run_for_test:
    from sklearn.naive_bayes import GaussianNB

    # config parameters
    PARAM_FILE = "python_temp.pickle"
    pp = {'internalSystemParam': {'tempModelLocalPath': '/tmp/__mlsql__/3242514c-4113-4105-bdc5-9987b28f9764/0',
                                  'tempDataLocalPath': '/Users/allwefantasy/Downloads/data1', 'stopFlagNum': 9},
          'systemParam': {'pythonVer': '2.7', 'pythonPath': 'python'},
          'fitParam': {'labelCol': 'label', 'featureCol': 'features', 'height': '100', 'width': '100',
                       'modelPath': '/tmp/pa_model', 'labelSize': '2', 'class_weight': '{"1":2}',
                       'moduleName': 'sklearn.svm',
                       'className': 'SVC'},
          'kafkaParam': {'topic': 'zhuhl_1528712229620', 'bootstrap.servers': '127.0.0.1:9092',
                         'group_id': 'zhuhl_test_0', 'userName': 'zhuhl', 'reuse': 'true'}}

    with open(PARAM_FILE, "wb") as f:
        pickle.dump(pp, f)

    # test data
    VALIDATE_FILE = "validate_table.pickle"
    # 1, 100,100,4
    with open(VALIDATE_FILE, "wb") as f:
        data = np.random.random((10, 100, 100, 3))
        pickle.dump([pickle.dumps({"feature": i, "label": [0, 0, 0, 1]}) for i in data.tolist()], f)

import mlsql


def param(key, value):
    if key in mlsql.fit_param:
        res = mlsql.fit_param[key]
    else:
        res = value
    return res


featureCol = param("featureCol", "features")
labelCol = param("labelCol", "label")
moduleName = param("moduleName", "sklearn.svm")
className = param("className", "SVC")

batchSize = int(param("batchSize", "64"))
labelSize = int(param("labelSize", "-1"))


def load_sparse_data():
    import mlsql
    tempDataLocalPath = mlsql.internal_system_param["tempDataLocalPath"]
    # train the model on the new data for a few epochs
    datafiles = [file for file in os.listdir(tempDataLocalPath) if file.endswith(".json")]
    row_n = []
    col_n = []
    data_n = []
    y = []
    feature_size = 0
    row_index = 0
    for file in datafiles:
        with open(tempDataLocalPath + "/" + file) as f:
            for line in f.readlines():
                obj = json.loads(line)
                fc = obj[featureCol]
                if "size" not in fc and "type" not in fc:
                    feature_size = len(fc)
                    dic = [(i, a) for i, a in enumerate(fc)]
                    sv = SparseVector(len(fc), dic)
                elif "size" not in fc and "type" in fc and fc["type"] == 1:
                    values = fc["values"]
                    feature_size = len(values)
                    dic = [(i, a) for i, a in enumerate(values)]
                    sv = SparseVector(len(values), dic)

                else:
                    feature_size = fc["size"]
                    sv = Vectors.sparse(fc["size"], list(zip(fc["indices"], fc["values"])))

                for c in sv.indices:
                    row_n.append(row_index)
                    col_n.append(c)
                    data_n.append(sv.values[list(sv.indices).index(c)])

                if type(obj[labelCol]) is list:
                    y.append(np.array(obj[labelCol]).argmax())
                else:
                    y.append(obj[labelCol])
                row_index += 1
                if row_index % 10000 == 0:
                    print("processing lines: %s, values: %s" % (str(row_index), str(len(row_n))))
                    # sys.stdout.flush()
    print("X matrix : %s %s  row_n:%s col_n:%s classNum:%s" % (
        row_index, feature_size, len(row_n), len(col_n), ",".join([str(i) for i in list(set(y))])))
    sys.stdout.flush()
    return sp.csc_matrix((data_n, (row_n, col_n)), shape=(row_index, feature_size)), y


def load_batch_data():
    import mlsql
    tempDataLocalPath = mlsql.internal_system_param["tempDataLocalPath"]
    datafiles = [file for file in os.listdir(tempDataLocalPath) if file.endswith(".json")]
    X = []
    y = []
    count = 0
    for file in datafiles:
        with open(tempDataLocalPath + "/" + file) as f:
            for line in f.readlines():
                obj = json.loads(line)
                fc = obj[featureCol]
                if "size" not in fc and "type" not in fc:
                    dic = [(i, a) for i, a in enumerate(fc)]
                    sv = SparseVector(len(fc), dic)
                elif "size" not in fc and "type" in fc and fc["type"] == 1:
                    values = fc["values"]
                    dic = [(i, a) for i, a in enumerate(values)]
                    sv = SparseVector(len(values), dic)
                else:
                    sv = Vectors.sparse(fc["size"], list(zip(fc["indices"], fc["values"])))
                count += 1
                X.append(sv.toArray())
                if type(obj[labelCol]) is list:
                    y.append(np.array(obj[labelCol]).argmax())
                else:
                    y.append(obj[labelCol])
                if count % batchSize == 0:
                    yield X, y
                    X = []
                    y = []


def create_alg(module_name, class_name):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()


def configure_alg_params(clf):
    def class_weight(value):
        if value == "balanced":
            clf.class_weight = value
        else:
            clf.class_weight = dict([(int(k), int(v)) for (k, v) in json.loads(value).items()])

    def max_depth(value):
        clf.max_depth = int(value)

    options = {
        "class_weight": class_weight,
        "max_depth": max_depth
    }

    def t(v, convert_v):
        if type(v) == float:
            return float(convert_v)
        elif type(v) == int:
            return int(convert_v)
        elif type(v) == list:
            json.loads(convert_v)
        elif type(v) == dict:
            json.loads(convert_v)
        elif type(v) == bool:
            return bool(convert_v)
        else:
            return convert_v

    for name in clf.get_params():
        if name in mlsql.fit_param:
            if name in options:
                options[name](mlsql.fit_param[name])
            else:
                dv = clf.get_params()[name]
                setattr(clf, name, t(dv, mlsql.fit_param[name]))


model = create_alg(moduleName, className)
configure_alg_params(model)

if not hasattr(model, "partial_fit"):
    X, y = load_sparse_data()
    model.fit(X, y)
else:
    assert labelSize != -1
    print("using partial_fit to batch_train:")
    batch_count = 0
    for X, y in load_batch_data():
        model.partial_fit(X, y, [i for i in xrange(labelSize)])
        batch_count += 1
        print("partial_fit iteration: %s, batch_size:%s" % (str(batch_count), str(batchSize)))

if "tempModelLocalPath" not in mlsql.internal_system_param:
    raise Exception("tempModelLocalPath is not configured")

tempModelLocalPath = mlsql.internal_system_param["tempModelLocalPath"]

if not os.path.exists(tempModelLocalPath):
    os.makedirs(tempModelLocalPath)

model_file_path = tempModelLocalPath + "/model.pkl"
print("Save model to %s" % model_file_path)
pickle.dump(model, open(model_file_path, "wb"))
