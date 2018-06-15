import numpy as np
import os
import json
import sys
import pickle
import scipy.sparse as sp
from sklearn import svm
from pyspark.mllib.linalg import Vectors, SparseVector

run_for_test = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if run_for_test:
    if sys.version < '3':
        import cPickle as pickle
    else:
        import pickle

    unicode = str

    # config parameters
    PARAM_FILE = "python_temp.pickle"
    pp = {'internalSystemParam': {'tempModelLocalPath': '/tmp/__mlsql__/3242514c-4113-4105-bdc5-9987b28f9764/0',
                                  'tempDataLocalPath': '/Users/allwefantasy/Downloads/data', 'stopFlagNum': 9},
          'systemParam': {'pythonVer': '2.7', 'pythonPath': 'python'},
          'fitParam': {'labelCol': 'label', 'featureCol': 'feature', 'height': '100', 'width': '100',
                       'modelPath': '/tmp/pa_model', 'labelSize': '4'},
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


def getFitParamWithDefault(key, value):
    res = None
    if key in mlsql.fit_param:
        res = mlsql.fit_param[key]
    else:
        res = value
    return res


featureCol = getFitParamWithDefault("featureCol", "features")
labelCol = getFitParamWithDefault("labelCol", "label")


def load_sparse_data():
    import mlsql
    tempDataLocalPath = mlsql.internal_system_param["tempDataLocalPath"]
    # train the model on the new data for a few epochs
    datafiles = [file for file in os.listdir(tempDataLocalPath) if file.endswith(".json")]
    row_n = []
    col_n = []
    data = []
    y = []
    feature_size = 0
    row_index = 0
    for file in datafiles:
        with open(tempDataLocalPath + "/" + file) as f:
            for line in f.readlines():
                obj = json.loads(line)
                fc = obj[featureCol]
                if "size" not in fc:
                    feature_size = len(fc)
                    dic = [(i, a) for i, a in enumerate(fc)]
                    sv = SparseVector(len(fc), dic)
                else:
                    feature_size = fc["size"]
                    sv = Vectors.sparse(fc["size"], list(zip(fc["indices"], fc["values"])))

                for c in sv.indices:
                    row_n.append(row_index)
                    col_n.append(c)
                    data.append(sv.values[list(sv.indices).index(c)])

                if type(obj[labelCol]) is list:
                    y.append(np.array(obj[labelCol]).argmax())
                else:
                    y.append(obj[labelCol])
                row_index += 1
    print("X matrix : %s %s  row_n:%s col_n:%s" % (row_index, feature_size, len(row_n), len(col_n)))
    return sp.csc_matrix((data, (row_n, col_n)), shape=(row_index, feature_size)), y


model = svm.SVC()
X, y = load_sparse_data()
model.fit(X, y)

isp = mlsql.params()["internalSystemParam"]

if "tempModelLocalPath" not in isp:
    raise Exception("tempModelLocalPath is not configured")

tempModelLocalPath = isp["tempModelLocalPath"]

if not os.path.exists(tempModelLocalPath):
    os.makedirs(tempModelLocalPath)

pickle.dump(model, open(tempModelLocalPath + "/model.pkl", "wb"))

# print(model.predict(np.array(X)))
