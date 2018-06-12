from __future__ import absolute_import
import mlsql


def sklearn_batch_data(fn):
    rd = mlsql.read_data()
    fitParams = mlsql.params()["fitParam"]
    batch_size = int(mlsql.get_param(fitParams, "batchSize", 1000))
    label_size = int(mlsql.get_param(fitParams, "labelSize", -1))
    x_name = mlsql.get_param(fitParams, "inputCol", "features")
    y_name = mlsql.get_param(fitParams, "label", "label")
    for items in rd(max_records=batch_size):
        if len(items) == 0:
            continue
        X = [item[x_name].toArray() for item in items]
        y = [item[y_name] for item in items]
        fn(X, y, label_size)


def sklearn_configure_params(clf):
    fitParams = mlsql.params()["fitParam"]

    def t(v, convert_v):
        if type(v) == float:
            return float(convert_v)
        elif type(v) == int:
            return int(convert_v)
        elif type(v) == list:
            if type(v[0]) == int:
                return [int(i) for i in v]
            if type(v[0]) == float:
                return [float(i) for i in v]
            return v
        else:
            return convert_v

    for name in clf.get_params():
        if name in fitParams:
            dv = clf.get_params()[name]
            setattr(clf, name, t(dv, fitParams[name]))


def sklearn_all_data():
    rd = mlsql.read_data()
    fitParams = mlsql.params()["fitParam"]
    X = []
    y = []
    x_name = fitParams["inputCol"] if "inputCol" in fitParams else "features"
    y_name = fitParams["label"] if "label" in fitParams else "label"
    debug = "debug" in fitParams and bool(fitParams["debug"])
    counter = 0
    for items in rd(max_records=1000):
        item_size = len(items)
        if debug:
            counter += item_size
            print("{} collect data from kafka:{}".format(fitParams["alg"], counter))
        if item_size == 0:
            continue
        X = X + [item[x_name].toArray() for item in items]
        y = y + [item[y_name] for item in items]
    return X, y
