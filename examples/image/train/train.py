from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
import numpy as np
import os
import json
import sys

run_for_test = True

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

tempDataLocalPath = mlsql.internal_system_param["tempDataLocalPath"]


def getFitParamWithDefault(key, value):
    res = None
    if key in mlsql.fit_param:
        res = mlsql.fit_param[key]
    else:
        res = value
    return res


width = int(getFitParamWithDefault("width", 100))
height = int(getFitParamWithDefault("height", 100))
labelSize = int(getFitParamWithDefault("labelSize", 10))
featureCol = getFitParamWithDefault("featureCol", "features")
labelCol = getFitParamWithDefault("labelCol", "label")


def get_validate_data():
    X = []
    y = []
    for item in mlsql.validate_data:
        X.append(item[featureCol])
        y.append(item[labelCol])
    return np.array(X), np.array(y)


X_test, y_test = get_validate_data()

input_tensor = Input(shape=(width, height, 3), name="X")

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(labelSize, activation='softmax', name="Y")(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
datafiles = [file for file in os.listdir(tempDataLocalPath) if file.endswith(".json")]

rd = mlsql.read_data()


def from_kafka():
    for items in rd(max_records=64):
        _x = [item[featureCol].toArray() for item in items]
        _y = [item[labelCol] for item in items]
        print(np.array(_x).shape)
        print(np.array(_y).shape)
        yield ({'X': _x}, {'Y': _y})


def from_file():
    while True:
        _x = []
        _y = []
        for file in datafiles:
            with open(tempDataLocalPath + "/" + file) as f:
                for line in f.readlines():
                    obj = json.loads(line)
                    _x.append(np.reshape(obj[featureCol], [100, 100, 3]))
                    _y.append(obj[labelCol])

                    if len(_x) == 64:
                        yield (np.array(_x), np.array(_y))
                        _x = []
                        _y = []


fetchData = from_file if len(datafiles) > 0 else from_kafka


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


model.fit_generator(fetchData(), epochs=10, steps_per_epoch=564 / 64, validation_data=(X_test, y_test),
                    callbacks=[TestCallback((X_test, y_test))])

isp = mlsql.params()["internalSystemParam"]

if "tempModelLocalPath" not in isp:
    raise Exception("tempModelLocalPath is not configured")

tempModelLocalPath = isp["tempModelLocalPath"]

if not os.path.exists(tempModelLocalPath):
    os.makedirs(tempDataLocalPath)
model.save(tempModelLocalPath + "/dxy-d-train.h5")

# print(model.predict(np.array(X)))
