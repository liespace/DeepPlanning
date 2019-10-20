from dataset import Pipeline
from model import DWModel
from cores import Core
import json
import os
import tensorflow as tf

cfg_name = 'config'
cfg_path = os.getcwd() + os.sep + 'src' + os.sep + cfg_name + '.json'
with open(cfg_path) as handle:
    config = json.loads(handle.read())

model = DWModel(
    config=config,
    core=Core(config=config),
    pipeline=Pipeline(config=config))
model.compile(config['Train']['summary'])
# predictions = model.predict_generator(weights_file='../checkpoint-800.h5')
model.train()
