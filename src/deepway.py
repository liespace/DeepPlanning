from dataset import Pipeline
from model import DWModel
from cores import Core
import json
import os
import backend

cfg_name = 'config'
cfg_path = os.getcwd() + os.sep + 'src' + os.sep + cfg_name + '.json'
with open(cfg_path) as handle:
    config = json.loads(handle.read())

model = DWModel(
    config=config,
    core=Core(config=config),
    pipeline=Pipeline(config=config))
model.compile(config['Train']['summary'])
if config['Pred']['enable']:
    pred = model.predict_generator(weights_file=config['Pred']['weights_file'])
    backend.save_predictions(pred)
else:
    model.train()
