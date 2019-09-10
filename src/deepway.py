from dataset import Pipeline
from model import Model
from cores import DWDark53, DWRes50, DWDark19, DWVGG19
import json
import os

cfg_path = os.getcwd() + os.sep + 'src' + os.sep + 'config.json'
with open(cfg_path) as handle:
    config = json.loads(handle.read())

core = DWDark53(
    i_shape=tuple(config['Model']['i_shape']),
    a=config['Model']['A'],
    b=config['Model']['B'],
    c=config['Model']['C'])

model = Model(core=core, config=config, pipeline=Pipeline(config=config))
model.compile()
model.train()
