from dataset import Pipeline
from model import DWModel
from cores import DWDark53, DWRes50, DWDark19, DWVGG19
import json
import os

cfg_path = os.getcwd() + os.sep + 'src' + os.sep + 'config.json'
with open(cfg_path) as handle:
    config = json.loads(handle.read())

if config['Model']['name'] == 'dark53':
    print ('Running DWDark53')
    core = DWDark53(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        c=config['Model']['C'])
elif config['Model']['name'] == 'res50':
    print ('Running DWRes50')
    core = DWRes50(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        c=config['Model']['C'])
elif config['Model']['name'] == 'dark19':
    print ('Running DWDark19')
    core = DWDark19(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        c=config['Model']['C'])
else:
    print ('Running DWVGG19')
    core = DWVGG19(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        c=config['Model']['C'])

model = DWModel(core=core, config=config, pipeline=Pipeline(config=config))
model.compile()
model.train()
