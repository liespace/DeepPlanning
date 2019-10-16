from dataset import Pipeline
from model import DWModel
from cores import DWDark53, DWRes50, DWDark19, DWVGG19
import json
import os
import tensorflow as tf

cfg_name = 'config'
cfg_path = os.getcwd() + os.sep + 'src' + os.sep + cfg_name + '.json'
with open(cfg_path) as handle:
    config = json.loads(handle.read())

if config['Model']['backbone'] == 'dark53':
    print ('Running DWDark53')
    core = DWDark53(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        weights=False)
elif config['Model']['backbone'] == 'res50':
    print ('Running DWRes50')
    core = DWRes50(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        weights=config['Model']['weights'])
elif config['Model']['backbone'] == 'dark19':
    print ('Running DWDark19')
    core = DWDark19(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        weights=False)
else:
    tf.logging.warning('Running DWVGG19')
    core = DWVGG19(
        i_shape=tuple(config['Model']['i_shape']),
        a=config['Model']['A'],
        b=config['Model']['B'],
        weights=config['Model']['weights'])

model = DWModel(core=core, config=config, pipeline=Pipeline(config=config))
model.compile(config['Model']['summary'])
# model.predict_generator(weights_file='../checkpoint-800.h5')
model.train()
