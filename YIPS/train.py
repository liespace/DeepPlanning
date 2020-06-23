#!/usr/bin/env python
import json
from yips.model import DWModel


def main(config_filename):
    config = json.loads(open(config_filename).read())
    # Buildup and Training
    model = DWModel(config=config)
    model.compile().train()


if __name__ == '__main__':
    main('config.json')
