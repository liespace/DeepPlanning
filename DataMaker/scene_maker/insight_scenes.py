#!/usr/bin/env python
import json
import numpy

with open('./scene_maker/script.json') as handle:
    script = json.loads(handle.read())

scenes = script['scenes']
print ('amount of scenes:', len(scenes))
gap2d_from_sources_to_goals = []
for i, scene in enumerate(scenes):
    source = numpy.array([scene['source']['x'], scene['source']['y']])
    target = numpy.array([scene['target']['x'], scene['target']['y']])
    gap = numpy.linalg.norm(target-source)
    gap2d_from_sources_to_goals.append(gap)
    if gap > 20:
        print ('{}th scene is overline: '.format(i), gap, source, target)

gaps_array = numpy.array(gap2d_from_sources_to_goals)
print('average:', gaps_array.mean())
print('maximum:', gaps_array.max())
print('minimum:', gaps_array.min())
