import json

label_file = 'datasets/alien-barley/annotations/instances_test.json'
inferences_file = 'output/inference/coco_instances_results.json'

with open(label_file, 'rb') as f:
    labels = json.load(f)

with open(inferences_file, 'r') as f:
    inferences = f.read()
    inferences = json.loads(f)

labels['annotations'] = inferences

with open(inferences_file, 'wb') as f:
    json.dump(labels, f)
