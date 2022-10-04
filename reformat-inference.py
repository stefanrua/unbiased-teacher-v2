import json

label_file = 'datasets/alien-barley/annotations/instances_test.json'
inferences_file = 'output-test/inference/coco_instances_results.json'
inferences_file_out = 'output-test/inference/coco_instances_results_fixed.json'

with open(label_file, 'rb') as f:
    labels = json.load(f)

with open(inferences_file, 'r') as f:
    s = f.read()
    s = '{"annotations": ' + s + '}'
    inferences = json.loads(s)

for a in inferences['annotations']:
    a['category_id'] = 1
labels['annotations'] += inferences['annotations']
labels['categories'] = [
        {'supercategory': 'none', 'id': 0, 'name': 'label'},
        {'supercategory': 'none', 'id': 1, 'name': 'prediction'},
        ]

with open(inferences_file_out, 'w') as f:
    json.dump(labels, f)
