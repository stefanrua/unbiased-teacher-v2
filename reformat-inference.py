import json

label_file = 'datasets/alien-barley/annotations/instances_test.json'
inferences_file = 'output-test/inference/coco_instances_results.json'
inferences_file_out = 'output-test/inference/coco_instances_results_fixed.json'

with open(label_file, 'rb') as f:
    labels = json.load(f)

with open(inferences_file, 'rb') as f:
    inferences = json.load(f)

for a in inferences['annotations']:
    a['category_id'] = 1
labels['annotations'] += inferences['annotations']
labels['categories'] = [
        {'supercategory': 'none', 'id': 0, 'name': 'labels'},
        {'supercategory': 'none', 'id': 1, 'name': 'predictions'},
        ]

print(labels)

with open(inferences_file_out, 'w') as f:
    json.dump(labels, f)
