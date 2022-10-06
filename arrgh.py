import json
import os

train_file = 'datasets/alien-barley/annotations/instances_train.json'
label_file = 'fiftyone/test-good/labels.json'
gt_file = 'arrgh/gt.json'
pred_file = 'arrgh/pred.json'

with open(label_file, 'rb') as f:
    labels = json.load(f)

with open(train_file, 'rb') as f:
    train_instances = json.load(f)

ann = labels['annotations'] 
gt = labels
pred = []
gt['annotations'] = [x.copy() for x in ann if x['category_id'] == 0]
pred = [x.copy() for x in ann if x['category_id'] == 1]
for a in pred:
    a['category_id'] = 0

train_set = set()
for a in train_instances['annotations']:
    train_set = train_set | {a['image_id']}

test_set = set()
for a in gt['annotations']:
    test_set = test_set | {a['image_id']}

print('datasets:')
print('  train', len(train_set))
print('  test ', len(test_set))

print('annotations:')
print('  gt   ', len(gt['annotations']))
print('  pred ', len(pred))

with open(gt_file, 'w') as f:
    f.write(json.dumps(gt))

with open(pred_file, 'w') as f:
    f.write(json.dumps(pred))
