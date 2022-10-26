from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO('datasets/alien-barley/annotations/instances_test.json')
pred = gt.loadRes('output/inference/coco_instances_results.json')
E = COCOeval(gt, pred, iouType='bbox')
E.params.iouThrs = [.5]
E.evaluate()
E.accumulate()
E.summarize()
