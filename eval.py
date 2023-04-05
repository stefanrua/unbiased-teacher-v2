from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

                   # For example:
gt = sys.argv[1]   # datasets/alien-barley/annotations/instances_test.json
pred = sys.argv[2] # output/inference/coco_instances_results.json

gt = COCO(gt)
pred = gt.loadRes(pred)

E = COCOeval(gt, pred, iouType='bbox')
E.params.iouThrs = [.5]
E.evaluate()
E.accumulate()
E.summarize()
