from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO('arrgh/gt.json')
pred = gt.loadRes('arrgh/pred.json')
E = COCOeval(gt, pred, iouType='bbox')
E.params.iouThrs = [.5]
E.evaluate()
E.accumulate()
E.summarize()
