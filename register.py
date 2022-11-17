from detectron2.data.datasets import register_coco_instances
import os

datasets = os.getenv('LOCAL_SCRATCH')

register_coco_instances("alien_barley_train",
        {},
        f"{datasets}/alien-barley/annotations/instances_train.json",
        f"{datasets}/alien-barley/train/")
register_coco_instances("alien_barley_unlabeled",
        {},
        f"{datasets}/alien-barley/annotations/instances_unlabeled.json",
        f"{datasets}/alien-barley/unlabeled/")
register_coco_instances("alien_barley_test",
        {},
        f"{datasets}/alien-barley/annotations/instances_test.json",
        f"{datasets}/alien-barley/test/")

register_coco_instances("inference",
        {},
        f"instances.json",
        f"tiles/")
