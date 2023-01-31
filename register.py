from detectron2.data.datasets import register_coco_instances
import os

datasets = os.getenv('LOCAL_SCRATCH')

# alien-barley-1
register_coco_instances("alien_barley_train",
        {},
        f"{datasets}/alien-barley-1/annotations/instances_train.json",
        f"{datasets}/alien-barley-1/train/")
register_coco_instances("alien_barley_unlabeled",
        {},
        f"{datasets}/alien-barley-1/annotations/instances_unlabeled.json",
        f"{datasets}/alien-barley-1/unlabeled/")
register_coco_instances("alien_barley_test",
        {},
        f"{datasets}/alien-barley-1/annotations/instances_test.json",
        f"{datasets}/alien-barley-1/test/")

# alien-barley-2
register_coco_instances("alien_barley_2_train_labeled",
        {},
        f"{datasets}/alien-barley-2-train/annotations/train_labeled.json",
        f"{datasets}/alien-barley-2-train/train_labeled/")
register_coco_instances("alien_barley_2_train_unlabeled",
        {},
        f"{datasets}/alien-barley-2-train/annotations/train_unlabeled.json",
        f"{datasets}/alien-barley-2-train/train_unlabeled/")
register_coco_instances("alien_barley_2_val",
        {},
        f"{datasets}/alien-barley-2-train/annotations/val.json",
        f"{datasets}/alien-barley-2-train/train_labeled/")
register_coco_instances("alien_barley_2_test",
        {},
        f"{datasets}/alien-barley-2-test/annotations/test.json",
        f"{datasets}/alien-barley-2-test/test/")

# for barleynet
register_coco_instances("inference",
        {},
        f"instances.json",
        f"tiles/")
