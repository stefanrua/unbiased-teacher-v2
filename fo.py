import fiftyone as fo

data_path = "datasets/alien-barley/test/"
labels_path = "fiftyone/test-good/labels.json"

dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        )

session = fo.launch_app(dataset)
session.wait()
