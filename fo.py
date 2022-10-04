import fiftyone as fo

# A name for the dataset
name = "alien-barley-test"

# The directory containing the source images
data_path = "datasets/alien-barley/test/"

# The path to the COCO labels JSON file
#labels_path = "datasets/alien-barley/annotations/instances_test.json"
labels_path = "output-test/inference/coco_instances_results_fixed.json"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
)

session = fo.launch_app(dataset)
session.wait()
