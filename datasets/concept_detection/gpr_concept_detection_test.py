import os
from datasets.concept_detection.gpr_concept_detection import GPRConceptsDataset


def main():
    os.environ['GPR_DATASET_CONCEPT_DETECTION'] = '/Users/iivan/datasets/gpr12000_concept_detection_v0'
    dataset = GPRConceptsDataset()
    item = dataset[1]
    print(item)


if __name__ == '__main__':
    main()