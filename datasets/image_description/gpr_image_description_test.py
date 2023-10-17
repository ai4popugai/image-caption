import os
from datasets.image_description.gpr_image_description import GPRConceptsDataset


def main():
    dataset = GPRConceptsDataset()
    item = dataset[1]
    print(item)


if __name__ == '__main__':
    main()