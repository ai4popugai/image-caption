import os

from torch.utils.data import Dataset

DUBAI_AERIAL_DATASET = 'DUBAI_AERIAL_DATASET'


class DubaiAerial(Dataset):
    def __init__(self):
        super().__init__()
        if DUBAI_AERIAL_DATASET not in os.environ:
            raise RuntimeError('Dataset root not in environment.')
        self.root = os.environ[DUBAI_AERIAL_DATASET]
        tile_list = [tile for tile in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, tile))]
        self.image_path_list = []
        self.ground_true_path_list = []
        for tile in tile_list:
            tile_path = os.path.join(self.root, tile)
            self.image_path_list += [os.path.join(tile_path, 'images', img)
                                     for img in sorted(os.listdir(os.path.join(tile_path, 'images')))]
            self.ground_true_path_list += [os.path.join(tile_path, 'masks', img)
                                           for img in sorted(os.listdir(os.path.join(tile_path, 'masks')))]

    def __len__(self) -> int:
        return len(self.image_path_list)


