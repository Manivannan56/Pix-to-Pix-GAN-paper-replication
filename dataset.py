import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import config
from PIL import Image
from torchvision.utils import save_image

class ShoesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
     
     img_file = self.list_files[index]
     image_path = os.path.join(self.root_dir, img_file)
     image = np.array(Image.open(image_path))

     input_image = image[:, :256, :]
     target_image = image[:, 256:, :]

     input_image = config.transform_only_input(image=input_image)
     input_image_1= input_image['image']
     target_image = config.transform_only_mask(image=target_image)
     target_image_1=target_image['image']

     return input_image_1, target_image_1

      
if __name__ == "__main__":
    dataset = ShoesDataset("/Users/manivannans/Downloads/archive-2/edges2shoes/edges2shoes/train")
    dataloader = DataLoader(dataset=dataset, batch_size=5)
    for x, y in dataloader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()
