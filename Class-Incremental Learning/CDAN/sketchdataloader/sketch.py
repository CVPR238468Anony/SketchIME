import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import json

class SketchPngDataset(data.Dataset):
    def __init__(self, root, list_txt_url, data_transforms = None) -> None:
        
        self.labelmap = [i for i in range(374)]
        self.relabelmap = [i for i in range(374)]
        # init category map
        pwd = os.path.dirname(__file__)
        with open(os.path.join(pwd, "data_partition.json"), "r") as ctx:
            map_file = json.load(ctx)
            index = 0
            for session in map_file:
                for cate in map_file[session]:
                    self.labelmap[int(cate) - 1] = index
                    self.relabelmap[index] = int(cate) - 1
                    index = index + 1
                    
        with open(list_txt_url, "r") as sketch_list_file:
            sketch_list = sketch_list_file.readlines()
            
            self.sketch_urls = [(os.path.join(root, sketch_url.strip().split(' ')[0])) for sketch_url in sketch_list]
            self.targets = [self.labelmap[int(sketch_url.strip().split(' ')[-1])] for sketch_url in sketch_list]
            self.transform = data_transforms
                    
    def __len__(self):
        return len(self.sketch_urls)

    def __getitem__(self, item):

        sketch_url = self.sketch_urls[item]

        target = self.targets[item]
        
        img = Image.open(sketch_url, 'r')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, target