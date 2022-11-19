import os
import json
from torch.utils.data import Dataset
import os
import warnings
from PIL import Image
import numpy as np
from collections import defaultdict

class Sketch(Dataset):
    
    def __init__(self, root, train=True,
                 transform=None, target_transform=None, args=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self._exts = ['.jpg', '.jpeg', '.png']

        self.labelmap = [i for i in range(374)]
        self.relabelmap = [i for i in range(374)]

        data_partition=os.path.join(root, 'data_partition.json')
        list_txt_path=os.path.join(root, 'datalist')

        self.data = []
        self.targets = []
        self.synsets = []

        with open(data_partition, "r") as ctx:
            map_file = json.load(ctx)
            index = 0
            for session in map_file:
                for cate in map_file[session]:
                    self.labelmap[int(cate) - 1] = index
                    self.relabelmap[index] = int(cate) - 1
                    index = index + 1
        if self.train:
            list_type="session_"
        else:
            list_type="test_"

        list_cnt=0

        for i in range(args.session):
            list_txt_url=os.path.join(list_txt_path, list_type+str(i)+".txt")
            with open(list_txt_url, "r") as sketch_list_file:
                if self.train:
                    sketch_list = sketch_list_file.readlines()
                else :
                    sketch_list = sketch_list_file.readlines()[list_cnt:]
                    list_cnt+=len(sketch_list)
                for sketch_url in sketch_list:
                    self.data.append(os.path.join(root, sketch_url.strip().split(' ')[0]))
                    self.targets.append(self.labelmap[int(sketch_url.strip().split(' ')[-1])])

        if not args.no_order:
            self.targets = self._map_new_class_index(self.targets, args.orders)
        self.targets = np.array(self.targets)
        self.sub_indexes = defaultdict(list)
        target_max = np.max(self.targets)
        for i in range(target_max + 1):
            self.sub_indexes[i] = np.where(self.targets == i)[0]

    def __getitem__(self, idx):
        img = Image.open(self.data[idx],'r')
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)

    def _map_new_class_index(self, y, order):
        """Transforms targets for new class order."""
        return list(map(lambda x: order.index(x), y))

