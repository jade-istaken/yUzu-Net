import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import cv2


class YUzuNetDataset(Dataset):
    """Create a dataset for the YUzuNet instance."""
    #expects to be given a path with 3 folders, images, masks, and labels.
    #images contains the base images. masks contains png segmentation masks. labels contains yolo-formatted bounding box labels
    def __init__(self, path, size=512, verbose=False):
        self.data_folder= path
        self.image_folder = os.path.join(self.data_folder, "images")
        self.mask_folder = os.path.join(self.data_folder, "masks")
        self.label_folder = os.path.join(self.data_folder, "labels")
        self.img_ids = [f for f in os.listdir(self.image_folder)
                        if os.path.isfile(os.path.join(self.image_folder, f))]
        self.size=size # this is just the size that it gets resized to. I'm leaving this a parameter, but I genuinely don't know what'd happen if it was anything other than 512. buyer beware I guess!

        self.mask_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((self.size, self.size)),
             transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST, antialias=True) #we use nearest neighbor interpolation so that the pixel boundaries don't get messed up during resizing
        ])

        self.img_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((self.size, self.size))
        ])

        self.verbose = verbose # this is a surprise tool that will help us later

    def __len__(self):
        return len(self.img_ids)

    def _load_yolo_labels(self, label_path):
        """Parses YOLO .txt format into a [N, 5] tensor: [class, x_center, y_center, w, h]"""
        if not os.path.exists(label_path):
            if self.verbose: print(f"Couldn't read file at {label_path}, skipping")
            return torch.zeros(0,5)

        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(p) for p in parts]) #ONLY accept it if it's a proper yolo formatted label

        if not boxes:
            return torch.zeros((0,5)) # no objects at all :(

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        image_name: str = self.img_ids[idx]
        img_path = os.path.join(self.image_folder, image_name)
        mask_path = os.path.join(self.mask_folder, os.path.splitext(image_name)[0] + ".png")
        label_path = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + ".txt")



        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transforms(img)

        mask = cv2.imread(mask_path, 0)
        mask = self.mask_transforms(mask)
        mask = torch.where(mask > 0.5, 1, 0).float() # this just REALLY clamps the mask down to make sure there's no noise from the transform or anything

        boxes = self._load_yolo_labels(label_path)

        return img, mask, boxes

def yuzunet_collate_fn(batch):
    """Handles variable number of boxes per image."""
    #IF THE DATALOADER DOESN'T USE THIS COLLATE FUNCTION IT'S GOING TO BE A BAD TIME
    images, masks, boxes = zip(*batch)
    return torch.stack(images), torch.stack(masks), boxes