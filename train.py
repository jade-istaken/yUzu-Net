import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import metrics


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

#LOSS FUNCTION


class YUzuNetLoss(nn.Module):
    def __init__(self, img_size=512, strides=None, lambda_det=2.5, lambda_seg=1.0):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32]
        self.img_size = img_size
        self.strides = strides
        self.lambda_det = lambda_det
        self.lambda_seg = lambda_seg

    def dice_loss(self, preds, targets,smooth=1.0):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return 1 - dice

    def _cxcywh_to_xyxy(self, boxes):
        """Convert [cx, cy, w, h] -> [x1, y1, x2, y2] for torchvision ops"""
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    def forward(self, preds, targets, seg_pred, seg_target):
        device = preds[0].device
        loss_box, loss_obj, loss_cls = 0.0, 0.0, 0.0

        for stride, pred in zip(self.strides, preds):
            B, C, H, W = pred.shape
            # Decode predictions (YOLOv5/v8 style anchor-free)
            pred_xy = torch.sigmoid(pred[:, :2]) * 2.0 - 0.5  # Offset [-0.5, 1.5]
            pred_wh = torch.sigmoid(pred[:, 2:4]) * 4.0  # Scale [0, 4]
            pred_obj = pred[:, 4]  # Objectness logits
            pred_cls = pred[:, 5]  # Class logits

            # Create grid coordinates [B, 2, H, W]
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1).float()

            # Decode to normalized [0,1] image coordinates
            pred_boxes = torch.zeros(B, 4, H, W, device=device)
            pred_boxes[:, 0] = (grid_xy[:, 0] + pred_xy[:, 0]) / W  # x_center
            pred_boxes[:, 1] = (grid_xy[:, 1] + pred_xy[:, 1]) / H  # y_center
            pred_boxes[:, 2] = pred_wh[:, 0] * stride / self.img_size  # w
            pred_boxes[:, 3] = pred_wh[:, 1] * stride / self.img_size  # h

            # Target masks
            obj_mask = torch.zeros_like(pred_obj)
            gt_boxes_flat = torch.zeros(B, 4, H, W, device=device)
            cls_target = torch.zeros_like(pred_cls)

            # loop over GT boxes (N is small, negligible overhead)
            for b in range(B):
                gt = targets[b]  # [N, 5]
                if len(gt) == 0: continue

                # Map GT center to grid indices for this scale
                gx = (gt[:, 1] * self.img_size / stride).floor().clamp(0, W - 1).long()
                gy = (gt[:, 2] * self.img_size / stride).floor().clamp(0, H - 1).long()

                for i in range(len(gt)):
                    obj_mask[b, gy[i], gx[i]] = 1.0
                    gt_boxes_flat[b, :, gy[i], gx[i]] = gt[i, 1:5]
                    cls_target[b, gy[i], gx[i]] = gt[i, 0]

            pos = obj_mask.bool()
            if pos.sum() > 0:
                # Extract positive predictions & GT safely
                pred_pos = pred_boxes.permute(0, 2, 3, 1)[pos]  # [N_pos, 4]
                gt_pos = gt_boxes_flat.permute(0, 2, 3, 1)[pos]  # [N_pos, 4]

                # Convert to xyxy for torchvision
                pred_xyxy = self._cxcywh_to_xyxy(pred_pos)
                gt_xyxy = self._cxcywh_to_xyxy(gt_pos)

                # CIoU loss (using IoU as stable base)
                iou = ops.box_iou(pred_xyxy, gt_xyxy)
                loss_box += (1 - iou.diag()).mean()

                # Objectness & Classification
                loss_obj += F.binary_cross_entropy_with_logits(pred_obj, obj_mask)
                loss_cls += F.binary_cross_entropy_with_logits(pred_cls[pos], cls_target[pos])

        loss_det = (loss_box + loss_obj + loss_cls) / len(self.strides)

        # Segmentation Loss (Dice)
        loss_seg = self.dice_loss(seg_pred, seg_target)
        #print(f"Det: {loss_det:.3f} | Seg: {loss_seg.item():.3f}")

        return self.lambda_det * loss_det + self.lambda_seg * loss_seg

def visualize_preds(img, det_preds, stride=16, conf_thresh=0.25):
    device = img.device
    B, _, H, W = det_preds[1].shape  # P4 scale
    pred = det_preds[1][0]  # First image

    pred_xy = torch.sigmoid(pred[:2]) * 2.0 - 0.5
    pred_wh = torch.sigmoid(pred[2:4]) * 4.0
    pred_conf = torch.sigmoid(pred[4])

    gy, gx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    cx = (gx + pred_xy[0]) * stride
    cy = (gy + pred_xy[1]) * stride
    w  = pred_wh[0] * stride
    h  = pred_wh[1] * stride

    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2

    mask = pred_conf > conf_thresh
    boxes = torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], dim=1)

    # Plot (must move to CPU for matplotlib)
    img_np = TF.to_pil_image(img[0].cpu())
    plt.imshow(img_np)
    ax = plt.gca()
    for box in boxes.cpu().detach():
        ax.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            edgecolor='red', facecolor='none', linewidth=2
        ))
    plt.axis('off')
    plt.show()

def main():
    from model import YUzuNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = YUzuNetDataset(path='dataset')
    print(dataset.__len__())
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=yuzunet_collate_fn
    )

    yuzu = YUzuNet().to(device)
    criterion = YUzuNetLoss().to(device)
    optimizer = torch.optim.AdamW(yuzu.parameters(), lr=1e-3, weight_decay=1e-4)
    yuzu.train()

    val_metrics = {'mAP@50': [], 'precision': [], 'recall': [], 'iou': [], 'dice': []}
    for i in range (1,100):
        for imgs, masks, boxes in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            boxes = [b.to(device) for b in boxes]

            optimizer.zero_grad()
            det_preds, seg_pred = yuzu(imgs)
            loss = criterion(det_preds,boxes, seg_pred ,masks)
            loss.backward()
            optimizer.step()
            seg_m = metrics.seg_metrics(seg_pred, masks)
            det_m = metrics.det_metrics(det_preds, boxes)
            val_metrics['mAP@50'].append(det_m['mAP@50'])
            val_metrics['precision'].append(det_m['precision'])
            val_metrics['recall'].append(det_m['recall'])
            val_metrics['iou'].append(seg_m['iou'])
            val_metrics['dice'].append(seg_m['dice'])

if __name__ == '__main__':
    main()
