
import torch
import torchvision.ops as ops
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

OPTIMAL_SCALING_TEMP = 1.135 #Temp taken from full-set training

# segmentation metrics
def seg_metrics(outputs, targets, threshold=0.5, eps=1e-6):
    """Helper to binarize, flatten, and compute TP/FP/FN safely per image"""
    outputs = (torch.sigmoid(outputs) >= threshold).float()
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (outputs * targets).sum(dim=1)
    fp = (outputs * (1 - targets)).sum(dim=1)
    fn = ((1 - outputs) * targets).sum(dim=1)

    union = tp + fp + fn
    iou_score = (tp / (union + eps)).mean().item()
    prec = (tp / (tp + fp + eps)).mean().item()
    rec = (tp / (tp + fn + eps)).mean().item()
    f1_score = (2 * prec * rec) / (prec + rec + eps)
    dice_score = ((2 * tp) / (2 * tp + fp + fn + eps)).mean().item()
    return {'iou': float(iou_score), 'dice': float(dice_score), 'f1': float(f1_score)}





def det_metrics(preds, targets, img_size=512, strides=None,
                conf_thresh=0.25, iou_thresh=0.5, temp=OPTIMAL_SCALING_TEMP):
    """
    Computes detection metrics for single-class YOLO-style heads.
    Returns: {'mAP@50': float, 'precision': float, 'recall': float}
    """
    if strides is None:
        strides = [8, 16, 32]
    device = preds[0].device
    all_preds, all_gts = [], []

    # Decode predictions across all scales
    for stride, pred in zip(strides, preds):
        B, _, H, W = pred.shape
        # Decode boxes (same logic as loss function)
        pred_xy = torch.sigmoid(pred[:, :2]) * 2.0 - 0.5
        pred_wh = torch.sigmoid(pred[:, 2:4]) * 4.0
        pred_conf = torch.sigmoid(pred[:, 4:5] / temp) # apply temp scaling to objectness logits

        # Grid coordinates
        gy, gx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing='ij')
        gx = gx.unsqueeze(0).unsqueeze(0).float()
        gy = gy.unsqueeze(0).unsqueeze(0).float()

        # Absolute coordinates in image space
        cx = (gx + pred_xy[:, 0:1]) * stride
        cy = (gy + pred_xy[:, 1:2]) * stride
        w = pred_wh[:, 0:1] * stride
        h = pred_wh[:, 1:2] * stride

        x1 = cx - w / 2;
        y1 = cy - h / 2
        x2 = cx + w / 2;
        y2 = cy + h / 2

        # Flatten to [B, N_pts, 5] -> [x1, y1, x2, y2, conf]
        det = torch.cat([x1, y1, x2, y2, pred_conf], dim=1)
        det = det.view(B, 5, H, W).permute(0, 2, 3, 1).reshape(B, -1, 5)
        all_preds.append(det)

    # Combine scales: [B, total_pts, 5]
    all_preds = torch.cat(all_preds, dim=1)

    # Filter by confidence, apply NMS per image, and collect GTs
    pred_boxes_list, pred_confs_list = [], []
    gt_boxes_list = []

    for b in range(all_preds.shape[0]):
        dets = all_preds[b]
        mask = dets[:, 4] > conf_thresh
        dets = dets[mask]

        if len(dets) > 0:
            keep = ops.nms(dets[:, :4], dets[:, 4], iou_thresh)
            dets = dets[keep]
            pred_boxes_list.append(dets[:, :4].cpu())
            pred_confs_list.append(dets[:, 4].cpu())
        else:
            pred_boxes_list.append(torch.zeros(0, 4))
            pred_confs_list.append(torch.zeros(0))

        # Convert GT from normalized [cx,cy,w,h] to absolute [x1,y1,x2,y2]
        gts = targets[b].cpu()
        if len(gts) > 0:
            cx, cy, w, h = gts[:, 1:5].T * img_size
            gt_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            gt_boxes_list.append(gt_xyxy)
        else:
            gt_boxes_list.append(torch.zeros(0, 4))

    # Match predictions to GTs & build TP/FP/Conf arrays
    tp, fp, conf_scores = [], [], []
    gt_count = 0

    for preds_i, confs_i, gts_i in zip(pred_boxes_list, pred_confs_list, gt_boxes_list):
        gt_count += len(gts_i)
        if len(gts_i) == 0:
            # No GTs in this image → all preds are False Positives
            tp.extend([0] * len(preds_i))
            fp.extend([1] * len(preds_i))
            conf_scores.extend(confs_i.tolist())
            continue

        if len(preds_i) == 0:
            # No preds → all GTs are missed (handled by denominator later)
            continue

        # Compute IoU matrix [N_pred, N_gt]
        ious = ops.box_iou(preds_i, gts_i)

        # Greedy matching (standard COCO protocol)
        matched_gt = torch.zeros(len(gts_i))
        for idx in range(len(preds_i)):
            best_iou, best_gt = ious[idx].max(0)
            if best_iou >= iou_thresh and matched_gt[best_gt] == 0:
                tp.append(1)
                fp.append(0)
                matched_gt[best_gt] = 1
            else:
                tp.append(0)
                fp.append(1)
            conf_scores.append(confs_i[idx].item())

        # Remaining unmatched GTs are False Negatives (counted via gt_count - sum(tp))

    # Compute Precision, Recall, and mAP@50
    if gt_count == 0:
        return {'mAP@50': 0.0, 'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'accuracy': 0.0}

    # Sort by confidence descending
    indices = np.argsort(conf_scores)[::-1]
    tp = np.array(tp)[indices]
    fp = np.array(fp)[indices]

    # Cumulative sums
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    prec = tp_cum / (tp_cum + fp_cum + 1e-6)
    rec = tp_cum / (gt_count + 1e-6)

    # Precision at threshold (using last valid recall)
    precision = prec[-1] if len(prec) > 0 else 0.0
    recall = rec[-1] if len(rec) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = gt_count - total_tp
    accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-6)

    # Interpolate AP@50 (standard 11-point or full curve)
    # Full curve integration (more accurate for custom models)
    prec_interp = np.maximum.accumulate(prec[::-1])[::-1]
    rec_interp = np.concatenate([[0], rec, [1]])
    prec_interp = np.concatenate([[0], prec_interp, [0]])

    ap = np.sum((rec_interp[1:] - rec_interp[:-1]) * prec_interp[1:])

    return {'mAP@50': float(ap), 'precision': float(precision), 'recall': float(recall), 'F1' : float(f1), 'accuract': float(accuracy)}

def object_count_metrics(det_preds, gt_boxes, conf_thresh=0.21, nms_iou_thresh=0.5):
    """
    Calculates RMSE and MAE for the number of predicted objects vs. ground truth objects,
    applying NMS to predicted objects.

    Args:
        det_preds (list): List of detection predictions (e.g., [p3, p4, p5]),
                          where each p is a tensor of shape (B, C, H, W).
                          C is total channels from all anchors at a location.
        gt_boxes (list): List of ground truth boxes per image in the batch.
                         Each element is a tensor of shape (N, 5) where 5 is (class_id, x, y, w, h).
        conf_thresh (float): Confidence threshold for filtering predicted objects.
        nms_iou_thresh (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        dict: Dictionary containing 'MAE' and 'RMSE'.
    """
    predicted_counts = []
    ground_truth_counts = []

    batch_size = det_preds[0].shape[0]

    # Each detection is assumed to have 6 features: (x, y, w, h, confidence, class_score)
    features_per_detection = 6

    for i in range(batch_size): # Iterate over images in the batch
        ground_truth_counts.append(len(gt_boxes[i]))
        image_decoded_preds = []
        for p_scale in det_preds: # p_scale is (B, C, H, W)
            # Take predictions for current image, permute to (H, W, C), and flatten to (H*W, C)
            preds_i = p_scale[i].permute(1, 2, 0).reshape(-1, p_scale.shape[1]) # Shape: (H*W, C)

            C_channels = p_scale.shape[1]
            # Reshape from (H*W, C_channels) to (H*W * num_anchors_per_location, features_per_detection)
            if C_channels % features_per_detection != 0:
                print(f"Warning: Unexpected number of channels {C_channels} for detection head. Expected multiple of {features_per_detection}.")
                # Fallback to a single detection if unsure, or raise error.
                if C_channels == features_per_detection:
                    preds_i_reshaped = preds_i # Already in correct shape (H*W, 6)
                else:
                    raise ValueError(f"Detection head output channels {C_channels} not a multiple of {features_per_detection}.")
            else:
                preds_i_reshaped = preds_i.reshape(-1, features_per_detection)

            image_decoded_preds.append(preds_i_reshaped)

        # Concatenate all decoded predictions for the current image across all scales
        all_decoded_preds_for_image = torch.cat(image_decoded_preds, dim=0) # Shape: (Total_predictions, features_per_detection)

        # Filter by confidence
        confidences = all_decoded_preds_for_image[:, 4].sigmoid()
        high_conf_preds_indices = (confidences > conf_thresh).nonzero(as_tuple=True)[0]

        if high_conf_preds_indices.numel() == 0:
            predicted_counts.append(0)
            continue

        high_conf_preds = all_decoded_preds_for_image[high_conf_preds_indices]

        # Apply NMS
        boxes_xywh = high_conf_preds[:, :4]
        boxes_x1y1x2y2 = torch.empty_like(boxes_xywh)
        boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 # x1
        boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 # y1
        boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 # x2
        boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 # y2

        scores = confidences[high_conf_preds_indices] # Use scores corresponding to high_conf_preds

        keep_indices = ops.nms(boxes_x1y1x2y2, scores, nms_iou_thresh)

        predicted_counts.append(len(keep_indices))

    if not predicted_counts or len(ground_truth_counts) == 0:
        return {'MAE': 0.0, 'RMSE': 0.0}

    predicted_counts_np = np.array(predicted_counts)
    ground_truth_counts_np = np.array(ground_truth_counts)

    rmse = np.sqrt(mean_squared_error(ground_truth_counts_np, predicted_counts_np))
    mae = mean_absolute_error(ground_truth_counts_np, predicted_counts_np)

    return {'MAE': mae, 'RMSE': rmse}
