from ultralytics import YOLO, RTDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def convert_coco_ids_to_str(coco_obj: COCO) -> COCO:
    for img_info in coco_obj.dataset['images']:
        img_info['id'] = str(img_info['id'])

    if 'annotations' in coco_obj.dataset:
        for ann_info in coco_obj.dataset['annotations']:
            ann_info['image_id'] = str(ann_info['image_id'])

    coco_obj.createIndex()
    return coco_obj


def coco_val(gt_json_path, pred_json_path, title=""):
    print("-" * 50)
    print(title + " COCO Eval Start")

    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)

    print("Converting ground truth and prediction IDs to string for pycocotools compatibility...")
    coco_gt = convert_coco_ids_to_str(coco_gt)
    coco_pred = convert_coco_ids_to_str(coco_pred)
    print("Conversion complete.")

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(title + " COCO Eval End")
    print("-" * 50)

# model_path = "runs/detect/fda_yolo11l"
# model_path = "runs/detect/yolo11l"
# model_path = "runs/detect/fda_yolov5l"
# model_path = "runs/detect/yolov5l"
# model_path = "runs/detect/fda_rtdetr-l"
model_path = "runs/detect/rtdetr-l"

# model_path = "runs/detect_plane/fda_yolo11l"
# model_path = "detect_plane/yolo11l"
# model_path = "detect_plane/fda_yolov5l"
# model_path = "detect_plane/yolov5l"
# model_path = "runs/detect_plane/fda_rtdetr-l"
# model_path = "runs/detect_plane/rtdetr-l"

# model_path = "runs/detect_ship/fda_yolo11l"
# model_path = "runs/detect_ship/yolo11l"
# model_path = "runs/detect_ship/fda_yolov5l"
# model_path = "runs/detect_ship/yolov5l"
# model_path = "runs/detect_ship/fda_rtdetr-l"
# model_path = "runs/detect_ship/rtdetr-l"

# model = YOLO(model_path + "/weights/best.pt")
model = RTDETR(model_path+"/weights/best.pt")

"""
in ultralytics/utils/ops.py  line 128 -> 131 (Modifications are required during validation)
        # gain = ratio_pad[0][0] #For YOLO
        # pad_x, pad_y = ratio_pad[1]
        gain = ratio_pad[0] #For RT-DETR
        pad_x, pad_y = ratio_pad
"""

orin_data_path = "Datasets/RS-AOD-YOLO/"
noise_data_path = "RS-AOD-YOLO-NOISE/" # (You need to first use add_noise.py to build a new noise dataset for validation)
orin_data = orin_data_path + "RS-AOD.yaml"
noise_data = noise_data_path + "RS-AOD.yaml"
orin_json_name = 'instances_val2017.json'

# orin_data_path = "Datasets/ShipRSImageNet-YOLO/"
# noise_data_path = "Datasets/ShipRSImageNet-YOLO-NOISE/" # (You need to first use add_noise.py to build a new noise dataset for validation)
# orin_data = orin_data_path + "ShipRSImageNet-YOLO.yaml"
# noise_data = noise_data_path + "ShipRSImageNet-YOLO.yaml"
# orin_json_name = 'val_annotations.json'

if __name__ == '__main__':
    model.val(data=orin_data, save_json=True, iou=0.5, half=False, project=model_path + "/val_orin", device="cuda:0")
    model.val(data=noise_data, save_json=True, iou=0.5, half=False, project=model_path + "/val_noise", device="cuda:0")

    orin_gt_json_path = orin_data_path + orin_json_name
    orin_pred_json_path = model_path + "/val_orin/val/predictions.json"

    # check_coco_json_ids(orin_gt_json_path, orin_pred_json_path)

    coco_val(orin_gt_json_path, orin_pred_json_path, "Orin")

    noise_gt_json_path = orin_data_path + orin_json_name
    noise_pred_json_path = model_path + "/val_noise/val/predictions.json"
    coco_val(noise_gt_json_path, noise_pred_json_path, "Noise")