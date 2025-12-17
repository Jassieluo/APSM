from ultralytics import YOLO, RTDETR

model_config = "apsm_works/model_config/apsm_yolo11l.yaml"
# model_config = "apsm_works/model_config/apsm_yolov5l.yaml"
# model_config = "apsm_works/model_config/apsm_rtdetr-l.yaml"
# model_config = "yolo11l.yaml"
# model_config = "yolov5l.yaml"
# model_config = "rtdetr-l.yaml"

model = YOLO(model_config)
# model = RTDETR(model_config)

dataset = "Datasets/RS-AOD-YOLO/RS-AOD.yaml"
# dataset = "Datasets/ShipRSImageNet-YOLO/ShipRSImageNet-YOLO.yaml"

if __name__ == '__main__':
    model.train(data=dataset, epochs=300, imgsz=640, device='cuda', lr0=0.01, amp=False, name="apsm_yolo11l", batch=16)