论文中使用到的数据集的下载链接：

    百度网盘地址：https://pan.baidu.com/s/11CErw77QGsR4ggeV6_2a2A?pwd=bw2k
    提取码：bw2k


我们提供了coco格式的RS-AOD数据集和用于ultralytics目标检测框架训练的YOLO格式，还提供了ShipRSImageNet数据集的用于ultralytics目标检测框架训练的YOLO格式。

其中ShipRSImageNet数据集做了一些改动，具体改动如下：

    只保留了Ship一个类别
    所有的图像都resize到了640x640



这么做是为了便于APSM的通用幅度谱提取以及APSM训练和验证时要求的图像输入size和提取得到的通用幅度谱size相等或相近，避免大的偏差。
