# LAI
Lettuce area index

![lettuce_area_index](https://github.com/basakyalciner/LAI/assets/56589435/f6e794d7-540c-4477-82b2-18ed4bbb9af9)

# Marul Area Index: Automatic Sizing for Lettuce Harvesting using YOLOv4

## Abstract
In this study, we present an automated method for lettuce harvesting in the agricultural industry. We utilized the YOLOv4 model for lettuce detection and calculated the lettuce area index based on the known dimensions of the detected bounding boxes. By comparing the pixel count of the boxes to the pixel count of the lettuce, we determined the lettuce area index in cm units. We established a suitable harvesting range based on the size of the lettuce, identifying which lettuces are ready for harvesting. This AI-powered approach has the potential to significantly impact lettuce harvesting practices.

## Introduction
To increase efficiency and reduce labor costs in the agriculture sector, automated harvesting techniques are being explored. This research proposes a lettuce area index calculation method to identify lettuces ready for harvesting. The YOLOv4 model was employed for lettuce detection, and the lettuce area index was computed from the dimensions of the bounding boxes.

## Lettuce Detection with YOLOv4
YOLOv4 is a deep learning-based object detection model used for lettuce detection. The model was trained on a dataset containing lettuce samples to achieve accurate lettuce detection, as validated on test data.

## Lettuce Area Index Calculation
After lettuce detection, the pixel count for each lettuce bounding box was computed. Using a reference object (e.g., a ruler) in the same image, the dimensions of the lettuce bounding boxes were converted to real-world units (e.g., cm). Based on this information, the area of each lettuce was calculated, and the lettuce area index was derived.

## Sizing for Harvesting
The lettuce area index serves as a measure of the lettuce's growth status. The study determined a suitable range of lettuce sizes for harvesting, assisting farm laborers in identifying lettuces that are ready for harvest.

## Results
The YOLOv4-based lettuce detection and lettuce area index calculation method demonstrate the potential of AI in the agriculture sector. The study's findings indicate that this approach can be an effective tool for lettuce harvesting. Identifying lettuces suitable for harvesting reduces labor costs and improves harvest efficiency.

## Conclusion
This study introduces a YOLOv4-based lettuce detection system and lettuce area index calculation method for lettuce harvesting. The adoption of automated harvesting systems, based on this research, is expected to grow in the agricultural industry, enhancing harvest efficiency and reducing labor costs.

## References
https://universe.roboflow.com/yolo-zgi34/lettuce-detect-qi0eu

