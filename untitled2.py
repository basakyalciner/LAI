# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:07:19 2023

@author: bskylcnr
"""

import cv2
import numpy as np

def yolov4_test_et(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (416, 416))

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    labels = ["lettuce"]
    
    colors = ["0,0,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

    model = cv2.dnn.readNetFromDarknet("yolov4_ders.cfg", "yolov4_ders_last.weights")

    layers = model.getLayerNames()
    output_layer = [layer for layer in model.getLayerNames() if "yolo" in layer]

    model.setInput(frame_blob)

    detection_layers = model.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.20:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

    
        max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
        
        for max_id in max_ids:
            max_class_id = int(max_id)
            predicted_id = ids_list[max_class_id]
            
            box = boxes_list[max_class_id]
            start_x = box[0]
            start_y = box[1]
            box_width = box[2]
            box_height = box[3]
            confidence = confidences_list[max_class_id]
            end_x = start_x + box_width
            end_y = start_y + box_height

            # Print rectangle values for enemy
            ref_x=160
            ref_y=190
            lettuce_x=2*(end_x-start_x)
            lettuce_y=2*(end_y-start_y)
            x=(lettuce_x*17)//ref_x
            y=(lettuce_y*25)//ref_y
            dimension=str(x)+","+str(y)+"cm"
            
            print(ref_x,lettuce_x)
            print(ref_y,lettuce_y)
            
            rect_x = start_x + box_width // 2
            rect_y = start_y + box_height // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            plus_size = 10
            plus_x = frame_center_x
            plus_y = frame_center_y

            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
            label = "{}: {:.2f}%".format(label, confidence * 100)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
            cv2.rectangle(frame, (start_x - 1, start_y), (end_x + 1, start_y - 30), box_color, -1)
            cv2.putText(frame, dimension, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            

            return frame

if __name__ == "__main__":
    resim_yolu = "RGB_198.jpg"  # Test etmek istediğiniz resmin dosya yolu
    image = cv2.imread(resim_yolu)
    result_frame = yolov4_test_et(image)
    cv2.imshow("Detector", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
