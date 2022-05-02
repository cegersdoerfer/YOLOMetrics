# YOLOMetrics
Calculate mean average precision (mAP) values for multi-object detection models such as YOLO.

This code was created with intetion to be used for Darknet YoloV3 implemented here: https://github.com/AlexeyAB/darknet. With some edits it could easily be adapted to other formats.

# How to run
```
python calculate_map.py <"test_images_file.txt"> <"Predictions_file.json">
```
In the above line of code, replace the two file names with your test and prediction files.

Format for test image txt file is as follows:
```
PATH/TO/TEST/IMAGE_1.jpg
PATH/TO/TEST/IMAGE_2.jpg
PATH/TO/TEST/IMAGE_3.jpg
...
```
Format for predictions json file is as follows:
```
[
 {
  "frame_id": int,
  "filename": path_to_img_file,
  "objects": [
              {
               "class_id": int class id, 
               "name": str class name, 
               "relative_coordinates": {
                                        "center_x": Float between 0 and 1, 
                                        "center_y": Float between 0 and 1, 
                                        "width": Float between 0 and 1, 
                                        "height": Float between 0 and 1
                                       }, 
               "confidence": Float between 0 and 1
              },
              {
               "class_id": int class id, 
               "name": str class name, 
               "relative_coordinates": {
                                        "center_x": Float between 0 and 1, 
                                        "center_y": Float between 0 and 1, 
                                        "width": Float between 0 and 1, 
                                        "height": Float between 0 and 1
                                       }, 
               "confidence": Float between 0 and 1
             }
            ]
 },
...
]
```
# Argument options
```
python calculate_map.py test.txt out.json -h 
```
```
positional arguments:
  label_files           Txt file containing paths to labelled images (such as
                        test.txt used for darknet)
  prediction_file       JSON file containing darknet output predictions

optional arguments:
  -h, --help            show this help message and exit
  -m METRIC, --metric METRIC
                        Specifies iou calculation method (iou, giou, diou,
                        ciou)
  -f, --use_f1          Toggle using f1 value instead of AP (default) in
                        prediction image
  -bl, --black_labels   Toggle whether label boxes should be color given by
                        class or black (default)
  -ec, --color_correct  Toggle whether to encode prediction box color as green
                        and red to represent correct or incorrect class
                        prediciton
  -hi, --hide_images    Toggle creating images with predicted bounding boxes
  -hp, --hide_preds     Toggle whether to show prediction boxes
  -l, --hide_labels     Toggle whether to show or hide labels in predicition
                        images
  -hl, --hide_legend    Toggle whether to show class and color legend in
                        prediction images
  -v, --verbose         Toggle whether to print file and metrics as they are
                        being calculated
  -o IMAGE_OUT, --image_out IMAGE_OUT
                        Specifies custom path for prediction images to be
                        stored in
  -sc, --hide_count     Toggle whether to show predicted pill count
  -ip IMAGES_PATH, --images_path IMAGES_PATH
                        Path to all images refrenced in labels file
  -tr THRESH_RANGE, --thresh_range THRESH_RANGE
                        Sets the range and step for iou thresholds format:
                        .3:.8:.05 -> min bound: 0.3, max bound: 0.8, step:
                        0.05
```

# Example Output
Output is from a pill detection model with 5 pill classes

Input:
```
python calculate_map.py test.txt out.json -o pred_imgs  -m giou --black_labels
```
Output:
```
MAP:  0.948
```
Example from pred_imgs folder after above run:

![20220326_110758](https://user-images.githubusercontent.com/29511758/166291105-f4bb48b0-fc81-489a-b316-9223565c13f6.jpg)

Input:
```
python calculate_map.py test.txt out.json -o pred_imgs  -m diou --use_f1
```
Output:
```
MAP:  0.952
```
Example from pred_imgs folder after above run:

![20220326_113330](https://user-images.githubusercontent.com/29511758/166292242-bfa331c2-3b7b-434b-80db-e264e73d916b.jpg)





