# YOLOMetrics
Calculate mean average precision (mAP) values for multi-object detection models such as YOLO.

This code was created with intetion to be used for Darknet YoloV3 implemented here: https://github.com/AlexeyAB/darknet. With some edits it could easily be adapted to other formats.

# How to run
1. Edit the class_colors file to match the amount of classes in your model and specify the desired color for each.
   
   Ex. My model had 5 classes so the class_colors.txt looks like this:
   ```
   0 #FF0000
   1 #00FF00
   2 #CEFF00
   3 #00FFB0
   4 #00C3FF
   label #000000
   ```
2. Run the calculate_map.py file
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
                 },<img width="447" alt="Screen Shot 2022-05-05 at 1 04 13 AM" src="https://user-images.githubusercontent.com/29511758/166866145-c2632976-c49a-4bba-8da1-2c07a56c2024.png">

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


# IoU algorithms
As shown in the argument options, there are multiple types of iou variations available to choose from. Though the real benefit of the enhanced equations only truly shines when they are applied as loss functions, they are fun to explore and easily implemented so why not put them here. As an additional note, if these are to be used as loss functions they should be modified to maximize rather than minimize if predictions are far apart. This can be done by subtracting from 1. For example, IoU as a metric would become 1-IoU as a loss function.

The following is a brief explanation of each equation:

## IoU
Regular Intersection over Union is the most basic form of IoU algorithm. It simply takes the intersection area of two bounding boxes and divides that by the union area of the bounding boxes.

<img width="500" alt="Screen Shot 2022-05-04 at 4 50 48 PM" src="https://user-images.githubusercontent.com/29511758/166823775-55a1e18d-3622-4ed0-b908-a631b1dda207.png">

## GIoU
Generalized Intersection over Union incorporates the area of the smallest rectangle which encloses both bounding boxes. The calculation breaks down to finding the regular IoU and subtracting the ratio of the smallest closure area excluding union and the smallest closure area.

Here's a great link to further explore this metric: https://giou.stanford.edu/

<img width="500" alt="giou_img" src="https://user-images.githubusercontent.com/29511758/166825904-764919ad-a50b-4796-9cda-55ffa2983812.jpeg">

## DIoU
Distance Intersection over Union is similar to GIoU, as it replaces smallest are enclosure of thwo bounding boxes by distance measures. This is calculated by taking the original IoU and subtracting the ratio of squares of both the distance between the two centers and the two farthest corners.

Here's a link to the paper which introduced this method: https://arxiv.org/abs/1911.08287

<img width="500" alt="Screen Shot 2022-05-05 at 1 03 32 AM" src="https://user-images.githubusercontent.com/29511758/166866096-3e7b5d1a-a48f-430d-ba58-19f40981d855.png">

<img width="300" alt="Screen Shot 2022-05-05 at 1 04 13 AM" src="https://user-images.githubusercontent.com/29511758/166866155-0d07cd69-fd79-47ac-9dbc-6504aad33804.png">

## CIoU
Complete Intersection over Union takes DIoU one step further by incorporating the comparison of bounding box aspect ratio. This is done by calculating DIoU and adding a new term, alpha * v, where alpha is used to give precedence to optimizing IoU first, and v is used to compare aspect ratios of the bounding boxes.

This method is included as part of the paper for DIoU but here is the link again: https://arxiv.org/abs/1911.08287

I could not find any helpful depiction of CIoU but the equations should do the trick:


<img width="300" alt="Screen Shot 2022-05-05 at 1 32 35 AM" src="https://user-images.githubusercontent.com/29511758/166867973-7152635e-78e7-4b6f-96a0-6232c25ab1e7.png">

<img width="300" alt="Screen Shot 2022-05-05 at 1 33 08 AM" src="https://user-images.githubusercontent.com/29511758/166867999-d3aaa7d8-524b-462d-aa3f-545803d939fc.png">

<img width="300" alt="Screen Shot 2022-05-05 at 1 33 33 AM" src="https://user-images.githubusercontent.com/29511758/166868034-06e1dfca-d134-4d4a-a1b7-f85d01b8f7de.png">


