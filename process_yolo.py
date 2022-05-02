import os
import sys
import argparse
import numpy as np
import json
import scipy.optimize
import cv2
import matplotlib.pyplot as plt


def process_test_file(test_file):
	"""
	This method creates a dictionary containing the name of 
	each test file as keys annd the corresponding classification 
	and bounding box labels for each file as values

	Arguments:
		test_file (str): 
				Represents the path to the txt file containing the test set

	Return:
		processed_files (Dict): 
				Contains file names as keys and labels as values 
	"""
	ref_files = []
	with open(test_file, 'r') as tf:
		lines = tf.readlines()
		for line in lines:
			f = line.split('/')[-1]
			f = f.split('.')[0]
			ref_file = f + '.txt'
			ref_files.append(ref_file)
	tf.close()

	processed_files = {}
	for ref_file in ref_files:
		with open(ref_file, 'r') as f:
			processed_file = []
			lines = f.readlines()
			for line in lines:
				processed_line = [float(i) for i in line.split(' ')]
				processed_line[0] = int(processed_line[0])
				processed_line[1] = processed_line[1] - (processed_line[3]/2)
				processed_line[2] = processed_line[2] - (processed_line[4]/2)
				processed_line[3] = processed_line[3] + processed_line[1]
				processed_line[4] = processed_line[4] + processed_line[2]
				processed_file.append(processed_line)
			processed_file = np.array(processed_file)
		f.close()
		processed_files[ref_file.split('.')[0]] = processed_file

	return processed_files


def process_preds(pred_values):
	"""
	This method processes the prediction boxes so that they are formatted correctly

	Arguments:
		pred_values (Dict): 
				Contains class id, class name, box coordinates (x,y,width,height), 
				class_confidence for all boxes in one image

	Return:
		pred_boxes (np.array): 
				array of shape (num boxes in image, label and 4 coordinates) where box is 
				defined by top left corner and bottom left corner coordinates
	"""
	pred_boxes = []
	for box in pred_values:
		label = box['class_id']
		width = box['relative_coordinates']['width']
		height = box['relative_coordinates']['height']
		x0 = box['relative_coordinates']['center_x'] - (width/2)
		y0 = box['relative_coordinates']['center_y'] - (height/2)
		x1 = x0 + width
		y1 = y0 + height
		pred_boxes.append([label, x0, y0, x1, y1])
	pred_boxes = np.array(pred_boxes)

	return pred_boxes


def get_matches(pred_boxes, label_boxes, args, iou_thresh):
	"""
	This method calculates matches between prediction boxes and bounding boxes. Based on these
	matches, the intersection over union is calculated via the specified method. From here F1 and AP
	scores are calculated based on a list of iou thresholds.

	Arguments:
		pred_boxes (List): 
				List of lists where each sublist defines a prediction box for a given image. 
				Prediction box is defined by top left corner and bottom right corner coordinates

		label_boxes (List): 
				List of lists where each sublist defines a labeled box for a given image. 
				Label box is defined by top left corner and bottom right corner coordinates

		args (object):
				parsed arguments (args.metric and args.verbose may be used in this method)

		iou_thresh (List):
				List of threshold values to be tested to calculate AP and F1

	Returns:
		average_precision (Float):
				Calculated using precision and recall based on list of iou thresholds

		F1 (Float):
				Calculated as mean of precision and recall for each iou threshold


	
	"""
	num_preds = pred_boxes.shape[0]
	num_labels = label_boxes.shape[0]
	iou_matrix = np.zeros((num_labels, num_preds))
	label_matrix = np.zeros((num_labels, num_preds))
	for pred_index, pred_box in enumerate(pred_boxes):
		for label_index, label_box in enumerate(label_boxes):
			if args.metric == 'iou':
				iou_matrix[label_index, pred_index] = IoU(pred_box[1:], label_box[1:])
				label_matrix[label_index, pred_index] = int(pred_box[0]==label_box[0])
			elif args.metric == 'giou':
				iou_matrix[label_index, pred_index] = GIoU(pred_box[1:], label_box[1:])
				label_matrix[label_index, pred_index] = int(pred_box[0]==label_box[0])
			elif args.metric == 'diou':
				iou_matrix[label_index, pred_index] = DIoU(pred_box[1:], label_box[1:])
				label_matrix[label_index, pred_index] = int(pred_box[0]==label_box[0])
			elif args.metric == 'ciou':
				iou_matrix[label_index, pred_index] = CIoU(pred_box[1:], label_box[1:])
				label_matrix[label_index, pred_index] = int(pred_box[0]==label_box[0])
			else:
				return False

	if num_preds > num_labels:
		margin = num_preds - num_labels
		iou_matrix = np.concatenate((iou_matrix, np.zeros((margin, num_preds))), axis=0)
		label_matrix = np.concatenate((label_matrix, np.zeros((margin, num_preds))), axis=0)

	if num_labels > num_preds:
		margin = num_labels - num_preds
		iou_matrix = np.concatenate((iou_matrix, np.zeros((num_labels, margin))), axis=1)
		label_matrix = np.concatenate((label_matrix, np.zeros((margin, num_preds))), axis=0)

	label_indexes, pred_indexes = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

	if(not label_indexes.size) or (not pred_indexes.size):
		ious = np.array([])
	else:
		ious = iou_matrix[label_indexes, pred_indexes]

	sel_pred = pred_indexes < num_preds
	real_preds_indexes = pred_indexes[sel_pred] 
	real_labels_indexes = label_indexes[sel_pred]
	real_ious = iou_matrix[real_labels_indexes, real_preds_indexes]
	valids = []
	invalids = []
	misslabeled = []
	for thresh in iou_thresh:
		valids.append((real_ious > thresh))
		invalids.append(valids == 0)
		misslabeled.append(label_matrix[real_labels_indexes, real_preds_indexes] == 0)
		

	precision = []
	recall = []
	for i in range(len(valids)):
		true_positives = np.sum(label_matrix[real_labels_indexes, real_preds_indexes] * valids[i])
		if num_labels > num_preds:
			false_negatives = num_labels - num_preds
		else:
			false_negatives = 0
		false_positives = np.sum(np.bitwise_xor(invalids[i], misslabeled[i]))
		if true_positives > 0:
			p = true_positives/(true_positives+false_positives)
			r = true_positives/(true_positives+false_negatives)
		else:
			p = 0
			r = 0
		precision.append(p)
		recall.append(r)
	precision.append(0)
	recall.append(0)
	precision = np.array(precision)
	recall = np.array(recall)
	average_precision = np.sum((recall[:-1] - recall[1:]) * precision[:-1])
	f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])

	if args.verbose:
		print('thresholds: ', iou_thresh)
		print('precision: ', precision[:-1])
		print('recall: ', recall[:-1])
		print('F1: ', f1)
		print('\n')
	avg_f1 = np.sum(f1)/len(f1)


	return average_precision, avg_f1



def IoU(pred_box, label_box, return_inter = False):
	"""
	This method calculates classic Intersection over Union

	Arguments:
		pred_box (List):
				Represents prediction box defined by upper left corner
				and lower right corner coordinates

		label_box (List):
				Represents label box defined by upper left corner
				and lower right corner coordinates

		return_inter (Bool):
				Optional Bool used to return intersection area
	
	Returns:
		iou (Float):
				Represents calculated Intersection over Union

		inter_area (Float):
				Represents calculated intersection area

	"""
	inter_x0 = max([pred_box[0], label_box[0]])
	inter_y0 = max([pred_box[1], label_box[1]])
	inter_x1 = min([pred_box[2], label_box[2]])
	inter_y1 = min([pred_box[3], label_box[3]])

	a0 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
	a1 = (label_box[2] - label_box[0]) * (label_box[3] - label_box[1])

	inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
	iou = inter_area / (a0 + a1 - inter_area)
	if return_inter:
		return iou, inter_area
	else:
		return iou


def GIoU(pred_box, label_box):
	"""
	This method calculates Generalized Intersection over Union

	Arguments:
		pred_box (List):
				Represents prediction box defined by upper left corner
				and lower right corner coordinates

		label_box (List):
				Represents label box defined by upper left corner
				and lower right corner coordinates
	
	Returns:
		giou (Float):
				Represents calculated Generalized Intersection over Union
	"""
	iou, intersection = IoU(pred_box, label_box, True)
	total_area_closure = (max(pred_box[0], pred_box[2], label_box[0], pred_box[2]) - \
						  min(pred_box[0], pred_box[2], label_box[0], pred_box[2])) * \
						 (max(pred_box[1], pred_box[3], label_box[1], pred_box[3]) - \
						  min(pred_box[1], pred_box[3], label_box[1], pred_box[3]))
	sum_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) + \
				(label_box[2] - label_box[0]) * (label_box[3] - label_box[1]) - \
				intersection
	giou = iou - ((total_area_closure - sum_area) / total_area_closure)

	return giou


def DIoU(pred_box, label_box):
	"""
	This method calculates Distance Intersection over Union

	Arguments:
		pred_box (List):
				Represents prediction box defined by upper left corner
				and lower right corner coordinates

		label_box (List):
				Represents label box defined by upper left corner
				and lower right corner coordinates
	
	Returns:
		giou (Float):
				Represents calculated Distance Intersection over Union
	"""
	iou = IoU(pred_box, label_box)
	c1 = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
	c2 = ((label_box[0] + label_box[2]) / 2, (label_box[1] + label_box[3]) / 2)
	center_dist = (c2[0] - c1[0])**2 + (c2[1] - c1[1])**2

	closure1 = (min(pred_box[0], pred_box[2], label_box[0], label_box[2]), 
				min(pred_box[1], pred_box[3], label_box[1], label_box[3]))
	closure2 = (max(pred_box[0], pred_box[2], label_box[0], label_box[2]), 
				max(pred_box[1], pred_box[3], label_box[1], label_box[3]))
	corner_dist = (closure2[0] - closure1[0])**2 + (closure2[1] - closure1[1])**2

	diou = iou - (center_dist/corner_dist)

	return diou



def CIoU(pred_box, label_box):
	"""
	This method calculates Combined Intersection over Union

	Arguments:
		pred_box (List):
				Represents prediction box defined by upper left corner
				and lower right corner coordinates

		label_box (List):
				Represents label box defined by upper left corner
				and lower right corner coordinates
	
	Returns:
		giou (Float):
				Represents calculated Combined Intersection over Union
	"""
	iou = IoU(pred_box, label_box)
	c1 = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
	c2 = ((label_box[0] + label_box[2]) / 2, (label_box[1] + label_box[3]) / 2)
	center_dist = (c2[0] - c1[0])**2 + (c2[1] - c1[1])**2

	closure1 = (min(pred_box[0], pred_box[2], label_box[0], label_box[2]), 
				min(pred_box[1], pred_box[3], label_box[1], label_box[3]))
	closure2 = (max(pred_box[0], pred_box[2], label_box[0], label_box[2]), 
				max(pred_box[1], pred_box[3], label_box[1], label_box[3]))
	corner_dist = (closure2[0] - closure1[0])**2 + (closure2[1] - closure1[1])**2

	w1, h1 = np.abs(pred_box[2]-pred_box[0]), np.abs(pred_box[3]-pred_box[1])
	w2, h2 = np.abs(label_box[2]-label_box[0]), np.abs(label_box[3]-label_box[1])

	v = 4*(np.arctan(w1/h1) - np.arctan(w2/h2)) ** 2 / (np.pi ** 2)
	a = v / (1 - iou + v)
	ciou = iou - (center_dist / corner_dist) - a * v

	return ciou


	
def calc_map(json_file, labels, args):
	"""
	This method is used to calculate the mean average precision for all samples

	Arguments:
		json_file (str):
				File path to darknet output json

		labels (Dict):
				Contains mapping between file and labels for ground truth bounding boxes in the image

		args (object):
				parsed arguments (args.verbose and args.hide_images may be used in this method)

	Returns:
		mean_avg_precision (float):
				Represents the mean average precision calculated for all images passed in

		precisions_dict (Dict):
				mapping from file to bounding box and metric values
	"""
	with open(json_file, 'r') as f:
		pred_data = json.load(f)
	f.close()

	avg_accuracy = 0
	avg_precisions = []
	f1_values = []
	precisions_dict = {}
	for pred in pred_data:
		pred_file = (pred['filename'].split('/')[-1]).split('.')[0]
		pred_values = pred['objects']
		pred_boxes = process_preds(pred_values)
		thresh_list = [float(i) for i in args.thresh_range.split(':')]
		thresholds = np.arange(start=thresh_list[0], stop=thresh_list[1], step=thresh_list[2])
		if args.verbose:
			print('File: ', pred_file)
		avg_precision, f1 = get_matches(pred_boxes, labels[pred_file], 
										args, thresholds)
		avg_precisions.append(avg_precision)
		f1_values.append(f1)
		precisions_dict[pred_file] = [pred_boxes, labels[pred_file], 
									  avg_precisions[-1], f1_values[-1]]
	mean_avg_precision = sum(avg_precisions)/len(avg_precisions)

	if not args.hide_images:
		return mean_avg_precision, precisions_dict
	else:
		return mean_avg_precision, None

def get_class_colors(encoding_file):
	"""
	This method reads the colors txt file and returns a mapping between
	classes and colors as (r,g,b).
	
	Arguments:
		encoding_file (str):
				Path to color encoding file

	Returns:
		mapping for each class to corresponding color represented as (r, g, b)
	
	"""
	encoding_dict = {}
	with open(encoding_file, 'r') as f:
		for line in f.readlines():
			try:
				object_class = int(line.split(' ')[0])
			except ValueError:
				object_class = str(line.split(' ')[0])
			h = str(line.split('#')[1])
			rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
			encoding_dict[object_class] = rgb
	f.close()
	return encoding_dict

def show_predictions(precision_dict, args):
	"""
	This method creates and saves the prediction images

	Arguments:
		precision_dict (Dict):
				mapping between file name and pred/label boxes, average precision, and F1
				corresponding to the file
		args (Object):
				parsed arguments (args.images_path, args.hide_labels, args.black_labels, args.hide_preds, 
				args.use_f1, args.hide_count, and args.image_out may be used in this method)


	"""
	color_encodings = get_class_colors('class_colors.txt')
	for key, value in precision_dict.items():
		if args.images_path:
			img = cv2.imread(imgs_path + key + '.jpg')
		else:
			img = cv2.imread(key + '.jpg')

		img = cv2.resize(img, (416,416))
		IMG_Y = img.shape[0]
		IMG_X = img.shape[1]

		for i in range(max(len(value[0]), len(value[1]))):
			if i < len(value[1]):
				if not args.hide_labels:
					if args.black_labels:
						cv2.rectangle(img, 
                                                              (int(IMG_X*value[1][i][1]), int(IMG_Y*value[1][i][2])),
                                                              (int(IMG_X*value[1][i][3]),int(IMG_Y*value[1][i][4])),
                                                              color_encodings['label'])
					else:
						cv2.rectangle(img,
                                                              (int(IMG_X*value[1][i][1]),int(IMG_Y*value[1][i][2])),
                                                              (int(IMG_X*value[1][i][3]),int(IMG_Y*value[1][i][4])),
                                                              color_encodings[value[1][i][0]])
			if i < len(value[0]):
				if not args.hide_preds:
					cv2.rectangle(img,
                                                      (int(IMG_X*value[0][i][1]),int(IMG_Y*value[0][i][2])),
                                                      (int(IMG_X*value[0][i][3]),int(IMG_Y*value[0][i][4])),
                                                      color_encodings[value[0][i][0]])

		legend_label = 'Color legend: '
		cv2.putText(img, text=legend_label, org=(int(IMG_X*.75),int(IMG_Y*0.06)),
            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
            	thickness=1, lineType=cv2.LINE_AA)

		label_string = ''
		colors = []
		for i, c in color_encodings.items():
			label_string += f'{i} -> \n'
			colors.append(c)
		y0 = .085
		dy = .025
		for i, line in enumerate(label_string.split('\n')[:-1]):
			y = y0 + (i * dy)
			if 'label' in line:
				cv2.putText(img, text=line+'|||||||', org=(int(IMG_X*.75),int(IMG_Y*y)),
	            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
	            	thickness=1, lineType=cv2.LINE_AA)
			else:
				cv2.putText(img, text=line, org=(int(IMG_X*.75),int(IMG_Y*y)),
	            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
	            	thickness=1, lineType=cv2.LINE_AA)
				cv2.putText(img, text='|||||||', org=(int(IMG_X*.82),int(IMG_Y*y)),
	            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=colors[i],
	            	thickness=1, lineType=cv2.LINE_AA)

		if args.use_f1:
			cv2.putText(img, text=f'F1: {round(value[3], 2)}', org=(int(IMG_X*.75),int(IMG_Y*.94)),
            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
            	thickness=1, lineType=cv2.LINE_AA)
		else:
			cv2.putText(img, text=f'AP: {round(value[2], 2)}', org=(int(IMG_X*.75),int(IMG_Y*.94)),
	            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
	            	thickness=1, lineType=cv2.LINE_AA)

		if not args.hide_count:
			num_preds = len(value[0])
			num_labels = len(value[1])
			cv2.putText(img, text=f'Predictions: {num_preds}', org=(int(IMG_X*.75),int(IMG_Y*.96)),
            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
            	thickness=1, lineType=cv2.LINE_AA)
			cv2.putText(img, text=f'Labels: {num_labels}', org=(int(IMG_X*.75),int(IMG_Y*.98)),
            	fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=(0,0,0),
            	thickness=1, lineType=cv2.LINE_AA)
		cv2.imwrite(args.image_out+'/'+key+'.jpg', img)


if __name__ == '__main__':
	parser  = argparse.ArgumentParser(usage='Calculate evaluation metrics for yolo object detector')
	parser.add_argument('label_files', help='Txt file containing paths to labelled images (such as test.txt used for darknet)')
	parser.add_argument('prediction_file', help='JSON file containing darknet output predictions')
	parser.add_argument('-m', '--metric', default='iou', help='Specifies iou calculation method (iou, giou, diou, ciou)')
	parser.add_argument('-f', '--use_f1', action='store_true', help='Toggle using f1 value instead of AP (default) in prediction image')
	parser.add_argument('-bl', '--black_labels', action='store_true', help='Toggle whether label boxes should be color given by class or black (default)')
	parser.add_argument('-ec', '--color_correct', action='store_true', help='Toggle whether to encode prediction box color as green and red to represent correct or incorrect class prediciton')
	parser.add_argument('-hi', '--hide_images', action='store_true', help='Toggle creating images with predicted bounding boxes')
	parser.add_argument('-hp', '--hide_preds', action='store_true', help='Toggle whether to show prediction boxes')
	parser.add_argument('-l', '--hide_labels', action='store_true', help='Toggle whether to show or hide labels in predicition images')
	parser.add_argument('-hl', '--hide_legend', action='store_true', help='Toggle whether to show class and color legend in prediction images')
	parser.add_argument('-v', '--verbose', action='store_true', help='Toggle whether to print file and metrics as they are being calculated')
	parser.add_argument('-o', '--image_out', default='new_images', help='Specifies custom path for prediction images to be stored in')
	parser.add_argument('-sc', '--hide_count', action='store_true', help='Toggle whether to show predicted pill count')
	parser.add_argument('-ip', '--images_path', default=None, help='Path to all images refrenced in labels file')
	parser.add_argument('-tr', '--thresh_range', default='.3:.8:.05', help='Sets the range and step for iou thresholds format: .3:.8:.05 -> min bound: 0.3, max bound: 0.8, step: 0.05')
	args = parser.parse_args()

	processed_test_file = process_test_file(args.label_files)
	mean_avg_precision, precision_dict = calc_map(args.prediction_file, processed_test_file, args)
	print('MAP: ', round(mean_avg_precision, 3))
	if precision_dict:
		show_predictions(precision_dict, args)

	
















