import os
import cv2
import glob
import random
import argparse
random.seed(1)
def intersection(box1, box2):
	b1x1 = box1[0] - box1[2] / 2
	b1y1 = box1[1] - box1[3] / 2
	b1x2 = box1[0] + box1[2] / 2
	b1y2 = box1[1] + box1[3] / 2

	b2x1 = box2[0] - box2[2] / 2
	b2y1 = box2[1] - box2[3] / 2
	b2x2 = box2[0] + box2[2] / 2
	b2y2 = box2[1] + box2[3] / 2

	union = min(b1x1, b2x1), min(b1y1, b2y1), max(b1x2, b2x2), max(b1y2, b2y2)
	inter = max(b1x1, b2x1), max(b1y1, b2y1), min(b1x2, b2x2), min(b1y2, b2y2)

	if inter[2] < inter[0] or inter[3] < inter[1]:
		return 0
	else:
		I = (inter[2] - inter[0]) * (inter[2] - inter[0])
		return I

def union(box1, box2):
	b1x1 = box1[0] - box1[2] / 2
	b1y1 = box1[1] - box1[3] / 2
	b1x2 = box1[0] + box1[2] / 2
	b1y2 = box1[1] + box1[3] / 2

	b2x1 = box2[0] - box2[2] / 2
	b2y1 = box2[1] - box2[3] / 2
	b2x2 = box2[0] + box2[2] / 2
	b2y2 = box2[1] + box2[3] / 2

	union = min(b1x1, b2x1), min(b1y1, b2y1), max(b1x2, b2x2), max(b1y2, b2y2)
	inter = max(b1x1, b2x1), max(b1y1, b2y1), min(b1x2, b2x2), min(b1y2, b2y2)

	U = (union[2] - union[0]) * (union[3] - union[1])
	return U

if __name__ == "__main__":
	crops_dir = "ozfish/sampled_bg_crops"
	os.makedirs(crops_dir, exist_ok=True)
	with open("ozfish/train_combined.txt") as f:
		img_paths = f.read().splitlines()


	aspect_ratios = [0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 3]
	all_possible_boxes = []

	for x in range(0, 50, 3):
		x /= 50
		for y in range(0, 50, 3):
			y /= 50

			for i in range(2):
				aspect_ratio = random.choice(aspect_ratios)
				w = random.randint(50, 960) / 1920
				h = w * aspect_ratio

				if x + w > 1.:
					x -= ((x + w) - 1) / 2
					w = 1 - x
					if random.random() > 0.2:
						continue
				if y + h > 1.:
					y -= ((y + h) - 1) / 2
					h = 1 - y
					if random.random() > 0.2:
						continue

				if w * h > 0.25 or w * h < ((50*50)/(1920*1080)):
					continue

				_x = x + w / 2
				_y = y + h / 2

				all_possible_boxes.append([_x, _y, w, h])

	for img_path in img_paths:
		print(img_path)
		img = cv2.imread(img_path)
		with open(img_path.replace(".png", ".txt")) as f:
			gt_boxes = list(map(lambda x: list(map(float, x.split())), f.read().splitlines()))

		possible_boxes = all_possible_boxes.copy()

		for gt_box in gt_boxes:
			to_remove = []
			for box in possible_boxes:
				I = intersection(gt_box[1:5], box)
				U = union(gt_box[1:5], box)
				area1 = gt_box[3] * gt_box[4]
				area2 = box[2] * box[3]

				if I / U > 0.1 or I / min(area1, area2) > 0.1:
					to_remove.append(box)
			possible_boxes = [box for box in possible_boxes if box not in to_remove]

		if len(possible_boxes) > 20:
			possible_boxes = random.sample(possible_boxes, 20)

		for i, box in enumerate(possible_boxes):
			x, y, w, h = box
			x1 = max(0, int((x - w/2) * 1920))
			y1 = max(0, int((y - h/2) * 1080))
			x2 = int((x + w/2) * 1920)
			y2 = int((y + h/2) * 1080)

			crop = img[y1:y2, x1:x2]
			# if os.path.basename(img_path) == 'A000001_L.avi.44226.png' and i==8:
			# 	for gt_box in gt_boxes:
			# 		to_remove = []
			# 		I = intersection(gt_box[1:5], box)
			# 		U = union(gt_box[1:5], box)
			# 		area1 = gt_box[3] * gt_box[4]
			# 		area2 = box[2] * box[3]

			# 		print(I / U, I / min(area1, area2))

			cv2.imwrite(os.path.join(crops_dir, os.path.basename(img_path).replace('.png', '-{}.png'.format(i))), crop)