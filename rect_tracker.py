# import the necessary packages
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import numpy as np

ALPHA = 0.5
inf = 10000000

class RectTracker():
	def __init__(self, max_disappeared=10):
		self.next_object_id = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.velocities = OrderedDict()

		self.max_disappeared = max_disappeared
		self.frame = 0

	def register(self, rect):
		self.objects[self.next_object_id] = rect
		self.disappeared[self.next_object_id] = 0
		self.velocities[self.next_object_id] = 50
		self.next_object_id += 1

	def deregister(self, object_id):
		del self.objects[object_id]
		del self.disappeared[object_id]

	def update(self, input_rects):
		if len(input_rects) == 0:
			for object_id in list(self.disappeared.keys()):
				self.disappeared[object_id] += 1

				if self.disappeared[object_id] > self.max_disappeared:
					self.deregister(object_id)

			return self.objects

		input_rects = np.array(input_rects)

		if len(self.objects) == 0:
			for i in range(0, len(input_rects)):
				self.register(input_rects[i])
		else:
			object_ids = list(self.objects.keys())
			object_rects = list(self.objects.values())
			object_velocities = [(np.linalg.norm(self.velocities[o_id]) + 10) * (self.disappeared[o_id] + 1) for o_id in object_ids]

			velocities = np.repeat(np.array(object_velocities)[np.newaxis, :], len(input_rects), axis=0)

			# D = np.zeros((len(object_rects), len(input_rects)), dtype=np.float32)
			D = dist.cdist(np.array(object_rects)[:, :4], input_rects[:, :4])

			for i in range(len(object_rects)):
				x1_o, y1_o, x2_o, y2_o = object_rects[i][:4]
				for j in range(len(input_rects)):
					x1_i, y1_i, x2_i, y2_i = input_rects[j][:4]
					inter 	= max(x1_o, x1_i), max(y1_o, y1_i), min(x2_o, x2_i), min(y2_o, y2_i)
					inter_area = max(inter[2] - inter[0], 0) * max(inter[3] - inter[1], 0)
					union = ((x2_o - x1_o) * (y2_o - y1_o)) + ((x2_i - x1_i) * (y2_i - y1_i)) - inter_area
					if inter_area == 0:
						D[i][j] = inf	
					# else:
					# 	D[i][j] = 1 - inter_area / union

			# row_ind, col_ind = linear_sum_assignment(D)
			row_ind, col_ind = [], []

			while True:
				idx = np.unravel_index(np.argmin(D, axis=None), D.shape)
				min_val = D[idx]
				if min_val == inf:
					break

				D[idx] = inf

				if idx[0] in row_ind or idx[1] in col_ind:
					continue

				D[row_ind, :] = inf
				D[:, col_ind] = inf
				row_ind.append(idx[0])
				col_ind.append(idx[1])

			unused_rows = set(range(0, D.shape[0])).difference(set(row_ind))
			unused_cols = set(range(0, D.shape[1])).difference(set(col_ind))

			for (row, col) in zip(row_ind, col_ind):
				object_id = object_ids[row]

				x1_o, y1_o, x2_o, y2_o = self.objects[object_id][:4]
				x1_i, y1_i, x2_i, y2_i = input_rects[col][:4]
				# if D[row, col] == inf:
				# 	unused_rows.add(row)
				# 	unused_cols.add(col)
				# else:
				self.objects[object_id] = input_rects[col]
				self.disappeared[object_id] = 0
				self.velocities[object_id] = (1-ALPHA) * self.velocities[object_id] + ALPHA * np.linalg.norm([
					((x1_i + x2_i) // 2) - ((x1_o + x2_o) // 2),
					((y1_i + y2_i) // 2) - ((y1_o + y2_o) // 2),
				])

			for row in unused_rows:
				object_id = object_ids[row]
				self.disappeared[object_id] += 1

				if self.disappeared[object_id] > self.max_disappeared:
					self.deregister(object_id)

			for col in unused_cols:
				self.register(input_rects[col])

		self.frame += 1
		return self.objects