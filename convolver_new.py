import numpy as np
import math
import os
import matplotlib.pyplot as plt
import inspect
import time
from typing import *
import imageio


class InvalidArgumentError(Exception):
	__slots__ = ["message"]

	def __init__(self, message):
		self.message = message

	def __repr__(self):
		return "InvalidArgumentError: " + self.message


def print2d(array, space=3):
	for row in array:
		for item in row:
			print("~" * (space - len(str(item))), end="")
			print(item, end=" ")
		print()


def any_above(iterable, value) -> bool:
	for it in iterable:
		if it > value:
			return True
	return False


def eq(a: Union[bool, int, float, np.float, str, np.int], b: Union[bool, int, float, np.float, str, np.int]):
	return a.__eq__(b)


def gt(a: Union[bool, int, float, np.float, np.int], b: Union[bool, int, float, np.float, np.int]):
	return a.__gt__(b)


def lt(a: Union[bool, int, float, np.float, np.int], b: Union[bool, int, float, np.float, np.int]):
	return a.__lt__(b)


def ge(a: Union[bool, int, float, np.float, np.int], b: Union[bool, int, float, np.float, np.int]):
	return a.__ge__(b)


def le(a: Union[bool, int, float, np.float, np.int], b: Union[bool, int, float, np.float, np.int]):
	return a.__le__(b)


def getsides(array, *value, cond=eq):
	L = len(array)
	b, e = None, None
	# print(array)
	for i, a in enumerate(array):
		if cond(a, *value):
			b = i
			break
	if b is None:
		return b, e, L
	for i, a in reversed(list(enumerate(array))):
		if cond(a, *value):
			e = i
			break
	return b, e, L


def compare(a, b, maximum=1, scaled=True, opacity_mode=False, return_tuple=False, channels: tuple = (1, 1)):  # return_tuple requires inputting lists/ndarrays, channel values of both 1 treats inputs as single numbers, 0 as strings
	# print(a, b, channels, flush=True)
	if channels[0] == 1 and channels[1] == 1:
		try:
			absolute = 1 - abs((b - a)/maximum)
			if scaled:
				return absolute * maximum
			else:
				return absolute
		except ZeroDivisionError or ValueError:
			return None
	elif (channels[0] == 0) or (channels[1] == 0):
		a = float(a)
		b = float(b)
		try:
			absolute = 1 - abs((b - a) / maximum)
			if scaled:
				return absolute * maximum
			else:
				return absolute
		except ZeroDivisionError or ValueError:
			return None
	elif any_above(channels, 1):
		if channels[0] == 1:
			a = np.asarray([a] * b)
		elif channels[1] == 1:
			b = np.asarray([b] * a)
		elif channels[0] != channels[1]:
			if channels[0] > channels[1]:
				channels = (channels[1], channels[1])
				a = a[:len(b)]
			elif channels[0] < channels[1]:
				channels = (channels[0], channels[0])
				b = b[:len(a)]
		try:
			if return_tuple:
				absolute = 1 - abs((b - a) / maximum)  # this will break if not ndarrays
				if scaled:
					return absolute * maximum
				else:
					return absolute
			else:
				absolute = 1 - abs((b - a) / maximum)  # this will break if not ndarrays
				if scaled:
					return np.mean(absolute * maximum)
				else:
					return np.mean(absolute)
		except ZeroDivisionError or ValueError:
			return None
	else:
		raise InvalidArgumentError("channel parameter is not formatted correctly")


def compare_num(a, b, maximum=1, scaled=True, opacity_mode=False):  # return_tuple requires inputting lists/ndarrays, channel values of both 1 treats inputs as single numbers, 0 as strings
	# print(a, b, channels, flush=True)
	try:
		absolute = 1 - abs((b - a)/maximum)
		if scaled:
			return absolute * maximum
		else:
			return absolute
	except ZeroDivisionError or ValueError:
		return None


def compare_array(a, b, maximum=1, scaled=True, opacity_mode=False, return_tuple=False):  # return_tuple requires inputting lists/ndarrays, channel values of both 1 treats inputs as single numbers, 0 as strings
	# print(a, b, channels, flush=True)
	try:
		if return_tuple:
			absolute = 1 - abs((b - a) / maximum)  # this will break if not ndarrays
			if scaled:
				return absolute * maximum
			else:
				return absolute
		else:
			absolute = 1 - abs((b - a) / maximum)  # this will break if not ndarrays
			if scaled:
				return np.mean(absolute * maximum)
			else:
				return np.mean(absolute)
	except ZeroDivisionError or ValueError:
		return None


def time_dif(offset=0, mul=1000):
	print((time.time() * mul) - offset)


class FeatureMatrix:  # needs odd sized matrix
	"""importing non-square odd-sided images has unpredictable results. do _not_ complain if this crashes or gives incorrect results unless you are doing so in a PR to add support for that. Much of the convolution logic assumes a square odd-sided matrix and would slow down, potentially by a lot, if this was removed. If support for such features is added, it will be in the form of opacity support; the image would be fit into the smallest odd-sided square with the extra pixels set as transparent."""
	def __init__(self, input_feature: str, multichannel: bool = False, opacity: bool = False):
		"""enabling multichannel if the image is not multichannel has unpredictable results. do _not_ complain if this crashes or gives incorrect results. gray images imported in rgb will function identical to if they were imported as grayscale, other than array dimension differences."""
		if multichannel is False:
			self.multi = False
			self.feature = np.asarray(imageio.imread(str(abs_path(input_feature)), pilmode="L"), dtype=np.float32)
			self.a, self.b = self.feature.shape
			self.c = 1
		elif multichannel is True:
			self.multi = True
			if opacity is False:
				self.opacity = False
				self.feature = np.asarray(imageio.imread(str(abs_path(input_feature)), pilmode="RGB"), dtype=np.float32)
				self.a, self.b, self.c = self.feature.shape
			elif opacity is True:
				self.opacity = True
				self.feature = np.asarray(imageio.imread(str(abs_path(input_feature)), pilmode="RGBA"), dtype=np.float32)
				self.a, self.b, self.c = self.feature.shape
			else:
				pass
		else:
			pass
		self.x_offset = math.ceil(self.b / 2) - 1
		self.y_offset = math.ceil(self.a / 2) - 1

	def coord(self, x, y):  # centered, use negative coords too, again, matrix must be odd sized
		x = x + self.x_offset
		y = (-1 * y) + self.y_offset
		if self.multi:
			return self.feature[y][x]
		else:
			return self.feature[y][x]

	def __getitem__(self, item):
		return self.feature.__getitem__(item)

	def __repr__(self):
		out = ""
		if not self.multi:
			for row in self.feature:
				for item in row:
					# print(" " * (space - len(str(item))), end="")
					out += str("~" * (3 - len(str(item)))) + str(item) + "|"
				out += "\n"
			return out
		elif self.multi:
			for row in self.feature:
				for item in row:
					for bit in item:
						out += str("~" * (3 - len(str(bit)))) + str(bit) + "~"
					out += "|"
				out += "\n"
			return out


class ImageMatrix:
	def __init__(self, input_image: str, multichannel: bool = False, opacity: bool = False):
		if multichannel is False:
			self.multi = False
			self.opacity = False
			self.image = np.asarray(imageio.imread(str(abs_path(input_image)), pilmode="L"), dtype=np.float32)
			self.a, self.b = self.image.shape
			self.c = 1
		elif multichannel is True:
			self.multi = True
			if opacity is False:
				self.opacity = False
				self.image = np.asarray(imageio.imread(str(abs_path(input_image)), pilmode="RGB"), dtype=np.float32)
				self.a, self.b, self.c = self.image.shape
			elif opacity is True:
				self.opacity = True
				self.image = np.asarray(imageio.imread(str(abs_path(input_image)), pilmode="RGBA"), dtype=np.float32)
				self.a, self.b, self.c = self.image.shape
			else:
				pass
		else:
			pass

	def convolve(self, input_feature: FeatureMatrix, mono_out: bool = False, progress_bar: bool = True, run_bar: bool = True, name_from: str = "", name_feature: str = "", name_append: str = "", buffer_load_bar: bool = False):

		segment_value = (self.a * self.b) / 80
		segments = [int(segment_value * n) for n in range(1, 81)]

		if run_bar:
			if progress_bar:
				if name_from and name_feature:
					name_len = len(name_from) + len(name_feature) + len(name_append) + 8
					if name_len > 80:
						if name_append:
							name_len = len(name_from) + len(name_feature) + 5
							if name_len > 80:
								print("{", end="", flush=True)
								print("-" * 80, end="", flush=True)
								print("}", flush=True)
							else:
								print("{", end="", flush=True)
								print(name_from, end="", flush=True)
								print(" @> ", end="", flush=True)
								print(name_feature, end="", flush=True)
								print(" ", end="", flush=True)
								print("-" * (80 - name_len), end="", flush=True)
								print("}", flush=True)
						print("{", end="", flush=True)
						print("-" * 80, end="", flush=True)
						print("}", flush=True)
					else:
						print("{", end="", flush=True)
						print(name_from, end="", flush=True)
						print(" @> ", end="", flush=True)
						print(name_feature, end="", flush=True)
						if name_append:
							print(" + ", end="", flush=True)
							print(name_append, end="", flush=True)
						else:
							name_len -= 3
						print(" ", end="", flush=True)
						print("-" * (80 - name_len), end="", flush=True)
						print("}", flush=True)
				else:
					print("{", end="", flush=True)
					print("-" * 80, end="", flush=True)
					print("}", flush=True)
			else:
				if name_from and name_feature:
					name_len = len(name_from) + len(name_feature) + len(name_append) + 8
					if name_len > 80:
						if name_append:
							name_len = len(name_from) + len(name_feature) + 5
							if name_len > 80:
								print("<", end="", flush=True)
								print("-" * 80, end="", flush=True)
								print(">", flush=True)
							else:
								print("<", end="", flush=True)
								print(name_from, end="", flush=True)
								print(" @> ", end="", flush=True)
								print(name_feature, end="", flush=True)
								print(" ", end="", flush=True)
								print("-" * (80 - name_len), end="", flush=True)
								print(">", flush=True)
						print("<", end="", flush=True)
						print("-" * 80, end="", flush=True)
						print(">", flush=True)
					else:
						print("<", end="", flush=True)
						print(name_from, end="", flush=True)
						print(" @> ", end="", flush=True)
						print(name_feature, end="", flush=True)
						if name_append:
							print(" + ", end="", flush=True)
							print(name_append, end="", flush=True)
						else:
							name_len -= 3
						print(" ", end="", flush=True)
						print("-" * (80 - name_len), end="", flush=True)
						print(">", flush=True)
				else:
					print("<--small-image-no-progress-bar--------------------------------------------------->", flush=True)

		seg_counter = done_amount = 0

		output = np.zeros(self.image.shape, dtype=np.float32)

		width, height, depth = input_feature.a, input_feature.b, input_feature.c
		width_partial, height_partial = int(math.floor(width/2)), int(math.floor(height/2))  # avg_val = width * height

		rows = np.array(range(-width_partial, width_partial + 1), dtype=np.intp)
		cols = np.array(range(-height_partial, height_partial + 1), dtype=np.intp)

		feat_height = range(0, height)
		feat_width = range(0, width)

		clrmono = (0, 1, 2) if self.multi and mono_out else (0, 1)

		if progress_bar:
			print("(", end="", flush=True)
		for y in range(self.a):
			check_bounds_y = (y <= height_partial + 1) or (y >= (self.a - height_partial - 1))

			for x in range(self.b):
				check_bounds_x = (x <= width_partial + 1) or (x >= (self.b - width_partial - 1))

				value = self.convolve_sub(input_feature.feature, x, y, rows, cols, feat_width, feat_height, width, height, check_bounds_x, check_bounds_y, clrmono=clrmono)

				output[y][x] = value
				done_amount += 1

				if progress_bar and run_bar:
					if buffer_load_bar:
						try:
							if int(done_amount) == segments[seg_counter]:
								seg_counter += 1
								print("=", end="", flush=True)
								time.sleep(0.0008)
						except IndexError:
							pass
					else:
						try:
							if int(done_amount) == segments[seg_counter]:
								seg_counter += 1
								print("=", end="", flush=True)
						except IndexError:
							pass
		if run_bar:
			if progress_bar:
				print(")", flush=True)
		return output

	def convolve_sub(self, input_feature: np.ndarray, x, y, rows_in: np.ndarray, cols_in: np.ndarray, feat_width, feat_height, width, height, check_bounds_x, check_bounds_y, clrmono=(0, 1)):
		rows = rows_in + x
		cols = cols_in + y

		if check_bounds_x:
			masks_r = getsides(
					(0 <= rows).__and__(
							(rows < self.b)),
					True, cond=eq)
		else:
			masks_r = (0, width - 1, width)

		if check_bounds_y:
			masks_c = getsides(
					(0 <= cols).__and__(
							(cols < self.a)),
					True, cond=eq)
		else:
			masks_c = (0, height - 1, height)

		data = self.image[np.ix_(

				cols[masks_c[0]
					:masks_c[1]+1],

				rows[masks_r[0]
					:masks_r[1]+1])]

		if check_bounds_x or check_bounds_y:
			feature_masked = input_feature[np.ix_(

					feat_height[masks_c[0]
						:masks_c[1]+1],

					feat_width[masks_r[0]
						:masks_r[1]+1])]
		else:
			feature_masked = input_feature

		try:
			cvld = (1 - abs((data - feature_masked) / 255)) * 255
		except ValueError as error:
			print2d(data)
			print("###")
			print2d(feature_masked)
			print("###")
			print(x, y, check_bounds_x, check_bounds_y, width, height, int(math.floor(width/2)), int(math.floor(height/2)))
			print(x, y, self.a, self.b, rows, cols, masks_r, masks_c, feat_width, feat_height, rows[masks_r[0]:masks_r[1] + 1], cols[masks_c[0]:masks_c[1] + 1])
			raise error
		return np.mean(cvld, axis=clrmono, dtype=np.float64)


class Features:
	__slots__ = []

	class Three:
		__slots__ = []

		class Arrow:
			__slots__ = []

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/arrow_up.png"

			class Down:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/arrow_down.png"

			class Left:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/arrow_left.png"

			class Right:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/arrow_right.png"

			def __repr__(self):
				return "featuresbasic/features3/arrow_up.png"

		class Line:
			__slots__ = []

			class Across:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/line_across.png"

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/line_up.png"

			class DownLeft:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/line_dl.png"

			class DownRight:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features3/line_dr.png"

			def __repr__(self):
				return "featuresbasic/features3/line_up.png"

		class Dot:
			__slots__ = []

			def __repr__(self):
				return "featuresbasic/features3/dot.png"

		class Donut:
			__slots__ = []

			def __repr__(self):
				return "featuresbasic/features3/donut.png"

		class Hole:
			__slots__ = []

			def __repr__(self):
				return "featuresbasic/features3/hole.png"

		class X:
			__slots__ = []

			def __repr__(self):
				return "featuresbasic/features3/x.png"

	class Five:
		__slots__ = []

		class Arrow:
			__slots__ = []

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/arrow_up.png"

			class Down:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/arrow_down.png"

			class Left:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/arrow_left.png"

			class Right:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/arrow_right.png"

			def __repr__(self):
				return "featuresbasic/features5/arrow_up.png"

		class Line:
			__slots__ = []

			class Across:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/line_horizontal.png"

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/line_vertical.png"

			class DownLeft:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/line_dl.png"

			class DownRight:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/line_dr.png"

			class Broken:
				__slots__ = []

				class Across:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features5/line_horizontal_broken.png"

				class Up:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features5/line_vertical_broken.png"

				class DownLeft:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features5/line_dl_broken.png"

				class DownRight:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features5/line_dr_broken.png"

				def __repr__(self):
					return "featuresbasic/features5/line_vertical_broken.png"

			def __repr__(self):
				return "featuresbasic/features5/line_vertical.png"

		class Dot:
			__slots__ = []

			class Large:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/dot_large.png"

			def __repr__(self):
				return "featuresbasic/features5/dot.png"

		class Donut:
			class Filled:
				def __repr__(self):
					return "featuresbasic/features5/donut_large.png"

			def __repr__(self):
				return "featuresbasic/features5/donut.png"

		class Diamond:
			__slots__ = []

			class Center:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/diamond_center.png"

			class Solid:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/diamond_full.png"

			class Small:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/diamond_small.png"

			def __repr__(self):
				return "featuresbasic/features5/diamond.png"

		class Curve:
			__slots__ = []

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/curve_up.png"

			class Down:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/curve_down.png"

			class Left:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/curve_left.png"

			class Right:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/curve_right.png"

			def __repr__(self):
				return "featuresbasic/features5/curve_up.png"

		class Circle:
			__slots__ = []

			class Large:
				__slots__ = []

				class Filled:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features5/curve_right.png"

				def __repr__(self):
					return "featuresbasic/features5/circle_large.png"

			def __repr__(self):
				return "featuresbasic/features5/circle_small.png"

		class X:
			__slots__ = []

			class Large:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/x_large.png"

			def __repr__(self):
				return "featuresbasic/features5/x.png"

		class Cross:
			__slots__ = []

			class Hollow:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features5/cross_hollow.png"

			def __repr__(self):
				return "featuresbasic/features5/cross_solid.png"

	class Seven:
		__slots__ = []

		class Arrow:
			__slots__ = []

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/arrow_up.png"

			class Down:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/arrow_down.png"

			class Left:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/arrow_left.png"

			class Right:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/arrow_right.png"

			def __repr__(self):
				return "featuresbasic/features7/arrow_up.png"

		class Curve:
			__slots__ = []

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/curve_up.png"

			class Down:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/curve_down.png"

			class Left:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/curve_left.png"

			class Right:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/curve_right.png"

			def __repr__(self):
				return "featuresbasic/features7/curve_up.png"

		class Line:
			__slots__ = []

			class Across:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/line_across.png"

			class Up:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/line_up.png"

			class DownLeft:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/line_dl.png"

			class DownRight:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/line_dr.png"

			class Triple:
				__slots__ = []

				class DownLeft:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/line_triple_dl.png"

				class DownRight:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/line_triple_dr.png"

			class Double:
				__slots__ = []

				class DownLeft:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/line_double_dl.png"

				class DownRight:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/line_double_dr.png"

			def __repr__(self):
				return "featuresbasic/features7/line_up.png"

		class Dot:
			__slots__ = []

			def __repr__(self):
				return "featuresbasic/features7/dot.png"

		class Circle:
			__slots__ = []

			class Large:
				__slots__ = []

				class Center:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/circle_large_center.png"

				def __repr__(self):
					return "featuresbasic/features7/circle_large.png"

			class Medium:
				__slots__ = []

				class Broken:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/circle_broken.png"

				class Center:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/circle_med_center.png"

				def __repr__(self):
					return "featuresbasic/features7/circle_med.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/circle_small.png"

			class ExtraSmall:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/dot.png"

		class Cross:
			__slots__ = []

			class Large:
				__slots__ = []

				class Outline:
					__slots__ = []

					class Round:
						__slots__ = []

						def __repr__(self):
							return "featuresbasic/features7/cross_outline_round.png"

					def __repr__(self):
						return "featuresbasic/features7/cross_outline.png"

				def __repr__(self):
					return "featuresbasic/features7/cross_large.png"

			class Medium:
				__slots__ = []

				class Outline:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/cross_outline_small.png"

				def __repr__(self):
					return "featuresbasic/features7/cross_med.png"

			class Small:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/cross_small.png"

			class ExtraSmall:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/cross_med.png"

		class X:
			__slots__ = []

			class Large:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/x_large.png"

			class Medium:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/x_med.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/x_small.png"

			def __repr__(self):
				return "featuresbasic/features7/x_med.png"

		class Square:
			__slots__ = []

			class Large:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/square_large.png"

			class Medium:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/square_small.png"

			class Small:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/square_small.png"

		class Spiral:
			__slots__ = []

			class CW:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/spiral_cw.png"

			class CCW:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/spiral_ccw.png"

			def __repr__(self):
				return "featuresbasic/features7/spiral_cw.png"

		class Diamond:
			__slots__ = []

			class Large:
				__slots__ = []

				class Center:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/diamond_large_center.png"

				def __repr__(self):
					return "featuresbasic/features7/diamond_large.png"

			class Small:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/diamond_small.png"

			def __repr__(self):
				return "featuresbasic/features7/diamond_small.png"

		class Donut:
			__slots__ = []

			class Large:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/donut_large.png"

			class Medium:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/donut_med.png"

			class Small:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/donut_small.png"

			class ExtraSmall:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/donut_small.png"

		class Edges:
			__slots__ = []

			class Vertical:
				__slots__ = []

				class Small:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/edges_vert_small.png"

				def __repr__(self):
					return "featuresbasic/features7/edges_vert.png"

			class Sides:
				__slots__ = []

				class Small:
					__slots__ = []

					def __repr__(self):
						return "featuresbasic/features7/edges_sides_small.png"

				def __repr__(self):
					return "featuresbasic/features7/edges_sides.png"

			def __repr__(self):
				return "featuresbasic/features7/edges_sides.png"

	class Color5:
		__slots__ = []

		class Horiz:
			__slots__ = []

			class RToL:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color5/horiz.png"

			class LToR:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color5/r_horiz.png"

			def __repr__(self):
				return "featuresbasic/color5/horiz.png"

		class Vert:
			__slots__ = []

			class TToB:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color5/r_vert.png"

			class BToT:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color5/vert.png"

			def __repr__(self):
				return "featuresbasic/color5/r_vert.png"

	class Color9:
		__slots__ = []

		class CW:
			__slots__ = []

			class RedTop:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/cw_r_t.png"

			class RedRight:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/cw_r_r.png"

			class RedLeft:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/cw_r_l.png"

			class RedBottom:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/cw_r_b.png"

			def __repr__(self):
				return "featuresbasic/color9/cw_r_t.png"

		class CCW:
			__slots__ = []

			class RedTop:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/ccw_r_t.png"

			class RedRight:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/ccw_r_r.png"

			class RedLeft:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/ccw_r_l.png"

			class RedBottom:
				__slots__ = []

				def __repr__(self):
					return "featuresbasic/color9/ccw_r_b.png"

			def __repr__(self):
				return "featuresbasic/color9/ccw_r_t.png"


class DimensionError(Exception):
	__slots__ = ["message"]

	def __init__(self, message):
		self.message = message

	def __repr__(self):
		return "DimensionError: " + self.message


def enlarge(array: np.ndarray, amount):
	# print2d(array)
	# print(type(array))
	if (type(array) == np.ndarray) and (type(amount) == int):
		# dimensions = array.shape()
		if amount <= 1:
			return array
		else:
			if len(array.shape) == 2:
				size = amount, amount
			elif len(array.shape) == 3:
				size = amount, amount, 1
			else:
				raise DimensionError("unsupported array shape")
			# print2d(np.kron(array, np.ones(size, dtype="float")))
			return np.kron(array, np.ones(size, dtype="float"))


def remap(array: np.ndarray, enable=True, minimum_value: Optional[int] = None, maximum_value: Optional[int] = None, max_val=255):
	if not enable and (minimum_value is not None) and (maximum_value is not None):
		return array
	if minimum_value is not None:
		minimum_value = abs(minimum_value)
	if maximum_value is not None:
		maximum_value = abs(maximum_value)
	min = np.amin(array)
	max = np.amax(array)
	# print(array, min, max)
	if minimum_value is not None:
		array = array * (array > minimum_value) + np.ones_like(array) * ((array > minimum_value) ^ True) * minimum_value
	# print(array, min, max)
	if maximum_value is not None:
		array = array * (array < maximum_value) + np.ones_like(array) * ((array < maximum_value) ^ True) * maximum_value
	min = np.amin(array)
	max = np.amax(array)
	# print(array, min, max)
	dif = max_val / (max - min)
	out = (array - min) * dif
	# print(out, dif)
	return out


def bounds(i, l, u):
	if i > u:
		return u
	elif i < l:
		return l
	else:
		return i


def save(array: np.ndarray, name):
	# print2d(array)
	name = name + ".png"
	imageio.imsave(name, array.astype(dtype=np.uint8))


def show_image(ndarray: np.ndarray):
	plt.imshow(ndarray, interpolation='nearest')
	plt.show()


def saveview(array: np.ndarray, name, mode="save"):
	mode = mode.lower()
	if mode not in ["save", "view", "both"]:
		raise InvalidArgumentError("mode parameter must be 'save', 'view' or 'both'")
	if mode == "save":
		save(array, name)
		return
	elif mode == "view":
		show_image(array)
		return
	elif mode == "both":
		save(array, name)
		show_image(array)
		return
	else:
		raise InvalidArgumentError("mode parameter must be 'save', 'view' or 'both'")


def abs_path(rel_path):
	script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
	return os.path.join(str(script_dir), str(rel_path))


def called_from(depth):
	out = []
	for x in range(0, depth):
		try:
			out.append(inspect.stack()[x + 2][3])
		except IndexError:
			return out
	return out


def debug(*values, stack_size=4, mini=False):
	typedvals = list(map(lambda x: str(x) + " is " + str(type(x)) + ",", values))
	caller = inspect.getframeinfo(inspect.stack()[1][0])
	if not mini:
		print("<!>", *called_from(stack_size), "IN:" + str(caller.filename) + ":" + str(caller.lineno), ":|:", *typedvals, flush=True)
	elif mini:
		print("<!>", *called_from(stack_size), "line:" + str(caller.lineno), ":|:", *typedvals, flush=True)


class Convolution:
	def __init__(self, *, input_image: str, multichannel: bool = False, opacity: bool = False):
		self.multichannel: bool = multichannel
		self.opacity: bool = opacity
		self.main_image: ImageMatrix = ImageMatrix(input_image, multichannel, opacity)
		self.name: str = str(input_image).split(".")[0]
		self.number: int = 0
		self.resize: int = 1
		self.size: float = self.main_image.a * self.main_image.b

	def scale(self, value: int):
		self.resize: int = value

	def convolve(self, input_feature: object, clip_tuple=(False, None, None, 255), resize: int = 1, append_number: bool = False, viewmode: str = "save", mono_out: bool = False, run_bar: bool = True, append_custom: Union[str, int] = ""):  # mono_out does nothing without multichannel

		append = self.name + "_" + str(input_feature).split("/")[-1].split(".")[0] + "_" + str(input_feature).split("/")[-2]
		name_feature: str = str(input_feature).split("/")[-1].split(".")[0] + "_" + str(input_feature).split("/")[-2]
		name_append: str = ""

		if mono_out:
			append += "_mono"
			name_append += "mono"
		self.number += 1

		if append_number is True:
			append += "_" + str(self.number)
			name_append += " " + str(self.number)
		if not (append_custom == ""):
			append += "_" + str(append_custom)
			name_append += " " + str(append_custom)

		convolve_feature = FeatureMatrix(input_feature, self.multichannel, self.opacity)

		if (self.size * convolve_feature.a * convolve_feature.b) <= 300_000:  # if you have a slower or faster computer, you may want to change this
			progress_bar: bool = False
		else:
			progress_bar: bool = True

		if (self.size * convolve_feature.a * convolve_feature.b) <= 400_000:  # if you have a slower or faster computer, you may want to change this
			buffer_load_bar: bool = True
		else:
			buffer_load_bar: bool = False

		if clip_tuple is False:
			clip_tuple = (False, None, None, 255)
		elif clip_tuple is True:
			clip_tuple = (True, None, None, 255)

		out: np.ndarray = self.main_image.convolve(convolve_feature, mono_out, progress_bar=progress_bar, run_bar=run_bar, name_from=str(self.name.split("/")[-1]), name_feature=str(name_feature), name_append=name_append, buffer_load_bar=buffer_load_bar)

		output: np.ndarray = enlarge(remap(out, *clip_tuple), int(resize * self.resize))

		saveview(output, append, viewmode)


# '''
color_tester = Convolution(input_image="test images/color_tester.png", multichannel=True, opacity=False)
color_tester.convolve(Features.Color9.CW.RedTop(), (True, None, None, 255), False, viewmode="save")
color_tester.convolve(Features.Color9.CW.RedTop(), (True, None, None, 255), False, viewmode="save", mono_out=True)
color_tester.convolve(Features.Color9.CW.RedRight(), (True, None, None, 255), False, viewmode="save", mono_out=True)
color_tester.convolve(Features.Color9.CW.RedBottom(), (True, None, None, 255), False, viewmode="save", mono_out=True)
color_tester.convolve(Features.Color9.CW.RedLeft(), (True, None, None, 255), False, viewmode="save", mono_out=True)
color_tester.convolve(Features.Color9.CCW.RedTop(), (True, None, None, 255), False, viewmode="save")
color_tester.convolve(Features.Color9.CCW.RedBottom(), (True, None, None, 255), False, viewmode="save")
# '''

# '''
input2 = Convolution(input_image="test images/input2 xsmall.png", multichannel=False, opacity=False)
input2.convolve(Features.Seven.X.Large(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.Up(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.Across(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.DownLeft(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.Double.DownLeft(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.Triple.DownLeft(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.DownRight(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.Double.DownRight(), (True, None, None, 255), False, viewmode="save")
input2.convolve(Features.Seven.Line.Triple.DownRight(), (True, None, None, 255), False, viewmode="save")
# '''

# o_large = Convolution(input_image="test images/input_o_large.png", multichannel=False, opacity=False)
# o_large.convolve(Features.Three.Arrow.Down(), (True, None, None, 255), viewmode="save")


# input2 = Convolution(input_image="test images/colorstripes_small.png", multichannel=True, opacity=False)
# input2.convolve(Features.Color9.CW.RedTop(), (True, None, None, 255), viewmode="save")
# input2.convolve(Features.Color9.CW.RedTop(), (True, None, None, 255), viewmode="save", mono_out=True)
