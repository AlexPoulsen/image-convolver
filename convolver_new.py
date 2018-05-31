import numpy as np
from scipy import misc
import math
import os
import matplotlib.pyplot as plt
import inspect
import time
from typing import *
import atexit
import imageio


class TimingVariables:
	def __init__(self):
		self.enable = False


def timeme(method, total_var=None):
	def wrapper(*args, **kw):
		if not time_vars.enable:
			return method(*args, **kw)
		start_time = int(round(time.time() * 1000))
		result = method(*args, **kw)
		end_time = int(round(time.time() * 1000))
		# print(end_time - start_time, 'ms')
		time_vars.timer += (end_time - start_time)
		return result
	return wrapper


time_vars = TimingVariables()


def curry(f, x):
	def curried_function(*args, **kw):
		return f(*((x, )+args), **kw)
	return curried_function


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


class DeprecatedFunctionError(Exception):
	__slots__ = ["message"]

	def __init__(self, message):
		self.message = message

	def __repr__(self):
		return "DeprecatedFunctionError: " + self.message


class DeprecationReturnObject(object):
	__slots__ = ["uuid", "options", "logs", "message", "log_var", "log_type", "logging", "persistence", "crash"]

	def __init__(self, uuid):
		self.uuid = uuid
		self.options = False
		self.logs = {}

	def log(self, L):
		try:
			return self.logs[L]
		except KeyError:
			return None

	def __call__(self, result: Any, message: Optional[str] = None, log_var: Optional[object] = None, log_type: Union[bool, int, str] = bool, persistence: bool = False, crash: bool = False):
		if self.options:
			if log_var != self.log_var:
				try:
					self.logs[log_var] = log_type()
					if log_type is int:
						self.logs[log_var] += 1
				except TypeError:
					self.logs[log_var] = type(log_type)()
			if self.crash:
				raise DeprecatedFunctionError(f"{self.message} result: {result}")
			if self.persistence:
				print(f"Deprecated function warning: {self.message}")
			if self.logging:
				if self.log_type is bool:
					self.logs[self.log_var] = True
					self.logging = True if self.persistence else False
				elif self.log_type is int:
					self.logs[self.log_var] += 1
				elif self.log_type is str:
					self.logs[self.log_var] = f"Deprecated Function {self.uuid} called"
				else:
					raise InvalidArgumentError(f"Variable {self.log_var} of type {self.log_type} is not a valid input")
			return result
		self.message: Optional[str] = message
		self.log_var: Optional[object] = log_var
		self.log_type: Union[bool, int, str] = log_type
		if log_var:
			self.logging = True
			try:
				self.logs[log_var] = log_type()
				if log_type is int:
					self.logs[log_var] += 1
			except TypeError:
				self.logs[log_var] = type(log_type)()
		else:
			self.logging = False
		self.persistence: bool = persistence
		self.crash: bool = crash
		self.options = True
		print(f"Deprecated function warning: {self.message}")
		return result


class DeprecationReturn(object):
	__slots__ = []

	ids = {}  # where = inspect.stack()[2][3] + str(inspect.getframeinfo(inspect.stack()[1][0]).lineno)

	def __init__(self):
		pass

	@atexit.register
	def end_stats():
		if len(deprecation_return.ids) == 0:
			return
		print(f"{len(deprecation_return.ids)} Deprecated Function Call{'s' if len(deprecation_return.ids) != 1 else ''}")
		for I in deprecation_return.ids:
			for L in deprecation_return.ids[I].logs:
				print(f"ID: {I} with log variable: {L} as {deprecation_return.ids[I].logs[L]}, {'persistent ' if deprecation_return.ids[I].persistence else ''}{'crashing' if deprecation_return.ids[I].crash else ''}")

	def __call__(self, uuid):
		if uuid in self.ids:
			return self.ids[uuid]
		else:
			self.ids[uuid] = DeprecationReturnObject(uuid)
			return self.ids[uuid]


deprecation_return = DeprecationReturn()


def time_dif(offset=0, mul=1000):
	print((time.time() * mul) - offset)


class ImageMatrix:
	def __init__(self, input_image, multichannel=False, opacity=False):
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

	def convolve(self, input_feature, mono_out=False, progress_bar=True, run_bar=True, name_from="", name_feature="", name_append="", buffer_load_bar=False):
		# time_offset = time.time() * 1000
		# time_dif(time_offset)
		segment_value = (self.a * self.b) / 80
		segments = [int(segment_value * n) for n in range(1, 81)]
		# print(progress_bar)
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
		# print("\ndc", depth, self.c)
		rows = np.array(range(-width_partial, width_partial + 1), dtype=np.intp)
		cols = np.array(range(-height_partial, height_partial + 1), dtype=np.intp)
		feat_height = range(0, height)
		feat_width = range(0, width)
		clrmono = (0, 1, 2) if self.multi and mono_out else (0, 1)
		# time_dif(time_offset)
		if progress_bar:
			print("(", end="", flush=True)
		for y in range(self.a):
			check_bounds_y = (y <= height_partial + 1) or (y >= (self.a - height_partial - 1))
			for x in range(self.b):
				check_bounds_x = (x <= width_partial + 1) or (x >= (self.b - width_partial - 1))
				value = self.convolve_sub(input_feature.feature, x, y, rows, cols, feat_width, feat_height, width, height, check_bounds_x, check_bounds_y, return_tuple=((not mono_out) and self.multi), clrmono=clrmono)
				# print("value", value, type(value))
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
		# time_dif(time_offset)
		# print("#####")
		return output

	def old_convolve_sub(self, input_feature, x, y, width, height, depth, width_partial, height_partial, avg_val, return_tuple=False, increment=0, channels=(1, 1)):
		for ya in range(height):
			if (ya + y > self.a) or (ya + y < 0):
				avg_val -= width
				continue
			for xa in range(width):
				if (xa + x > self.b) or (xa + x < 0):
					avg_val -= 1
					continue
				out = compare(input_feature.coord(xa - width_partial, ya - height_partial), self.image[y + ya - height_partial][x + xa - width_partial], 255, self.opacity, return_tuple=return_tuple, channels=channels)
				# print(increment, out)
				increment += out
		return deprecation_return("old_convolve_sub")(increment / avg_val, "Please use convolve_sub instead. It is very fast compared to the older algorithm. If you must use this, you can easily change it to use a regular return, which will speed it up a little.")

	def convolve_sub(self, input_feature: np.ndarray, x, y, rows_in: np.ndarray, cols_in: np.ndarray, feat_width, feat_height, width, height, check_bounds_x, check_bounds_y, return_tuple=False, clrmono=(0, 1)):
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

		# print(x, y, self.a, self.b, rows, cols, masks_r, masks_c, feat_width, feat_height, rows[masks_r[0]:masks_r[1]+1], cols[masks_c[0]:masks_c[1]+1], flush=True)
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

		# print("###")
		# print2d(data)
		# print("###")
		# print2d(feature_masked)
		# print("###")
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


class FeatureMatrix:  # needs odd sized matrix
	"""importing non-square odd-sided images has unpredictable results. do _not_ complain if this crashes or gives incorrect results unless you are doing so in a PR to add support for that. Much of the convolution logic assumes a square odd-sided matrix and would slow down, potentially by a lot, if this was removed. If support for such features is added, it will be in the form of opacity support; the image would be fit into the smallest odd-sided square with the extra pixels set as transparent."""
	def __init__(self, input_feature, multichannel=False, opacity=False):
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


def enlarge(array, amount):
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


def remap(array, enable=True, minimum_value: Optional[int] = None, maximum_value: Optional[int] = None, max_val=255):
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


def save(array, name):
	# print2d(array)
	name = name + ".png"
	imageio.imsave(name, array.astype(dtype=np.uint8))


def show_image(ndarray):
	plt.imshow(ndarray, interpolation='nearest')
	plt.show()


def saveview(array, name, mode="save"):
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

	def convolve(self, input_feature: object, clip_tuple = (False, None, None, 255), resize: int = 1, append_number: bool = False, viewmode: str = "save", mono_out: bool = False, run_bar: bool = True, append_custom: Union[str, int] = ""):  # mono_out does nothing without multichannel

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

		# print(self.size * convolve_feature.a * convolve_feature.b)

		# print(progress_bar, self.size, convolve_size)
		out: np.ndarray = self.main_image.convolve(convolve_feature, mono_out, progress_bar=progress_bar, run_bar=run_bar, name_from=str(self.name.split("/")[-1]), name_feature=str(name_feature), name_append=name_append, buffer_load_bar=buffer_load_bar)

		output: np.ndarray = enlarge(remap(out, *clip_tuple), int(resize * self.resize))

		saveview(output, append, viewmode)


'''
o_large = Convolution("test images/input_o_large.png", False, False)
o_large.scale(4)
o_large.convolve(Features.Three.Line.Across(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Line.Up(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Line.DownLeft(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Line.DownRight(), 0, 1, False, view=False)
o_large.convolve(Features.Three.X(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Dot(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Arrow.Up(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Arrow.Down(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Arrow.Left(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Arrow.Right(), 0, 1, False, view=False)
# '''

'''
x_large = Convolution("test images/input_x_large.png", False, False)
x_large.scale(4)
x_large.convolve(Features.Three.Line.Across(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Line.Up(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Line.DownLeft(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Line.DownRight(), 0, 1, False, view=False)
x_large.convolve(Features.Three.X(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Dot(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Arrow.Up(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Arrow.Down(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Arrow.Left(), 0, 1, False, view=False)
x_large.convolve(Features.Three.Arrow.Right(), 0, 1, False, view=False)
# '''


"""

def time_tests(count):
	time_vars.timer = 0
	time_vars.enable = True
	approx_time_per_pixel = 0
	tests = 0
	x_large = Convolution("test images/input_x_large.png", False, False)
	for n in range(count):
		x_large.convolve(Features.Three.Donut(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 3x3 gs")
	approx_time_per_pixel = (time_vars.timer / count) / (21 * 21 * 3 * 3 * 1)
	x_large = Convolution("test images/input_x_large.png", True, False)
	for n in range(count):
		x_large.convolve(Features.Three.Donut(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 3x3 mono -> clr")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 3 * 3 * 3)
	tests += 1
	for n in range(count):
		x_large.convolve(Features.Three.Donut(), 0, 1, False, mono_out=True, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 3x3 mono -> clr -> mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 3 * 3 * 3)
	tests += 1
	for n in range(count):
		x_large.convolve(Features.Color5.Horiz.LToR(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 5x5 mono x clr")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 5 * 5 * 3)
	tests += 1
	for n in range(count):
		x_large.convolve(Features.Color5.Horiz.LToR(), 0, 1, False, mono_out=True, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 5x5 mono x clr -> mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 5 * 5 * 3)
	tests += 1
	for n in range(count):
		x_large.convolve(Features.Color9.CCW.RedBottom(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 9x9 mono x clr")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 9 * 9 * 3)
	tests += 1
	for n in range(count):
		x_large.convolve(Features.Color9.CCW.RedBottom(), 0, 1, False, mono_out=True, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 9x9 mono x clr -> mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 9 * 9 * 3)
	tests += 1
	clr_small = Convolution("test images/input_color_small.png", True, False)
	for n in range(count):
		clr_small.convolve(Features.Three.Donut(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 3x3 clr x mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 9 * 9 * 3)
	tests += 1
	for n in range(count):
		clr_small.convolve(Features.Three.Donut(), 0, 1, False, mono_out=True, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 3x3 clr x mono -> mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 3 * 3 * 3)
	tests += 1
	for n in range(count):
		clr_small.convolve(Features.Color5.Horiz.LToR(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 5x5 clr")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 5 * 5 * 3)
	tests += 1
	for n in range(count):
		clr_small.convolve(Features.Color5.Horiz.LToR(), 0, 1, False, mono_out=True, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 5x5 clr -> mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 5 * 5 * 3)
	tests += 1
	for n in range(count):
		clr_small.convolve(Features.Color9.CCW.RedBottom(), 0, 1, False, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 9x9 clr")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 9 * 9 * 3)
	tests += 1
	for n in range(count):
		clr_small.convolve(Features.Color9.CCW.RedBottom(), 0, 1, False, mono_out=True, view=False, run_bar=False, append_custom="test_output")
	print(time_vars.timer / count, "ms 21x21 9x9 clr -> mono")
	approx_time_per_pixel += (time_vars.timer / count) / (21 * 21 * 9 * 9 * 3)
	tests += 1
	approx_time_per_pixel /= tests
	print(approx_time_per_pixel, "ms, approximate time per pixel")
	time_vars.timer = 0
	time_vars.enable = False  # remove the decorator for normal use


# time_tests(1000)
# """


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
