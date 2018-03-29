import numpy as np
from scipy import misc
import math
import os
import matplotlib.pyplot as plt
import inspect


class ImageMatrix:
	def __init__(self, input_image, multichannel=False, opacity=False):  # only enable multichannel if the feature is also multichannel
		if multichannel is False:
			self.multi = False
			self.opacity = False
			self.image = misc.imread(str(abs_path(input_image)), mode="L")
			self.a, self.b = self.image.shape
			self.c = 1
		elif multichannel is True:
			self.multi = True
			if opacity is False:
				self.opacity = False
				self.image = misc.imread(str(abs_path(input_image)), mode="RGB")
				self.a, self.b, self.c = self.image.shape
			elif opacity is True:
				self.opacity = True
				self.image = misc.imread(str(abs_path(input_image)), mode="RGBA")
				self.a, self.b, self.c = self.image.shape
			else:
				pass
		else:
			pass

	def convolve(self, input_feature, min_val=0, check_min=False, mono_out=False, progress_bar=True):
		segment_value = (self.a * self.b) / 80
		segments = [int(segment_value * n) for n in range(1, 81)]
		# print(progress_bar)
		if progress_bar:
			print("#", end="", flush=True)
			print("-" * 78, end="", flush=True)
			print("#", flush=True)
		else:
			print("#-small-image-no-progress-bar--------------------------------------------------#", end="", flush=True)
		seg_counter = done_amount = 0
		output = np.zeros(self.image.shape, dtype=np.float32)
		for ax in range(self.a):
			for bx in range(self.b):
				increment = not (self.multi and (not mono_out))
				value = self.convolve_sub(input_feature, bx, ax, return_tuple=((not mono_out) and self.multi), increment=(0 if increment else [0] * self.c))
				if check_min:
					if value < min_val:
						value = min_val
					output[ax][bx] = value
				else:
					output[ax][bx] = value
				done_amount += 1
				if progress_bar:
					try:
						if int(done_amount) == segments[seg_counter]:
							seg_counter += 1
							print("#", end="", flush=True)
					except IndexError:
						pass
		print()
		return output

	'''
	def convolve_sub_multi(self, input_feature, x, y):
		width, height, depth = input_feature.dims()
		width_partial = math.floor(width/2)
		height_partial = math.floor(height/2)
		multi_out = [0] * depth
		for channel in range(depth):
			avg_val = width * height
			increment = 0
			for ya in range(height):
				if (ya + y > self.a) or (ya + y < 0):
					avg_val -= width
					continue
				for xa in range(width):
					if (xa + x > self.b) or (xa + x < 0):
						avg_val -= 1
						continue
					increment += compare(input_feature.coord(xa - width_partial, ya - height_partial)[channel], self.image[y + ya - height_partial][x + xa - width_partial][channel], 255, "&", self.opacity)
			multi_out[channel] = (increment / avg_val)
		return multi_out
	# '''

	def convolve_sub(self, input_feature, x, y, return_tuple=False, increment=0):
		width, height, depth = input_feature.dims()
		width_partial = math.floor(width/2)
		height_partial = math.floor(height/2)
		avg_val = width * height
		for ya in range(height):
			if (ya + y > self.a) or (ya + y < 0):
				avg_val -= width
				continue
			for xa in range(width):
				if (xa + x > self.b) or (xa + x < 0):
					avg_val -= 1
					continue
				out = compare(input_feature.coord(xa - width_partial, ya - height_partial), self.image[y + ya - height_partial][x + xa - width_partial], 255, "&", self.opacity, return_tuple=return_tuple)
				# print(increment, out)
				increment = multi_add(increment, out, lists=return_tuple)
		return multi_div(increment, avg_val)


class FeatureMatrix:  # needs odd sized matrix
	def __init__(self, input_feature, multichannel=False, opacity=False):  # only enable multichannel if the image is also multichannel
		if multichannel is False:
			self.multi = False
			self.feature = misc.imread(str(abs_path(input_feature)), mode="L")
			self.a, self.b = self.feature.shape
			self.c = 0
		elif multichannel is True:
			self.multi = True
			if opacity is False:
				self.opacity = False
				self.feature = misc.imread(str(abs_path(input_feature)), mode="RGB")
				self.a, self.b, self.c = self.feature.shape
			elif opacity is True:
				self.opacity = True
				self.feature = misc.imread(str(abs_path(input_feature)), mode="RGBA")
				self.a, self.b, self.c = self.feature.shape
			else:
				pass
		else:
			pass
		'''
		if (self.a/2 - math.floor(self.a/2) == 0.5) or (self.b/2 - math.floor(self.b/2) == 0.5):
			self.feature.resize((self.a - 1, self.b - 1))
		# '''

	def coord(self, x, y):  # centered, use negative coords too, again, matrix must be odd sized
		x = x + math.ceil(self.b / 2) - 1  # |      |       (-1,  1) (0,  1) (1,  1)
		y = (-1 * y) + math.ceil(self.a / 2) - 1  # |       (-1,  0) (0,  0) (1,  0)
		if type(self.feature[0][0]) == np.uint8:  # |       (-1, -1) (0, -1) (1, -1)
			return float(self.feature[y][x].astype(float))
		elif type(self.feature[0][0]) == np.ndarray:
			return self.feature[y][x].tolist()
		else:
			return "type check fail"

	# def __len__(self):
	# 	return self.b, self.a

	def dims(self):
		return self.b, self.a, self.c

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
	class Three:
		class Arrow:
			class Up:
				def __repr__(self):
					return "featuresbasic/features3/arrow_up.png"

			class Down:
				def __repr__(self):
					return "featuresbasic/features3/arrow_down.png"

			class Left:
				def __repr__(self):
					return "featuresbasic/features3/arrow_left.png"

			class Right:
				def __repr__(self):
					return "featuresbasic/features3/arrow_right.png"

			def __repr__(self):
				return "featuresbasic/features3/arrow_up.png"

		class Line:
			class Across:
				def __repr__(self):
					return "featuresbasic/features3/line_across.png"

			class Up:
				def __repr__(self):
					return "featuresbasic/features3/line_up.png"

			class DownLeft:
				def __repr__(self):
					return "featuresbasic/features3/line_dl.png"

			class DownRight:
				def __repr__(self):
					return "featuresbasic/features3/line_dr.png"

			def __repr__(self):
				return "featuresbasic/features3/line_up.png"

		class Dot:
			def __repr__(self):
				return "featuresbasic/features3/dot.png"

		class Donut:
			def __repr__(self):
				return "featuresbasic/features3/donut.png"

		class Hole:
			def __repr__(self):
				return "featuresbasic/features3/hole.png"

		class X:
			def __repr__(self):
				return "featuresbasic/features3/x.png"

	class Five:
		class Arrow:
			class Up:
				def __repr__(self):
					return "featuresbasic/features5/arrow_up.png"

			class Down:
				def __repr__(self):
					return "featuresbasic/features5/arrow_down.png"

			class Left:
				def __repr__(self):
					return "featuresbasic/features5/arrow_left.png"

			class Right:
				def __repr__(self):
					return "featuresbasic/features5/arrow_right.png"

			def __repr__(self):
				return "featuresbasic/features5/arrow_up.png"

		class Line:
			class Across:
				def __repr__(self):
					return "featuresbasic/features5/line_horizontal.png"

			class Up:
				def __repr__(self):
					return "featuresbasic/features5/line_vertical.png"

			class DownLeft:
				def __repr__(self):
					return "featuresbasic/features5/line_dl.png"

			class DownRight:
				def __repr__(self):
					return "featuresbasic/features5/line_dr.png"

			class Broken:
				class Across:
					def __repr__(self):
						return "featuresbasic/features5/line_horizontal_broken.png"

				class Up:
					def __repr__(self):
						return "featuresbasic/features5/line_vertical_broken.png"

				class DownLeft:
					def __repr__(self):
						return "featuresbasic/features5/line_dl_broken.png"

				class DownRight:
					def __repr__(self):
						return "featuresbasic/features5/line_dr_broken.png"

				def __repr__(self):
					return "featuresbasic/features5/line_vertical_broken.png"

			def __repr__(self):
				return "featuresbasic/features5/line_vertical.png"

		class Dot:
			class Large:
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
			class Center:
				def __repr__(self):
					return "featuresbasic/features5/diamond_center.png"

			class Solid:
				def __repr__(self):
					return "featuresbasic/features5/diamond_full.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features5/diamond_small.png"

			def __repr__(self):
				return "featuresbasic/features5/diamond.png"

		class Curve:
			class Up:
				def __repr__(self):
					return "featuresbasic/features5/curve_up.png"

			class Down:
				def __repr__(self):
					return "featuresbasic/features5/curve_down.png"

			class Left:
				def __repr__(self):
					return "featuresbasic/features5/curve_left.png"

			class Right:
				def __repr__(self):
					return "featuresbasic/features5/curve_right.png"

			def __repr__(self):
				return "featuresbasic/features5/curve_up.png"

		class Circle:
			class Large:
				class Filled:
					def __repr__(self):
						return "featuresbasic/features5/curve_right.png"

				def __repr__(self):
					return "featuresbasic/features5/circle_large.png"

			def __repr__(self):
				return "featuresbasic/features5/circle_small.png"

		class X:
			class Large:
				def __repr__(self):
					return "featuresbasic/features5/x_large.png"

			def __repr__(self):
				return "featuresbasic/features5/x.png"

		class Cross:
			class Hollow:
				def __repr__(self):
					return "featuresbasic/features5/cross_hollow.png"

			def __repr__(self):
				return "featuresbasic/features5/cross_solid.png"

	class Seven:
		class Arrow:
			class Up:
				def __repr__(self):
					return "featuresbasic/features7/arrow_up.png"

			class Down:
				def __repr__(self):
					return "featuresbasic/features7/arrow_down.png"

			class Left:
				def __repr__(self):
					return "featuresbasic/features7/arrow_left.png"

			class Right:
				def __repr__(self):
					return "featuresbasic/features7/arrow_right.png"

			def __repr__(self):
				return "featuresbasic/features7/arrow_up.png"

		class Curve:
			class Up:
				def __repr__(self):
					return "featuresbasic/features7/curve_up.png"

			class Down:
				def __repr__(self):
					return "featuresbasic/features7/curve_down.png"

			class Left:
				def __repr__(self):
					return "featuresbasic/features7/curve_left.png"

			class Right:
				def __repr__(self):
					return "featuresbasic/features7/curve_right.png"

			def __repr__(self):
				return "featuresbasic/features7/curve_up.png"

		class Line:
			class Across:
				def __repr__(self):
					return "featuresbasic/features7/line_across.png"

			class Up:
				def __repr__(self):
					return "featuresbasic/features7/line_up.png"

			class DownLeft:
				def __repr__(self):
					return "featuresbasic/features7/line_dl.png"

			class DownRight:
				def __repr__(self):
					return "featuresbasic/features7/line_dr.png"

			class Triple:
				class DownLeft:
					def __repr__(self):
						return "featuresbasic/features7/line_triple_dl.png"

				class DownRight:
					def __repr__(self):
						return "featuresbasic/features7/line_triple_dr.png"

			class Double:
				class DownLeft:
					def __repr__(self):
						return "featuresbasic/features7/line_double_dl.png"

				class DownRight:
					def __repr__(self):
						return "featuresbasic/features7/line_double_dr.png"

			def __repr__(self):
				return "featuresbasic/features7/line_up.png"

		class Dot:
			def __repr__(self):
				return "featuresbasic/features7/dot.png"

		class Circle:
			class Large:
				class Center:
					def __repr__(self):
						return "featuresbasic/features7/circle_large_center.png"

				def __repr__(self):
					return "featuresbasic/features7/circle_large.png"

			class Medium:
				class Broken:
					def __repr__(self):
						return "featuresbasic/features7/circle_broken.png"

				class Center:
					def __repr__(self):
						return "featuresbasic/features7/circle_med_center.png"

				def __repr__(self):
					return "featuresbasic/features7/circle_med.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/circle_small.png"

			class ExtraSmall:
				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/dot.png"

		class Cross:
			class Large:
				class Outline:
					class Round:
						def __repr__(self):
							return "featuresbasic/features7/cross_outline_round.png"

					def __repr__(self):
						return "featuresbasic/features7/cross_outline.png"

				def __repr__(self):
					return "featuresbasic/features7/cross_large.png"

			class Medium:
				class Outline:
					def __repr__(self):
						return "featuresbasic/features7/cross_outline_small.png"

				def __repr__(self):
					return "featuresbasic/features7/cross_med.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/cross_small.png"

			class ExtraSmall:
				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/cross_med.png"

		class X:
			class Large:
				def __repr__(self):
					return "featuresbasic/features7/x_large.png"

			class Medium:
				def __repr__(self):
					return "featuresbasic/features7/x_med.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/x_small.png"

			def __repr__(self):
				return "featuresbasic/features7/x_med.png"

		class Square:
			class Large:
				def __repr__(self):
					return "featuresbasic/features7/square_large.png"

			class Medium:
				def __repr__(self):
					return "featuresbasic/features7/square_small.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/square_small.png"

		class Spiral:
			class CW:
				def __repr__(self):
					return "featuresbasic/features7/spiral_cw.png"

			class CCW:
				def __repr__(self):
					return "featuresbasic/features7/spiral_ccw.png"

			def __repr__(self):
				return "featuresbasic/features7/spiral_cw.png"

		class Diamond:
			class Large:
				class Center:
					def __repr__(self):
						return "featuresbasic/features7/diamond_large_center.png"

				def __repr__(self):
					return "featuresbasic/features7/diamond_large.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/diamond_small.png"

			def __repr__(self):
				return "featuresbasic/features7/diamond_small.png"

		class Donut:
			class Large:
				def __repr__(self):
					return "featuresbasic/features7/donut_large.png"

			class Medium:
				def __repr__(self):
					return "featuresbasic/features7/donut_med.png"

			class Small:
				def __repr__(self):
					return "featuresbasic/features7/donut_small.png"

			class ExtraSmall:
				def __repr__(self):
					return "featuresbasic/features7/dot.png"

			def __repr__(self):
				return "featuresbasic/features7/donut_small.png"

		class Edges:
			class Vertical:
				class Small:
					def __repr__(self):
						return "featuresbasic/features7/edges_vert_small.png"

				def __repr__(self):
					return "featuresbasic/features7/edges_vert.png"

			class Sides:
				class Small:
					def __repr__(self):
						return "featuresbasic/features7/edges_sides_small.png"

				def __repr__(self):
					return "featuresbasic/features7/edges_sides.png"

			def __repr__(self):
				return "featuresbasic/features7/edges_sides.png"

	class Color5:
		class Horiz:
			class RToL:
				def __repr__(self):
					return "featuresbasic/color5/horiz.png"

			class LToR:
				def __repr__(self):
					return "featuresbasic/color5/r_horiz.png"

			def __repr__(self):
				return "featuresbasic/color5/horiz.png"

		class Vert:
			class TToB:
				def __repr__(self):
					return "featuresbasic/color5/r_vert.png"

			class BToT:
				def __repr__(self):
					return "featuresbasic/color5/vert.png"

			def __repr__(self):
				return "featuresbasic/color5/r_vert.png"

	class Color9:
		class CW:
			class RedTop:
				def __repr__(self):
					return "featuresbasic/color9/cw_r_t.png"

			class RedRight:
				def __repr__(self):
					return "featuresbasic/color9/cw_r_r.png"

			class RedLeft:
				def __repr__(self):
					return "featuresbasic/color9/cw_r_l.png"

			class RedBottom:
				def __repr__(self):
					return "featuresbasic/color9/cw_r_b.png"

			def __repr__(self):
				return "featuresbasic/color9/cw_r_t.png"

		class CCW:
			class RedTop:
				def __repr__(self):
					return "featuresbasic/color9/ccw_r_t.png"

			class RedRight:
				def __repr__(self):
					return "featuresbasic/color9/ccw_r_r.png"

			class RedLeft:
				def __repr__(self):
					return "featuresbasic/color9/ccw_r_l.png"

			class RedBottom:
				def __repr__(self):
					return "featuresbasic/color9/ccw_r_b.png"

			def __repr__(self):
				return "featuresbasic/color9/ccw_r_t.png"


class InvalidArgumentError(Exception):
	def __init__(self, message):
		self.message = message

	def __repr__(self):
		return "InvalidArgumentError: " + self.message


def multi_add(a, b, lists=True):
	if lists:
		return [a + b for a, b in zip(a, b)]
	else:
		return a + b


def multi_div(t, i):  # divides tuple and int
	try:
		return [n / i for n in t]
	except TypeError:
		return t / i


def print2d(array, space=3):
	for row in array:
		for item in row:
			print("~" * (space - len(str(item))), end="")
			print(item, end=" ")
		print()


def compare(a, b, maximum=1, polarity="&", scaled=True, opacity_mode=False, return_tuple=False):  # return_tuple requires inputting lists/ndarrays
	if not ((polarity == "&") or (polarity == "+") or (polarity == "-")):
		raise InvalidArgumentError("Polarity Argument must be \"&\" or \"+\" or \"-\"")
	a_type = type(a)
	b_type = type(b)
	# debug(a, b)
	if (a_type == list or b_type == list) or (a_type == np.ndarray or b_type == np.ndarray):  # and polarity == "&":
		if a_type == np.ndarray:
			a = a.tolist()
			a_type = type(a)
		if b_type == np.ndarray:
			b = b.tolist()
			b_type = type(b)
		if (a_type != list) and (b_type == list):
			a = [a for n in b]
		elif (b_type != list) and (a_type == list):
			b = [b for n in a]
		elif (a_type != list) and (b_type != list):
			a = [a]
			b = [b]
		if opacity_mode is True:
			try:
				o = (list(longer(a, b))[3]/maximum + 1) / 2
			except IndexError:
				o = 1
			l = larger(len(a), len(b)) - 1
		else:
			o = 1
			l = larger(len(a), len(b))
		try:
			if not return_tuple:
				# debug(a, b)
				out = [(b[x] - a[x]) for x in range(l)]
				absolute = [1 - abs(outx/maximum) for outx in out]
				if scaled is True:
					return sum([absx * maximum for absx in absolute])/l
				elif scaled is False:
					return sum(absolute)/l
				else:
					return None
			elif return_tuple:
				# debug(a, b)
				out = [(b[x] - a[x]) for x in range(l)]
				absolute = [1 - abs(outx/maximum) for outx in out]
				if scaled is True:
					return [absx * maximum for absx in absolute]
				elif scaled is False:
					return absolute
				else:
					return None
			else:
				print("<!> warning - incorrect input!")
		except ZeroDivisionError or ValueError:
			return None
	elif (a_type == str) or (b_type == str):  # and polarity == "&":
		a = float(a)
		b = float(b)
		try:
			absolute = 1 - abs((b - a) / maximum)
			if scaled is True:
				return absolute * maximum
			elif scaled is False:
				return absolute
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	elif ((a_type == int) or (b_type == int)) or ((a_type == float) or (b_type == float)):  # and polarity == "&":
		try:
			absolute = 1 - abs((b - a)/maximum)
			if scaled is True:
				return absolute * maximum
			elif scaled is False:
				return absolute
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	else:
		raise InvalidArgumentError("Input is neither an int, float, list, numpy array, or string convertable to int or float. compare() cannot parse this unknown type [ " + str(type(a)) + " | " + str(type(b)) + " ]")
	'''
	# so idk what these even did, and i've never used them afaik, and i wrote it a year ago. lazy coding ftw
	elif (a_type == list or b_type == list) or (a_type == np.ndarray or b_type == np.ndarray) and polarity == "+":
		if a_type == np.ndarray:
			a = a.tolist()
		if b_type == np.ndarray:
			b = b.tolist()
		if (a_type != list) and (b_type == list):
			a = [a for n in b]
		elif (b_type != list) and (a_type == list):
			b = [b for n in a]
		elif (a_type != list) and (b_type != list):
			a = [a]
			b = [b]
		if opacity_mode is True:
			try:
				o = (list(longer(a, b))[3]/maximum + 1) / 2
			except IndexError:
				o = 1
			l = larger(len(a), len(b)) - 1
		else:
			o = 1
			l = larger(len(a), len(b))
		try:
			out = [(a[x] - b[x]) for x in range(l)]
			absolute = [1 - (abs(outx)/maximum) for outx in out]
			if scaled is True:
				return (sum([absx * maximum for absx in absolute])/l + maximum) / 2
			elif scaled is False:
				return (sum(absolute)/l + 1) / 2
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	elif (a_type == list or b_type == list) or (a_type == np.ndarray or b_type == np.ndarray) and polarity == "-":
		if a_type == np.ndarray:
			a = a.tolist()
		if b_type == np.ndarray:
			b = b.tolist()
		if (a_type != list) and (b_type == list):
			a = [a for n in b]
		elif (b_type != list) and (a_type == list):
			b = [b for n in a]
		elif (a_type != list) and (b_type != list):
			a = [a]
			b = [b]
		if opacity_mode is True:
			try:
				o = (list(longer(a, b))[3]/maximum + 1) / 2
			except IndexError:
				o = 1
			l = larger(len(a), len(b)) - 1
		else:
			o = 1
			l = larger(len(a), len(b))
		try:
			out = [(a[x] - b[x]) for x in range(l)]
			absolute = [-(1 - (-abs(outx)/maximum)) for outx in out]
			if scaled is True:
				return (sum([absx * maximum for absx in absolute])/l + maximum) / 2
			elif scaled is False:
				return (sum(absolute)/l + 1) / 2
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	elif ((a_type == str) or (b_type == str)) and polarity == "+":
		a = float(a)
		b = float(b)
		try:
			absolute = 1 - (abs(a - b) / maximum)
			if scaled is True:
				return absolute * maximum
			elif scaled is False:
				return absolute
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	elif ((a_type == str) or (b_type == str)) and polarity == "-":
		a = float(a)
		b = float(b)
		try:
			absolute = -(1 - (-abs(a - b) / maximum))
			if scaled is True:
				return absolute * maximum
			elif scaled is False:
				return absolute
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	elif (((a_type == int) or (b_type == int)) or ((a_type == float) or (b_type == float))) and polarity == "+":
		try:
			absolute = 1 - (abs(a - b) / maximum)
			if scaled is True:
				return absolute * maximum
			elif scaled is False:
				return absolute
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	elif (((a_type == int) or (b_type == int)) or ((a_type == float) or (b_type == float))) and polarity == "-":
		try:
			absolute = -(1 - (-abs(a - b) / maximum))
			if scaled is True:
				return absolute * maximum
			elif scaled is False:
				return absolute
			else:
				return None
		except ZeroDivisionError or ValueError:
			return None
	# '''


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
				# print(array.shape)
				size = amount, amount, 1
			# print2d(np.kron(array, np.ones(size, dtype="float")))
			return np.kron(array, np.ones(size, dtype="float"))


def remap(array):
	min = np.amin(array)
	max = np.amax(array)
	# print(array)
	try:
		dif = 1 / (max - min)
	except RuntimeWarning:
		dif = 1
	out = np.multiply(np.subtract(array, min), dif)
	# print(out)
	return out


def larger(a, b):
	if a > b:
		return a
	else:
		return b


def longer(a, b):
	if len(a) > len(b):
		return a
	else:
		return b


def bounds(i, l, u):
	if i > u:
		i = u
	elif i < l:
		i = l
	else:
		pass
	return i


def save(array, name):
	# print2d(array)
	name = name + ".png"
	misc.imsave(name, array)


def show_image(ndarray):
	plt.imshow(ndarray, interpolation='nearest')
	plt.show()


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
	def __init__(self, input_image, multichannel=False, opacity=False):
		self.multichannel = multichannel
		self.opacity = opacity
		self.main_image = ImageMatrix(input_image, multichannel, opacity)
		self.name = str(input_image).split(".")[0]
		self.number = 0
		self.resize = 1
		self.size = self.main_image.a * self.main_image.b

	def scale(self, value):
		self.resize = value

	def convolve(self, input_feature, minimum_value=0, resize=1, append_number=False, view=True, mono_out=False):  # mono_out does nothing without multichannel
		append = self.name + "_" + str(input_feature).split("/")[-1].split(".")[0]
		self.number += 1
		if minimum_value > 0:
			min_toggle = True
		else:
			min_toggle = False
		if append_number is True:
			append += "_" + str(self.number)
		if resize == 1:
			resize = self.resize
		convolve_feature = FeatureMatrix(input_feature, self.multichannel, self.opacity)
		convolve_size = convolve_feature.a * convolve_feature.a
		if (self.size * convolve_size) <= 10000:
			progress_bar = False
		else:
			progress_bar = True
		# print(progress_bar, self.size, convolve_size)
		out = self.main_image.convolve(convolve_feature, minimum_value, min_toggle, mono_out, progress_bar=progress_bar)
		if not view:
			# print(out)
			save(enlarge(remap(out), int(resize)), append)
		elif view:
			# print(out)
			show_image(enlarge(remap(out), int(resize)))


'''
o_large = Convolution("test images/input_o_large.png", False, False)
o_large.scale(4)
o_large.convolve(Features.Three.Line.Across(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Line.Up(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Line.DownLeft(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Line.DownRight(), 0, 1, False, view=False)
o_large.convolve(Features.Three.X(), 0, 1, False, view=False)
o_large.convolve(Features.Three.Dot(), 0, 1, False, view=False)
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
# '''

'''
input2 = Convolution("test images/color_tester.png", True, False)
# input2.convolve(Features.Color9.CW.RedTop(), 0, 1, False, view=True)
input2.convolve(Features.Color9.CW.RedTop(), 0, 1, False, view=True, mono_out=True)
input2.convolve(Features.Color9.CW.RedRight(), 0, 1, False, view=True, mono_out=True)
input2.convolve(Features.Color9.CW.RedBottom(), 0, 1, False, view=True, mono_out=True)
input2.convolve(Features.Color9.CW.RedLeft(), 0, 1, False, view=True, mono_out=True)
# input2.convolve(Features.Color9.CCW.RedTop(), 0, 1, False, view=True)
# input2.convolve(Features.Color9.CCW.RedBottom(), 0, 1, False, view=True)
# '''

'''
input2 = Convolution("test images/input2 xsmall.png", False, False)
# input2.convolve(Features.Seven.Diamond.Large(), 0, 1, False, view=True)
# input2.convolve(Features.Seven.Donut.Large(), 0, 1, False, view=True)
input2.convolve(Features.Seven.X.Large(), 0, 1, False, view=True)
# input2.convolve(Features.Seven.Cross.Large(), 0, 1, False, view=True)
# '''
