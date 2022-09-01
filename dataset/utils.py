import os


class IndexToString(object):
	"""Convert index to the corrisponding string
	Args:
		file_path (string): path to txt file containing class ids (0 indexed) and strings
	"""

	def __init__(self, file_path):
		self.string_dict = {}
		with open(file_path) as f:
			lines = f.readlines()
			for i in lines:
				i = i.split(" ")
				self.string_dict[int(i[1])] = i[0]

	def __call__(self, index):
		"""
		Args:
			index (int): index of class
		Returns:
			String: String name
		"""
		return self.string_dict[index]


class StringToIndex(object):
	"""Convert string to the corrisponding index
	Args:
		file_path (string): path to txt file containing class ids (0 indexed) and strings
	"""

	def __init__(self, file_path):
		self.string_dict = {}
		with open(file_path) as f:
			lines = f.readlines()
			for i in lines:
				i = i.split(" ")
				self.string_dict[i[0]] = int(i[1])

	def __call__(self, string):
		"""
		Args:
			string (str): string of index
		Returns:
			int: index
		"""
		return self.string_dict[string]
