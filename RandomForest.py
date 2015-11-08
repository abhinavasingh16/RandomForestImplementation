import math
import operator
import stats
import sys
import random
from random import shuffle
import DecisionTree

class RandomForrest:
	def __init__(self,num_trees):
		self.num_trees = num_trees
		self.forrest = []

	def train(self,data,labels):
		#Set up bagging
		num_subsets = self.num_trees
		dataset_groups = [[] for i in range(num_subsets)]
		label_groups = [[] for i in range(num_subsets)]
		self.forrest = [DecisionTree() for i in range(num_subsets)]
		random_indices = shuffle(range(len(data)))
		for index in random_indices:
			self.forrest.append(DecisionTree())
			dataset_groups[index%num_subsets].append(data[index])
			label_groups[index%num_subsets].append(labels[index])
		#Train forrest
		for tree_index in range(len(self.forrest)):
			self.forrest[tree_index].train(dataset_groups[tree_index],label_groups[tree_index])

	def predict(self,vector):
		votes = []
		for tree in self.forrest:
			votes.append(tree.predict(vector))
		return max(set(votes), key=votes.count)