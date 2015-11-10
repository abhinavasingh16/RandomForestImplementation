import math
import operator
import sys
import random
from random import shuffle
import DecisionTree
from DecisionTree import Decision_Tree

class RandomForrest:
	def __init__(self,num_trees):
		self.num_trees = num_trees
		self.forrest = []

	def train(self,data,labels):
		#Set up bagging
		num_subsets = self.num_trees
		dataset_groups = [[] for i in range(num_subsets)]
		label_groups = [[] for i in range(num_subsets)]
		self.forrest = [Decision_Tree() for i in range(num_subsets)]
		random_indices = range(len(data))
		#print(random_indices)
		shuffle(random_indices)
		for index in random_indices:
			dataset_groups[index%num_subsets].append(data[index])
			label_groups[index%num_subsets].append(labels[index])
		#Train forrest
		for tree_index in range(num_subsets):
			self.forrest[tree_index].train(dataset_groups[tree_index],label_groups[tree_index])

	def predict(self,vector):
		votes = []
		i = 0
		for tree in self.forrest:
			print("Let tree {0} vote.".format(i))
			i+=1
			votes.append(tree.predict(vector))
		return max(set(votes), key=votes.count)