import math
import operator
from Node import Node 

def col_range(d):
	col_ranges = {}
	for column in range(len(d[0])):
	    for row in d:
	        if column in col_ranges:
	            col_ranges[column].add(row[column])
	        else:
	            col_ranges[column] = set([row[column]])
	return col_ranges

def entropy(hist):
	'''
	Calculates the entropy of a function.
	'''
	entropy = 0
	total = sum(hist.values())
	for label in hist.keys():
		probability = float(hist[label])/float(total)
		factor = (-probability) * math.log(probability,2)
		entropy+=factor
	return entropy

def merge(h1,h2):
	'''
	Merges 2 histogram dictionaries.
	'''
	for key in h2.keys():
		if key in h1:
			h1[key]+=h2[key]
		else:
			h1[key] = h2[key]
	return h1

def compute_c2(lhist,rhist):
	parent_hist = merge(lhist,rhist)
	total = sum(parent_hist.values())
	count = 0 
	for label in lhist.keys():
		nil = lhist[label]
		nie = float(parent_hist[label])/float(total)
		count+=float((nil-nie)**2)/float(nie)
	return count

class Decision_Tree:
	def __init__(self,max_depth=10):
		'''
		Initializes the decision tree object.
		'''
		self.root = None
		self.size = 1
		self.height = 1
		self.max_depth = max_depth

	def impurity(self,l_hist,r_hist):
		'''
		This function takes the entropy of kids and 
		averages them and subtracts them from the 
		parent entropy. We want the min.
		'''
		import copy
		temp_l_hist = copy.deepcopy(l_hist)
		parent_hist = merge(temp_l_hist,r_hist)
		parent_entropy = entropy(parent_hist)
		# print("\t\t\tTime to test the impurity of l:{0} and r:{1} and p:{2}.".format(l_hist,r_hist,parent_hist))
		# print("\t\t\tThe parent entropy is: {0}".format(parent_entropy))
		left_entropy = entropy(l_hist)
		right_entropy = entropy(r_hist)
		# print("\t\t\tThe children entropies are: {0},{1}".format(left_entropy,right_entropy))
		split_average = (sum(l_hist.values())*left_entropy) + (sum(r_hist.values())*right_entropy)
		split_average = float(split_average)/float(sum(parent_hist.values()))
		# print("\t\t\tThe resulting infromation gain is: {0}".format(parent_entropy-split_average))
		return parent_entropy-split_average

	def ideal_threshold(self,data,labels,feature):
		'''
		Assuming we split on a feature what is the split threshold?
		'''
		col_ranges = col_range(data)
		possible_thresholds = col_ranges[feature]
		information_gain = {}
		# print("\t\tThe possible thresholds are: {0}".format(possible_thresholds))
		# print("\t\tThe length of the data is {0}".format(len(data)))
		for thresh in possible_thresholds:
			#Given a thresh rule, find information gain
			lhist = {}
			rhist = {}
			for example in range(len(data)):
				value = data[example][feature]
				label = labels[example]
				if value <= thresh:
					if label not in lhist:
						lhist[label] = 1
					else:
						lhist[label]+=1
				else:
					if label not in rhist:
						rhist[label] = 1
					else:
						rhist[label]+=1
			# print("\t\tThe left histogram is: {0}".format(lhist))
			# print("\t\tThe right histogram is: {0}".format(rhist))
			# print("\t\tFor threshold {0}, we found that the split was: {1} on the left and {2} on the right".format(thresh,sum(lhist.values()),sum(rhist.values())))
			if len(lhist.keys()) != 0 and len(rhist.keys()) != 0:
				#If the thresh actually splits
				information_gain[thresh] = self.impurity(lhist,rhist)
		if len(information_gain.keys()) == 0:
			#If all thresholds don't split anything
			return None,None
		else:
			min_thresh = max(information_gain.iteritems(), key=operator.itemgetter(1))
			# print("\t\tThe min thresh and its impurity is: {0}".format(min_thresh))
			# print("")
			return min_thresh[0],min_thresh[1]

	def segment(self,data,labels):
		'''
		We have a set of training data and labels. We want to find 
		the ideal feature to split on and also find the ideal 
		threshold to split on. So for each feature we get the optimal 
		threshold that gets us the most information and then we 
		return the feature that gets us the most information. 
		'''
		label_set = set(labels)
		information_gain = {}
		for feature in range(len(data[0])):
			#Assuming this is the feature we split on, what is the ideal split threshold and the impurity it gets us?
			#print("\tTime to find the ideal threshold.")
			threshold,imp = self.ideal_threshold(data,labels,feature)
			#If no thresholds split anything then do nothing; otherwise.
			if threshold != None:
				information_gain[feature] = (imp,threshold)
		#Pick feature with the most information gain
		if len(information_gain.keys()) == 0:
			#If it is impossible to split the data in anyway, then you are at a leaf.
			return None,None
		else:
			min_feature_entry = max(information_gain.iteritems(), key=operator.itemgetter(1))
			return min_feature_entry[0],min_feature_entry[1][1]
	
	def isLeaf(self,node,labels,split_feature):
		#If it is impossible to split the data in anyway you are at a leaf.
		if split_feature == None:
			return True
		#If we have a clean split then stop
		elif len(list(set(labels))) == 1:
			return True
		#If we have reached our max height then stop.
		elif self.height == self.max_depth:
			return True
		else:
			return False

	def train_helper(self,node,data,labels):
		'''
		Construct a tree recursively.
		'''
		#print("Time to train with data length: {0}".format(len(data)))
		split_feature,threshold = self.segment(data,labels)
		if self.isLeaf(node,labels,split_feature):
			#Handle Leaf by setting label equal to the most frequent label in the labels list
			node.label = max(set(labels), key=labels.count)
			return node
		else:
			node.split_rule = (split_feature,threshold)
			#Otherwise perform split
			ldata = []
			rdata = []
			llabel = []
			rlabel = []
			#For each training example
			for example in range(len(data)):
				#Get the answer and segment based on split rule
				value = data[example][split_feature]
				if value <= threshold:
					ldata.append(data[example])
					llabel.append(labels[example])
				else:
					rdata.append(data[example])
					rlabel.append(labels[example])
			#Increment the height and let the children learn
			self.height+=1
			#print("The split is {0} on the left and {1} on the right. Left Train time.".format(len(ldata),len(rdata)))
			node.lchild = self.train_helper(Node(),ldata,llabel)
			#print("Trained the left, time to the train the right")
			node.rchild = self.train_helper(Node(),rdata,rlabel)
			return node

	def train(self,data,labels):
		print("Starting amount of training data is: {0}".format(len(data)))
		print("")
		self.root = self.train_helper(Node(),data,labels)

	def prediction_helper(self,node,vector):
		print("The vector is {0}".format(vector))
		print("The node is {0}".format(node))
		if node.label != None:
			print("WE ARE DONE")
			print("")
			return node.label
		else:
			feature = node.split_rule[0]
			threshold = node.split_rule[1]
			if vector[feature] <= threshold:
				print("Move left")
				return self.prediction_helper(node.lchild,vector)
			else:
				print("Move right")
				return self.prediction_helper(node.rchild,vector)

	def predict(self,vector):
		return self.prediction_helper(self.root,vector)

	def print_tree(self,node):
		if node:
			print node
			self.print_tree(node.lchild)
			self.print_tree(node.rchild)