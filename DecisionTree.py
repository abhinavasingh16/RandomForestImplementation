import math
import operator
import stats

def entropy(hist):
	'''
	Calculates the entropy of a function.
	'''
	entropy = 0
	total = sum(hist.values())
	for label in hist.keys():
		probability = float(hist[label])/float(total)
		factor = (-probability) * math.log(probability)
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

class DecisionTree:
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
		parent entropy.
		'''
		parent_hist = merge(l_hist,r_hist)
		parent_entropy = entropy(parent_hist)
		left_entropy = entropy(l_hist)
		right_entropy = entropy(r_hist)
		split_average = (sum(l_hist.values())*left_entropy) + (sum(r_hist.values())*right_entropy)
		split_average = float(split_average)/float(sum(parent_hist.values()))
		return parent_entropy-split_average

	def segment(self,data,labels):
		'''
		Given training data, find the best feature to split on and partition the node.
		'''
		#For each feature index
		label_set = set(labels)
		information_gain = {}
		for feature in range(len(data[0])):
			#Split on feature i which will be forced to be binary.
			for example in range(len(data)):
				lhist = {}
				rhist = {}
				value = data[example][feature]
				label = labels[example]
				if value == 1:
					if label in lhist:
						lhist[label] = 1
					else:
						lhist[label]+=1
				else:
					if label in rhist:
						rhist[label] = 1
					else:
						rhist[label]+=1
			#Once we have the split results, we want to see if that split 
			#is statistically significant relative to a random split
			c2 = compute_c2(lhist,rhist)
			#Only evaluate statistically significant splits.
			if (1 - stats.chi2.cdf(c2, 1))<0.05:
				information_gain[feature] = (impurity(lhist,rhist),ec2)
		#If no statistically significant splits for any feature stop because you have hit a leaf.
		if len(information_gain.keys()) == 0:
			return None,None
		else:
			min_feature_entry = min(information_gain.iteritems(), key=operator.itemgetter(1))
			return min_feature_entry[0],min_feature_entry[1][0]
	
	def isLeaf(self,node,labels,split_feature):
		if split_feature == None:
			return True
		#If we have a clean split then stop
		elif len(list(set(labels))) == 1:
			return True
		#If we have reached our max height then stop.
		elif node.height == self.max_depth:
			return True
		else:
			return False

	def train_helper(self,node,data,labels):
		split_feature,threshold = segment(data,labels)
		node.split_rule((split_feature,threshold))
		#If the stopping criteria is met, trigger base case
		if isLeaf(node,data,labels,threshold,split_feature):
			#Handle Leaf by setting label equal to the most frequent label in the labels list
			node.label = max(set(labels), key=labels.count)
			return node
		else:
			#Handle decision maker
			ldata = []
			rdata = []
			llabel = []
			rlabel = []
			#For each training example
			for example in range(len(data)):
				#Get the answer and segment based on split rule
				value = data[example][split_feature]
				if value = 0:
					ldata.append(data[example])
					llabel.append(labels[example])
				else:
					rdata.append(data[example])
					rlabel.append(labels[example])
			#Increment the height and let the children learn
			self.height+=1
			node.lchild = train_helper(Node(),ldata,llabel)
			node.rchild = train_helper(Node(),rdata,rlabel)
			return node

	def train(self,data,labels):
		 self.root = train_helper(Node(),data,labels)

	def prediction_helper(node,vector):
		if node.label != None:
			return node.label
		else:
			feature = node.split_rule[0]
			threshold = node.split_rule[1]
			if vector[feature] == 0:
				prediction_helper(node.lchild,vector)
			else:
				prediction_helper(node.rchild,vector)

	def predict(self,vector):
		return prediction_helper(self.root,vector)