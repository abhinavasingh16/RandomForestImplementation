class Node:
	'''
	This is a node class which keeps track of its children. 
	It learns a split rule and also becomes a label if 
	it is a leaf.
	'''
	def __init__(self,lc=None,rc=None,lbl=None):
		self.lchild = lc 
		self.rchild = rc
		self.label = lbl
		self.split_rule = (None,None)
