import sys,re
import nltk
from collections import defaultdict
import cfg_fix
from cfg_fix import parse_grammar, CFG
from pprint import pprint
# The printing and tracing functionality is in a separate file in order
#  to make this file easier to read
from cky_print import CKY_pprint, CKY_log, Cell__str__, Cell_str, Cell_log

class CKY:
    """An implementation of the Cocke-Kasami-Younger (bottom-up) CFG recogniser.

    Goes beyond strict CKY's insistance on Chomsky Normal Form.
    It allows arbitrary unary productions, not just NT->T
    ones, that is X -> Y with either Y -> A B or Y -> Z .
    It also allows mixed binary productions, that is NT -> NT T or -> T NT"""

    def __init__(self,grammar):
        '''Create an extended CKY processor for a particular grammar

        Grammar is an NLTK CFG
        consisting of unary and binary rules (no empty rules,
        no more than two symbols on the right-hand side

        (We use "symbol" throughout this code to refer to _either_ a string or
        an nltk.grammar.Nonterminal, that is, the two thinegs we find in
        nltk.grammar.Production)

        :type grammar: nltk.grammar.CFG, as fixed by cfg_fix
        :param grammar: A context-free grammar'''

        self.verbose=False
        assert(isinstance(grammar,CFG))
        self.grammar=grammar
        # split and index the grammar
        self.buildIndices(grammar.productions())

    def buildIndices(self,productions):
       '''This function converts the list of productions - i.e. CFG rules - into two dictionaries. One containing the Unary rules, and one containing the Binary rules. The data is stored such that each key in the dictionary corresponds with the right hand side of some rule, and the value in the dictionary is the left hand side. 

The function takes as its argument the productions in the grammar, which is a list of rules, and returns nothing. The dictionaries are defined and updated here as class variables, which are accessible to all methods in the class.'''

       self.unary=defaultdict(list) # initialise dictionary of unary rules
       self.binary=defaultdict(list) # initialise dictionary of binary rules
       for production in productions:
           rhs=production.rhs()
           lhs=production.lhs()
           assert(len(rhs)>0 and len(rhs)<=2)
           if len(rhs)==1:
               self.unary[rhs[0]].append(lhs) # assign unary rules to unary dict
           else:
               self.binary[rhs].append(lhs) # assign binary rules to binary dict

    def recognise(self,tokens,verbose=False):
       '''This function initialises a matrix from the sentence - which is passed in as the argument 'tokens', consisting of a list of words - and then runs the CKY algorithm over it. It does this by first filling in the words on the diagonal with the unaryFill method, which first adds the word label to the cell, and then checks to see if there are any possible unary updates which can be done from there. Once the diagonal is filled in, the parser procedes systematically across the upper diagonals, checking for each whether there are any possible labels it can add. It does this by calling the binaryScan function.

Once the parse if complete, the function checks if it was successful by checking the upper right cell. If the parse was successful, there should be the label S in that cell. If there were multiple possible parses, there should be multiple Ss.

If the parse has failed, and there are 0 Ss in the upper right corner, the function returns False. If there are some Ss in the upper right corner, the function returns the number of Ss - i.e. the number of possible parses found.

There is also an optional verbose argument controls debugging output, defaults to False '''
       self.verbose=verbose
       self.words = tokens
       self.n = len(self.words)+1
       self.matrix = []
        # We index by row, then column
        #  So Y below is 1,2 and Z is 0,3
        #    1   2   3  ...
        # 0  X   X   Z
        # 1      Y   X
        # 2          X
        # ...
       for r in range(self.n-1):
             # rows
            row=[]
            for c in range(self.n):
                 # columns
                if c>r:
                     # This is one we care about, add a cell
                     row.append(Cell(r,c,self))
                else:
                     # just a filler
                    row.append(None)
            self.matrix.append(row)
       self.unaryFill() # fill in the main diagonals with the words.
       self.binaryScan()# crawl through the rest of the upper diagonals
       print self.n
	#return self.matrix[0][self.n-1]
       self.first_tree() # build the parse tree and display it.
        # Replace the line below for Q6
	
       length =  len(self.matrix[0][self.n-1].labels())
       if length == 0:
	       return False
       else:
	       return length
	#this is q 6 done!	


    def unaryFill(self):
        '''This function fills in the diagonal row of the matrix. For each cell along the diagonal, it first initialises a cell for that position in the matrix, then adds the corresponding word of the sentence, and then calls unaryUpdate to see if any possible further labels can be added to the cell. 

This function takes no arguments except the class instance as all variables it needs are class variables. It returns no values because it updates all variables in place.'''
        for r in range(self.n-1):
            cell=self.matrix[r][r+1]
            word=self.words[r] # Adds the words as labels to the diagonal
	    cell.unaryUpdate(word) #calls unaryUpdate to see if any further advances can be made

   

    def binaryScan(self):
        '''The heart of the implementation:
            proceed across the upper-right diagonals
            in increasing order of constituent length'''
        for span in xrange(2, self.n):
            for start in xrange(self.n-span):
                end = start + span
                for mid in xrange(start+1, end):
                    self.maybeBuild(start, mid, end)
		    #print "scanned"

    def maybeBuild(self, start, mid, end):
        ''' The binary scan function considers every possible partition for each cell across the upper right diagoonals of the matrix. For each combination of partitions and cells it calls this function. This function checks if within the labels of each partition there are two labels which correspond to both parts of a binary rule. If there are then the rhs of that rule is added as a label to the current cell. Then unary update is called to check if there are any additional advances we can make using the new labels.

The function takes as its arguments the start mid and end positions defining the partitions. It returns nothing because all variables it updates are persistent class variables'''
        self.log("%s--%s--%s:",start, mid, end)
        cell=self.matrix[start][end]
        for s1 in self.matrix[start][mid].labels():
	    cell1 = self.matrix[start][mid] # remember the cell s1 came from
	    #print str(s1) + "  s1 " + str(start) + " " + str(mid)
            for s2 in self.matrix[mid][end].labels():
		cell2 = self.matrix[mid][end] # remember the cell the s2 came from.
		#print str(s2) + "  s2 " + str(mid) + " " + str(end)
                if (s1,s2) in self.binary:
                    for s in self.binary[(s1,s2)]:
                        self.log("%s -> %s %s", s, s1, s2, indent=1)
                        cell.addLabelBinary(s,s1,s2,cell1,cell2) # we need to add the cells here too, so we have sufficient backtrace information to build the tree.
                        cell.unaryUpdate(s,1)


    def first_tree(self):
	''' Once a parse has been reached, this function will compute the tree representing the parse and draw it using the nltk tree drawing package. It takes the labels of the cell in the upper diagonal, which if there was a successful parse will contain an S label. This function initialises the tree string and adds the S label to it, then calls the do_backtraces function which will pursue the backtraces of the S label and construct the tree string from there. '''
	newdict = self.matrix[0][self.n-1]._labels
	symbol = newdict.keys()[0] # get start symbol
	s = "( " + str(symbol) # add start symbol to tree string
	s = self.do_backtraces(s,symbol,newdict) # call this recursive function to pursue the backtrace
	s += " )" # add closing bracket
	print s # This can easily be commented out if it gets distracting.
	tree = nltk.tree.Tree.fromstring(s)
	tree.draw() # comment this out if the tree popups are annoying.
	return s

    def do_backtraces(self,s,symbol,newdict):
	'''This function pursues the backtraces back through the chart. When given a dictionary consisting of keys which are the labels in the current cell, and values which are lists of nonterminal symbols and their associated backtrace cells, it will pursue each path recursively, then passing on to the append_stuff function to add the correct information to the tree string. '''
	for key in newdict.keys():
		if key == symbol: # to ensure we're following the right backtrace path!
			dict2 = newdict[key]
			if type(dict2[0]) == type([""]): # this occurs when we're pursuing a binary rule; there are multiple options, and we must take both.
				for l in dict2:
					s+=self.append_stuff(l,s) 

			else:
				s+=self.append_stuff(dict2,s)
	return s

    def append_stuff(self,list1,s):
	'''This parses a list and adds the correct information to the tree string. The lists passedto this function by the do_backtraces function are of the form [nonterminal symbol, backtraced cell]. '''
	cell = list1[1]._labels # this gets us the backtraced cell's dictionary of labels. 
	symbol = list1[0] # get the correct nonterminal and then add it to the treestring
	s = "( " + str(symbol)
	if type(symbol) == type(" "):#And hence is a terminal node. We want to terminate here and not recurse any further.
		s += " )"
		return s 
	s = self.do_backtraces(s,symbol,cell) # recurse again to the new backtraced cell.
	s += " )" # remember to close the bracket!
	return s


CKY.pprint=CKY_pprint
CKY.log=CKY_log

class Cell:
    '''A cell in a CKY matrix'''
    def __init__(self,row,column,matrix):
        self._row=row
        self._column=column
        self.matrix=matrix
        self._labels={} # this has now become a dictionary, with the format {label : [nonterminal, backtraced cell]}. This is so we have sufficient backtrace information to construct the parse tree.

    def addLabel(self,label):
	#old function for Q9.
	if label not in self._labels.keys():
        	self._labels.append(label)
		self.unaryUpdate(label)

    def addLabelUnary(self,label,p1):
	'''We've now added backtrace information to the rules, albeit here it isn't very interesting because the backtraced cell is just this cell as the rule is unary. '''
	if label not in self._labels.keys():
		self._labels[label] = [p1,self]
		self.unaryUpdate(label)


    def addLabelBinary(self, label, p1, p2,cell1, cell2):
	'''This new addLabelBinary function constructs the new label augmented with backtrace information in a dictionary format. The format, when the label is derived from a given CFG rule is: {rhs: [lhs[0],cell of lhs[0]], [lhs[1], cell of lhs[1]]. As such the originator nonterminals and the cells where they reside of any given label is stored with it, so it can be easily followed up. ''' 
	self._labels[label] = [[p1,cell1],[p2,cell2]] # adds the backtraced cells also to a list.
	self.unaryUpdate(label) # in case there are any.
	


    def labels(self):
        return self._labels

    def unaryUpdate(self,symbol,depth=0,recursive=False):
        ''' This function checks whether, for a cell in the matrix, and a symbol (which is assumed to be a label of the cell) whether there are any unary rules that can be applied to the symbol and, if there are, to add the results of these rules as a label to the cell. The function is recursive so that multiple labels can be added in one pass if that is possible. This is often the case. For instance, consider the series of rules: 'book' -> Nsc -> N2sc ->NP0 -> NP1 -> NP. All of these rules are unary, and so should be generated in the same cell in one pass. unaryUpdate achieves this. 

The function takes as its argument a symbol which forms the lhs of the unary rules checked in this function, and a depth and recursive argument used to help format the final chart.'''
        if not recursive:
            self.log(str(symbol),indent=depth)
        if symbol in self.matrix.unary:# check if symbol is the lhs of any rule
           for parent in self.matrix.unary[symbol]: # for each element of rhs
               self.matrix.log("%s -> %s",parent,symbol,indent=depth+1)
               self.addLabelUnary(parent, symbol) # This now goes to the new and improved label adding method.
               self.unaryUpdate(parent,depth+1,True)  #and recurse to check if the new label can be extended further.

# helper methods from cky_print
Cell.__str__=Cell__str__
Cell.str=Cell_str
Cell.log=Cell_log

class Label:
    '''A label for a substring in a CKY chart Cell

    Includes a terminal or non-terminal symbol, possibly other
    information.  Add more to this docstring when you start using this
    class'''
    def __init__(self,symbol,
                 # Fill in here, if more needed
                 ):
        '''Create a label from a symbol and ...
        :type symbol: a string (for terminals) or an nltk.grammar.Nonterminal
        :param symbol: a terminal or non-terminal
        '''
        self._symbol=symbol
        # augment as appropriate, with comments

    def __str__(self):
        return str(self._symbol)

    def __eq__(self,other):
        '''How to test for equality -- other must be a label,
        and symbols have to be equal'''
        assert isinstance(other,Label)
        return self._symbol==other._symbol

    def symbol(self):
        return self._symbol
    # Add more methods as required, with docstring and comments

