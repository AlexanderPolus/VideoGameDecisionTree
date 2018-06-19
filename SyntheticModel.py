#-------------------libraries----------------------------------
import csv
import sys
import math
import matplotlib.pyplot as plt
#-------------------initialize my tree---------------------------


tree = []


#------import file into raw data---------------------------------
if (len(sys.argv) != 2):
    print("Usage: Python3 SyntheticModel.py <filename> ")
    sys.exit(0)
filename = sys.argv[-1]
file = open(filename)
reader = csv.reader(file, delimiter=',')
rawdata = list(reader)

print("\nRaw Data: ")
for i in rawdata:
    print(i)
#------organize data into a structured discretized 2D array-----------------
# make raw data into numbers:
for i in range(0,len(rawdata)):
    for j in range(0,len(rawdata[i])):
        rawdata[i][j] = float(rawdata[i][j])

#getting a copy so my plot works
copy_rawdata = rawdata


#------discretize function---------------------------------------
def discretize(data):
    A = []
    B = []
    #make my columns
    for i in range(0,len(data)):
        A.append(data[i][0])
        B.append(data[i][1])
    #process A:
    intervalA = (max(A) - min(A)) / 5.0
    intervalB = (max(B) - min(B)) / 5.0
    print("intervalA = " + str(intervalA))
    print("intervalB = " + str(intervalB))
    #iterate through and change values
    for i in range(0,len(data)):
        #bins for attribute A:
        if (min(A) <= data[i][0] < (min(A) + 1*intervalA)): data[i][0] = 0
        elif ((min(A) + 1*intervalA) <= data[i][0] < (min(A) + 2*intervalA)): data[i][0] = 1
        elif ((min(A) + 2*intervalA) <= data[i][0] < (min(A) + 3*intervalA)): data[i][0] = 2
        elif ((min(A) + 3*intervalA) <= data[i][0] < (min(A) + 4*intervalA)): data[i][0] = 3
        else: data[i][0] = 4
        #bins for attribute B:
        if (min(B) <= data[i][1] < (min(B) + 1*intervalB)): data[i][1] = 0
        elif ((min(B) + 1*intervalB) <= data[i][1] < (min(B) + 2*intervalB)): data[i][1] = 1
        elif ((min(B) + 2*intervalB) <= data[i][1] < (min(B) + 3*intervalB)): data[i][1] = 2
        elif ((min(B) + 3*intervalB) <= data[i][1] < (min(B) + 4*intervalB)): data[i][1] = 3
        else: data[i][1] = 4

    return data

#----------------discretize my data------------------------------
processed_data = discretize(rawdata)
print("\nData after being discretized: ")
print("Processed data rows: ", len(processed_data))
for i in processed_data:
    print(i)
#---------------------new data types-----------------------------
class node: #parent_index, children

    children = {}
    prediction_value = -1
    my_index = -1
    parent_index = None
    majority_class = -1
    feature = None
    def __init__(self):
        self.children = {}

#---------------------majority function--------------------------
def majority(data):
    possibilities = []
    #get my list of labels
    for i in range(0,len(data)):
        possibilities.append(data[i][-1])
    myMap = {}
    maximum = ( '', 0 )
    for n in possibilities:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        if myMap[n] > maximum[1]: maximum = (n,myMap[n])
    return maximum[0];
#---------------------Entropy------------------------------------
def entropy(data): # for the video game set, use the possibilities strategy...
    total = len(data)
    a = 0.0
    b = 0.0
    #calculate distribution of values in class label column
    for i in range(0,len(data)):
        if (data[i][-1] == 1):
            a += 1
        else:
            b += 1
    p = a/total
    q = b/total
    if (p == 0) or (q == 0):
        return 0;
    else:
        return -p*math.log(p,2) - q*math.log(q,2);

#---------------------information gain-----------------------------
def info_gain(parent_data, split_attribute):
    IG = entropy(parent_data)
    #get my attribute possibilities for weighted average purpose:
    possibilities = []
    for i in range(0,len(parent_data)):
        if parent_data[i][split_attribute] not in possibilities:
            possibilities.append(parent_data[i][split_attribute])
    #iterate through possibilities, subtract weighted entropy of each
    for i in possibilities:
        subtable = []
        #iterate through parent data, taking what rows we need
        for j in range(0,len(parent_data)):
            if i == parent_data[j][split_attribute]:
                subtable.append(parent_data[j])
        #subtract weighted child entropy
        IG -= ((len(subtable)/len(parent_data)) * entropy(subtable))
    return IG;

#---------------------ID3------------------------------------------
#examples is 2D dataset, target is label, attributes is an array of numbers

def ID3(data, list_features, parent_index, branch_value, num_bins=5):
    global tree

    #Create a root node for the tree
    root = node()     #is this initialized right?
    root.parent_index = parent_index
    root.my_index = len(tree)
    
    #update the parent to include this child if not top of tree
    if root.my_index != 0:
        tree[root.parent_index].children[branch_value] = len(tree)
    
    tree.append(root)
    print(tree[0].children)
#BASE CASES
    if len(data) == 0:
        root.prediction_value = tree[root.parent_index].majority_class
        return
    else: root.majority_class = majority(data)
    #If examples all have same class label, return root node with that label
    if entropy(data) == 0:
        root.prediction_value = data[0][-1]
        return
    #if there are no attributes left to split, return majority
    if len(list_features) == 0:
        root.prediction_value = majority(data)
        return
    #if no base case is true, we go into the recursion....

    #find best feature
    max_gain = -1.0
    #this is the index of the feauture in the data
    max_index = -1
    for i in list_features:
        gain = info_gain(data, i)
        if gain > max_gain:
            max_index = i
            max_gain = gain
    #set best feature for node
    tree[root.my_index].feature = max_index
    #fill out the children dictionary
    for i in range(num_bins):
        if i not in tree[root.my_index].children.keys():
            tree[root.my_index].children[i] = -1

    copy_list_features = list_features[:]

    for i in tree[root.my_index].children.keys():
        #make subtable for each i value (each possible value of the attribute)
        subtable = [x for x in data if x[max_index] == i]
        #remove max_index from list_features before it gets passed to recursive call
        
        if max_index in copy_list_features:
            copy_list_features.remove(max_index)
        #make the recursive call
        ID3(subtable, copy_list_features, root.my_index, i) #i is branch_index, representing each possible value of feature
    return


#--------------------Walk function to traverse tree and classify things---------------------------
def walk_tree(example_row):
    currIndex = 0
    currIndex = int(currIndex)
    predictClass = -1
    while predictClass == -1:
        if tree[currIndex].prediction_value == -1:
            test_feature = tree[currIndex].feature
            featureValue = example_row[test_feature]
                    #Problem Here: featureValue is wrong thing?
            currIndex = tree[currIndex].children[featureValue]
        else:
            return tree[currIndex].prediction_value


#--------------------main-----------------------------------------------
#outside of ID3 ,make tree, list of attributes, and data
#tree = [] was done at the top of the program to assure no issue with the ID3 function

#implement ID3 on processed data:
list_features = [0,1]
#-1 for parent, -1 for branch value cause it might not matter
ID3(processed_data, list_features, -1, -1)
num_correct = 0
for i in range(0,len(processed_data)):
    if walk_tree(processed_data[i]) == processed_data[i][2]: num_correct += 1
    print( processed_data[i][0], processed_data[i][1], processed_data[i][2], walk_tree(processed_data[i]) )
print("\n\n\n")
for i in tree:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(i.children)
    print(i.prediction_value)

percent_correct = num_correct/200 * 100
print("Model Accuracy: ", percent_correct, "%")
print("Training Set Error: ", 100-percent_correct, "%")


#plt.figure()
#plt.xlim(min(A),max(A))
#plt.ylim(min(B),max(B))
#plt.scatter(A,B)
#plt.title('Scatter Plot')
#plt.xlabel('A')
#plt.ylabel('B')
#plt.show()

#get columns in case I need them:
A = []
B = []
for i in range(0,len(copy_rawdata)):
    A.append(copy_rawdata[i][0])
    B.append(copy_rawdata[i][1])

#--------------vizualization-------------------------


