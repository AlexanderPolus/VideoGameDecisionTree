#-------------------libraries----------------------------------
import csv
import sys
import math
import matplotlib.pyplot as plt
#-------------------initialize my tree---------------------------


tree = []


#------import file into raw data---------------------------------
print("Running...")
if (len(sys.argv) != 1):
    print("Usage: Python3 SyntheticModel.py <filename> ")
    sys.exit(0)
filename = 'Video_Games_Sales.csv'
file = open(filename)
reader = csv.reader(file, delimiter=',')
rawdata = list(reader)
#print("\nRaw Data: ")
    #for i in rawdata:
        #print(i)
#get rid of the top list of column names:
rawdata.remove(rawdata[0])
#get any N/A's out of my data
for row in rawdata:
    for column in range(0,11):
        if row[column] == "N/A": rawdata.index(row)

#-------Discretize attribute Data------------------------------------------
#This should discretize the first 7 columns of data, will do label later
def discretize_attributes(data):
    for i in range(0,11):
        #get all possible values
        possibilities = []
        for row in data:
            if row[i] not in possibilities:
                possibilities.append(row[i])
        #set each item equal instead to its index in possibilities
        for row in data:
            if row[i] in possibilities:
                #set equal to that value's index in possibilities
                row[i] = possibilities.index(row[i])
    return data
        #I think this is done???


#--------Discretize the class label------------------------------
def discretize_label(data):
    #turn into floats
    for row in data:
        row[11] = float(row[11])
    #make my bins - 9 of them
    for row in data:
        if  0 <= row[11] < 20: row[11] = 15
        elif 20 <= row[11] < 30: row[11] = 25
        elif 30 <= row[11] < 40: row[11] = 35
        elif 40 <= row[11] < 50: row[11] = 45
        elif 50 <= row[11] < 60: row[11] = 55
        elif 60 <= row[11] < 70: row[11] = 65
        elif 70 <= row[11] < 80: row[11] = 75
        elif 80 <= row[11] < 90: row[11] = 85
        elif 90 <= row[11] < 100: row[11] = 95
    return data


    #turn back into strings???


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
#---------------------entropy------------------------------------
def entropy(data):
    total = len(data)
    fifteen = 0.0
    twentyfive = 0.0
    thirtyfive = 0.0
    fourtyfive = 0.0
    fiftyfive = 0.0
    sixtyfive = 0.0
    seventyfive = 0.0
    eightyfive = 0.0
    ninetyfive = 0.0

    #establish a count of each
    for row in data:
        if row[11] == 15: fifteen += 1
        if row[11] == 25: twentyfive += 1
        if row[11] == 35: thirtyfive += 1
        if row[11] == 45: fourtyfive += 1
        if row[11] == 55: fiftyfive += 1
        if row[11] == 65: sixtyfive += 1
        if row[11] == 75: seventyfive += 1
        if row[11] == 85: eightyfive += 1
        if row[11] == 95: ninetyfive += 1
    tens = fifteen / total
    twenties = twentyfive / total
    thirties = thirtyfive / total
    fourties = fourtyfive / total
    fifties = fiftyfive / total
    sixties = sixtyfive / total
    seventies = seventyfive / total
    eighties = eightyfive / total
    nineties = ninetyfive / total

    #Handle zero values:                    --------is this okay???-----
    if tens == 0: tens = 0.00001
    if twenties == 0: twenties = 0.00001
    if thirties == 0: thirties = 0.00001
    if fourties == 0: fourties = 0.00001
    if fifties == 0: fifties = 0.00001
    if sixties == 0: sixties = 0.00001
    if seventies == 0: seventies = 0.00001
    if eighties == 0: eighties = 0.00001
    if nineties == 0: nineties = 0.00001

    options = [tens, twenties, thirties, fourties, fifties, sixties, seventies, eighties, nineties]
    
    ent = 0.0
    for bin in options:
        if bin == 0: ent -= 0
        else: ent -= bin*math.log(bin,2)

    return ent

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
                                                        #removed num_bins=5 here

def ID3(data, list_features, parent_index, branch_value, depth_counter):
    #testing with final_data instead of data.....
    global final_data
    
    global tree
    
    
    #Create a root node for the tree
    root = node()     #is this initialized right?
    root.parent_index = parent_index
    root.my_index = len(tree)
    
    #update the parent to include this child if not top of tree
    if root.my_index != 0:
        tree[root.parent_index].children[branch_value] = len(tree)
    
    tree.append(root)
    #print(tree[0].children)

    #BASE CASES
    if len(data) == 0:
        root.prediction_value = tree[root.parent_index].majority_class
        return
    else: root.majority_class = majority(data)

    if depth_counter == 3:
        root.prediction_value = tree[root.parent_index].majority_class
        return

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



                            #made a change - beginning

    #get the possibilities in the column == max_index
    all_possibilities = []
    for row in final_data:
        if row[max_index] not in all_possibilities:
            all_possibilities.append(row[max_index])


                            #made a change - end




    #fill out the children dictionary -- changed num_bins to len(possibilities)
    for i in range(len(all_possibilities)):
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
        ID3(subtable, copy_list_features, root.my_index, i, depth_counter+1) #i is branch_index, representing each possible value of feature
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
            #print(tree[currIndex].children)
            #print(tree[currIndex].feature)
            
            
            
            currIndex = tree[currIndex].children[featureValue]
        else:
            return tree[currIndex].prediction_value



#-------process the data in prep for ID3---------------------------
attribute_discretized_data = discretize_attributes(rawdata)
final_data = discretize_label(attribute_discretized_data)

list_features = [0,1,2,3,4,5,6,7,8,9,10]
#-1 for parent, -1 for branch value cause it might not matter
ID3(final_data, list_features, -1, -1, 0)
num_correct = 0
for i in range(0,len(final_data)):
    if walk_tree(final_data[i]) == final_data[i][11]: num_correct += 1
    print ( final_data[i], " ", walk_tree(final_data[i]))

#could print tree here if I wanted

#see how many nodes I have
print("number of nodes: ", len(tree))
percent_correct = num_correct / 8134 * 100
print("Model Accuracy: ", percent_correct, "%")
print("Training Set Error: ", 100-percent_correct, "%")




















