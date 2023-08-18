import numpy as np
from scipy.io import loadmat
import numpy as np
import os


def exportToCSV(data, name):
    shape = data.shape
    print(shape)
    print("row:", shape[0])
    print("column:", shape[1])
    
    name = "Export/{}.txt". format(name)
    file = open(name, "w+")
    
    for i in range(int(shape[0])):
        for q in range(shape[1]):
            line =str(data[i][q])
            print(line)
            file.write(line)
            
            if q != (shape[1]-1):
                file.write(",")
                
        file.write("\n")
    file.close()


location = input("Where is the location?")
path = os.path.normpath(location)
data = loadmat(path[1:-1])
keys = data.keys()

i = 0
keyNum = {}
for key in keys:
    keyNum[i] = key
    print("[{}] [Key, data] : {},{}".format(i,key, data[key]))
    i += 1

print(keyNum)

extractNum = input("\nExtract which data to CSV? ex. (1), Pick only one:")
name = input("\nName of the file?:")

print(extractNum)
print(name)
print(data[keyNum[int(extractNum)]])

exportToCSV(data[keyNum[int(extractNum)]], name)




    
    
    