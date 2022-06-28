#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Question 1
import numpy
print("Creating 5X2 array using numpy.arrange")
sampleArray = numpy.arange(100,200,10)
sampleArray=sampleArray.reshape(5,2)
print(sampleArray)


# In[ ]:


Question 2
import numpy
sampleArray= numpy.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
print("Printing INput Array")
print("sampleArray")

print("Printing array of items in the third column from all rows")
newArray = sampleArray[...,2]
print(newArray)


# In[ ]:


Question 3
import numpy
print("Creating 8X3 array using numpy.arange")
sampleArray= numpy.arange(10,34,1)
sampleArray= sampleArray.reshape(8,3)
print(sampleArray)

print("Diving 8X3 array into 4 sub array")
subArrays = numpy.split(sampleArray, 4)
print(subArrays)


# In[ ]:


Question 4
import numpy
print("Printing Original array")
sampleArray= numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
print(sampleArray)

sortArrayByRow= sampleArray[:,sampleArray[1,:].argsort()]
print("Sorting Original array by secoond row")
print(sortArrayByRow)

print("Sorting Original array by secoond column")
sortArrayByColumn = sampleArray[sampleArray[:,1].argsort()]
print(sortArrayByColumn)


# In[ ]:


import numpy
print("Printing Original array")
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
print (sampleArray)

print("Array after deleting column 2 on axis 1")
sampleArray= numpy.delete(sampleArray,1,axis=1)
print(sampleArray)

arr = numpy.array([[10,10,10]])
print("Array after inserting column 2 on axis 1")
sampleArray= numpy.insert(sampleArray,1, arr, axis=1)
print(sampleArray)

