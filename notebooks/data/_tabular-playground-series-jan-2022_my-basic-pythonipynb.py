#!/usr/bin/env python
# coding: utf-8

# In[ ]:


a=2
b=4
c=a
a=b
b=c
print(a)
print(c)
print(b)
#swappingvariable


# In[ ]:


i =0
j =1
for o in range(0,5):
    print(i)
    c = j
    j = j+i
    i = c


# In[ ]:


a = 1
counter = 0
while True :
    if a%2 == 1:
        counter= counter +1
    if a==7:
        print(counter)

        break
    a = a+1
   


# In[ ]:


#Input variable
a = input("What is your name ? " )

print (a)


# In[ ]:


#explicit conversion
a=1
b=3.5
c=a+b
print("one and three makes " + str(c))


# **Arthimatic expression**

# In[ ]:


#Addition
a= 4
b= 6
c= a + b
print(c)
#subraction
a=5
b=4
c=b-a
print(c)
#multiplication
a=2
b=3
c=a*b
print(c)
#Division
a=4
b=2
c=a/b
print(c)
#modulus division for remainder value
a = 30
b = 2
c = a%b
if c == 1:
    print("Odd")
else :
    print("Even")
#Floor Division
a= 10
b= 3
c= a//b
print (c)
d= a%b
print (d)
#Exponention
a=5
b=2
c= a ** 2
print(c)


# **Comparision Operators**

# In[ ]:


#Equal to
a = 1
b = 2
a == b
#Not Equal to 
a = 1
b = 2
a != b
#Greater than
a = 20
b = 21
if a>=b:
    print("Majama")
else:
    print("Kaam kar aur paisa jama kar")
#Lesser than
a = 20
b = 19
if a<=b:
    print("Majama")
else:
    print("Kaam kar aur paisa jama kar")


# In[ ]:





# **Logical Operators**

# In[ ]:


#AND operator
a= -4
b= 1
if a>0 and b>0 :
    print("Pass")
else :
    print("Fail")
#Or Operator
a= -4
b= 1
if a>0 or b>0 :
    print("Pass")
else :
    print("Fail")
#NOT Operator
a=2
b=3
not a>b
#AND OR
Physics = False
Maths = False
Percentage = 34
if Physics == True or Maths == True and Percentage >= 35:
    print("Pass")
else :
    print("Fail")


# 

# In[ ]:


#operator precedency
(2 + 3)/6
a=2
b=3
c=a+b**2
print(c)
aa=2
bb=3
cc=(aa + bb)**2
print(cc)


# In[ ]:


#Conditional statement
#If and indentation
a = int(input("Enter a number : "))
if a>0 :
    print("Positive")
else :
    print ("Negative")
a = input("Enter the place you want to visit : ")
if a == "Mumbai" or "pune" or "p":
    print("Maharashtra")
else :
    print("Out of Maharsahtra")


# In[ ]:


#Nested if
place = input("Enter your destination : ")
people = input("Is She coming ? ")

if place == "USA" :
    print ("Okay!")
    if people == "Yes" :
        print ("I will come too")


# In[ ]:


#Elif
a = int(input("Enter a number : "))
if a>0 :
    print("Positive")
elif a<0 :
    print ("Negative")
else :
    print ("Neutral")
place = input("Enter your destination : ")
people = input("Is She coming ? ")
You = input("Will you pay the amount ? ")

if place == "USA" :
    print ("Okay!")
    if people == "Yes" :
        print ("I will come too")
elif You == "Yes":
    print("I will come")
else :
    print("let me see")


# In[ ]:


#Assignment_1
#Maximum value of 3 numbers 
First =int(input("Enter the 1st Number"))
Second =int(input("Enter the 2nd Number"))
Third =int(input("Enter the 3rd Number"))
Max = First
if Max < Second :
    Max = Second
if Max < Third :
    Max = Third
print(Max)


# In[ ]:





# In[ ]:





# In[ ]:


#Looping
#While Loop
i= 0
while i <= 10:
    print(i)
    i=i+1


# In[ ]:


#Print even numbers between 0 to 10
i=0
while i <= 10 :
    print(i)
    i=i+2
#or
i=0
while i <=  10 :
    if i%2==0 :
        print(i)
    i = i+1


# In[ ]:


#ForLoop
Name = "Prasad"
for i in Name:
    print(i)    


# In[ ]:


Students = ["Virat","Dhoni","Rohit","Rahul","Sachin"]
for i in Students :
    print(i)


# In[ ]:


a= 4
for i in a:
    print(i) #int cannot be used in for loop


# In[ ]:


#Range function in Forloop
for i in range(1,10) :
    print(i)
#Print Odd Numbers 
for i in range (1, 10):
    if i%2==1:
        print(i)


# In[ ]:


#Sum of first 20 numbers 
j=0
for i in range(1,21):
    j=j+i
print(j)
#Using While loop
a=0
j=0
while a <=20:
        j=j+a
        a=a+1

print(j)


# In[ ]:


#Range
#print reverse numbers 
for i in range(10,0,-1):
    print(i)
    


# In[ ]:


#Fibonacci series 
a= 0
b=1
for i in range(0,11):
        print(a)
        c=b
        b=b+a
        a=c


# In[ ]:


#break statement Find 20th odd number
a=1
b=0
while True :
    if a % 2 == 1:
        b=b+1
        
    if b == 20:
        print(a)
        break
    a=a+1


# In[ ]:


#continue statement
i=0
while i <= 5 : 
    i = i + 1
    if i==3 :
            continue
    print(i)


# In[ ]:


#Nested Loops 
i =1
while i <= 4:
    j=1
    while j <= i:
        print("*", end =" ",)
        j=j+1
    i=i+1
    print("\n")


# In[ ]:


#STRING
a = "Data analytic's"
print(a)


# In[ ]:


a = "Showdown"
len(a)
a[-1]
a[0:4]
a[0:8:2]


# In[ ]:


#substring
a = "India is the best"
a.find("best")
a[13]
#split
b=a.split()
b[0]


# In[ ]:


#strip
a = "   India"
b="India"
a==b
a=a.strip()
a==b


# In[ ]:


#inbuilt function
a = "Indiaismycountry"
a.isalnum()
a.isalpha()
b="INDIA"
b.islower()
c="windows"
c=c.upper()
c.isupper()
country= "india"
country.capitalize()
s= "rOHIT sHARMA"
s.swapcase()
s = "rohit sharma"
s.title()
s.endswith("sharma")


# In[ ]:


#looping in strings
a = "I want to learn"
b=a.split()
for i in b:
    print(i)


# In[ ]:


num = input("Enter the number : ")
total = 1
for i in num:
    total = total * int(i)
print(total)


# In[ ]:


a = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
l=len(a)
r= " "
while l > 0:
    r = r + a[l-1]
    l = l - 1
print(r)


# In[ ]:


#User defined function
def my_first_function():
    print("India is the best")
    a = input("Enter your name : ")
    print ("Your name is " + a)


# In[ ]:


my_first_function()


# In[ ]:


#Arguments and Parameters
def Code(a,b):
    add = a+b
    sub = a-b
    mul = a*b
    div = a/b
    return add, mul,div, sub
    


# Code(2,3)

# In[ ]:


a = int(input("Enter the first number : "))
b = int(input("Enter the second number : "))
c = int(input("Enter the third number : "))
if a > b and a > c :
    print("The Value of" + str(a) + "is greater")
elif a < b and c < b:
    print("The Value of" + str(b) + "is greater")
else :
    print("The Value of" + str(c) + "is greater")


# In[ ]:


a = int(input("Enter the first number : "))
b = int(input("Enter the second number : "))
c = int(input("Enter the third number : "))
Max = a
if Max < b :
    Max = b
if Max < c :
    Max = c
print(Max)


# #Reverse string
# a=""
# l(a)

# In[ ]:


while True :
    if i in


# In[ ]:


i = 1
counter=0
while True :
    if i%2==1 :
        counter = counter +1
    if counter == 20:
        print(i)
        break    
    i = i+1
   


# In[ ]:


for i in range(1,10):
    if i == 3:
        continue
    print(i)


# In[ ]:


a="123"
for i in a:
    print(a)


# In[ ]:


i = 1
while i <= 4:
    j= 1
    while j <= 3:
        print(j,end = " ")
        j = j+1
    i = i +1
    print("\n")


# In[ ]:


i=1
while  i <= 5:
    j=1
    while j <= i:
        print("*",end = " ")
        j= j+1
    i =i+1
    print("\n")


# In[ ]:


def my_book(a,b):
    add= a+b
    sub= a-b
    print(add,sub)


# In[ ]:


my_book(1,3)


# In[ ]:


a = "Happy Birthday"
len(a)
a.isalnum()
a.alph()


# In[ ]:


def upper(name):
    count=0
    for i in name:
        if i.isupper():
            count = count +1
    return count


# In[ ]:


c= upper("Dell comPaNy GughjIhBiNoBiiKhLHJhHJHhHKBJhjBJKBJKHhKBjkbjljjjBB")
print(c)


# In[ ]:


def calc(a,b,c):
    if c == "+":
        return a+b 
    elif c == "-":
        return a-b
    elif c == "*":
        return a*b
    elif c == "/":
        return a/b
    
        


# In[ ]:


num_1 = input("Enter first number : ")
num_2 = input("Enter second number : ")
expression = input("Enter the operation :")

result = calc(num_1,num_2,expression)
print(result)


# In[ ]:





# In[ ]:


First_name = ["Virat","Sachin","Rohit","Mahendra"]
Second_name = ["Kohli","Tendulkar","Sharma","Dhoni"]
l=len(First_name)
name=[]
for i in range(l):
    x = First_name[i]+" "+Second_name[i]
    name.append(x)
print(name)


# In[ ]:


l=[1,2,3,4,5,6,6,5,4,3,2,1,3,6,4,6,4,1,2,3,3,3,5]
num=int(input("Enter the num : "))
count = 0
for i in l:
    if num == i:
        count=count+1
print(count)


# In[ ]:


l=[1,2,3,4,5,6,6,5,4,3,2,1,3,6,4,6,4,1,2,3,3,3,5]
num=int(input("Enter the num : "))
a=l.remove(num)
print(a)


# List Sort

# In[ ]:


l=[1,2,3,4,5,6,6,5,4,3,2,1,3,6,4,6,4,1,2,3,3,3,5]
l.sort()
sum(l)
len(l)
average = sum(l)/len(l)
average


# String 
# 

# In[ ]:


a = "Data analytics"
print(a)


# String indexing

# In[ ]:


a[1]
a[0]
a[-1]


# String slicing
# a[start,end,step]

# In[ ]:


a ="Python"
a[0:4]
a[4:6]
a[0:7:2]


# In[ ]:


splitting


# In[ ]:


poem = "Jhonny Jhonny yes pappy"
poem.find("yes")
s = poem.split()
s[0]


# stripping - used for removing spaces

# In[ ]:


a = "Python     "
a.strip()


# In[ ]:


a= "     Python     "
a = a.lstrip()
a 
a = a.rstrip()
a


# Inbuilt Function in strings

# In[ ]:


password="Iuhdgt5@378"
if password.isalnum():
    print("valid")
else:
    print("Invalid")


# In[ ]:


a = "Hello"
a.isalpha()
a = "Hello"
a.islower()
a= "HELLO"
a.isupper()
a=a.lower()
a =a.capitalize()
a = "i am witcher of gerald"
a=a.capitalize()
a=a.title()
a.endswith("Gerald")


# Looping in strings

# In[ ]:


a = "I am witcher of gerald"
b=a.split()
for i in b :
    print(i)


# Reverse a string

# In[ ]:





# In[ ]:


a = "I am witcher of gerald"
l = len(a)
r = " "
while l > 0 :
    r = r + a[l-1]
    l = l -1
print(r)


# In[ ]:


Dictonary


# In[ ]:


age = {"Rahul" : 22,
      "Virat" : 25, "rohit":24}
type(age)


# In[ ]:


age.keys()
age.values()


# In[ ]:


for i in age:
    print(i,age[i])


# In[ ]:


Runs.update(age)
Runs


# In[ ]:


Runs = {"Rahul" : 22,
      "Virat" : 25, "rohit":{"Match_1":87,"Match_2":66,"Match_3":89}}


# Data retrival

# In[ ]:


age["Rahul"]


# In[ ]:


num= int(input("Enter the value : "))
d={}
for i in range(1,num+1):
    d[i]= i*i
print(d)


# In[ ]:


Runs['rohit']['Match_1']


# In[ ]:


name = ["Rohit","Virat","Rahul"]
age = [22,24,26]


# In[ ]:


j=0
d={}
for i in name:
        d[i]=age[j]
        j= j+1
print(d)


# In[ ]:


d1=sorted(d)
d1


# In[ ]:


a="Fear leads to anger and anger leads to hate"
b=a.split()
for i in b:
    print(i)


# In[ ]:


t=("Hello",34,-23,"there",["Black","Blue"])
type(t)


# Indexing

# In[ ]:


t[4][1]


# slicing

# In[ ]:


t[0:3]


# In[ ]:


for i in t:
    print(i)


# In[ ]:


l=len(t)
i=0
while l>0:
    print(t[i])
    i=i+1
    l=l-1


# Zip function

# In[ ]:


l1=["Dhoni","Virat","Sachin"]
l2=[33,44,55]
z=list(zip(l1,l2))
print(z)
for i,j in z:
    print(i,j)


# In[ ]:


#Tuple- method
t1=(12,34,56,78,90)
t2=(9,87,65,43,21)
t3=(t1+t2)
t3


# In[ ]:


max(t3)
min(t3)
t3.count(9)
sum(t3)


# *Sets*

# In[ ]:


s ={"India","Japan","Australia","Spain"}
type(s)


# In[ ]:


#intersection
A = {12,34,56,78,90}
B = {23,12,35,78,65}
C=A.intersection(B)
C


# In[ ]:


D = A&B
D


# In[ ]:


#Union
d= A.union(B)
dd= A|B
dd


# In[ ]:


#Difference
f=A-B
f
e=B-A
e


# In[ ]:


#Symetric difference
a  = A^B
a


# In[ ]:


#Methods
A.add(56)
A.pop()
A={45,66,77,99}
A.pop()
A
A.add(56)
A.remove(77)
A.discard(56)
A


# In[ ]:


A={1,2,3,4,5}
B={6,7,8,9,0}
A.update(B)
A.clear()
A


# In[ ]:


#Palangram
A="Waltz bad nymph for quick igs vex"
A=A.lower()
B=set(A)
count = 0
for i in B:
    if i.isalpha()==True:
        count = count + 1
if count == 26:
        print("This is palangram")
else :
        print("This is not a palangram")


# In[ ]:


#subset
A={1,2,3,4,5,6,7,8,9,0}
B={2,4,6,8,0}

count = 0
for i in B :
    for j in A:
        if i == j :
            count = count + 1
if count == len(B):
    print("Subset of A")


# In[ ]:


#FORMAT
A="We are learning {}"
A.format("pythn")
A


# In[ ]:


A="We are learning {1} and {0}"
b =A.format("python","SQL")
b


# In[ ]:


#Errors
Syntax error
runtime error
logical error


# In[ ]:


#lambda function
add1 = lambda a,b,c,d : (a + b)/(c-d)
a=add1(1,2,3,4)
a


# In[ ]:


#MAP
def attendance(No_of_Students):
    if No_of_Students > 60:
        return "Good"
    elif No_of_Students < 60 and No_of_Students > 35:
        return "Average"
    elif No_of_Students < 35 :
        return "Bad"
    


# In[ ]:


l=[12,34,45,46,67]
list(map(attendance, l))


# In[ ]:


#Library
DateTimelibrary


# In[ ]:


import datetime
x=datetime.date.today()
x
x.year
x.month
x.weekday()


# In[ ]:


x.strftime("%b %w %Y")


# In[ ]:


#Time  module
from datetime import time


# In[ ]:


y = time(22,56,34)
y=y.replace(hour=21)
y.strftime("%I %M %S %p")


# In[ ]:


#Datetime module
from datetime import datetime


# In[ ]:


z=datetime(2034,4,23,22,56,59)
z.now()


# 
