#!/usr/bin/env python
# coding: utf-8

# **if statement**

# In[ ]:


a = int(input('Enter the number'))
if a > 0:
    print('The number is positive')
    print('we are done')
    


# In[ ]:


place = input('Enter the visiting place')
a = int(input('Enter the number'))
if place == 'beach':
    print('Lets go!')
if a > 0:
    print('The number is positive')
    print('we are done')



# **Nested if**

# In[ ]:


place = input('Enter the place')
bestfriend = input('Is my bestfriend going?')
if place == 'beach':
    print('I may go!')
    if bestfriend == 'yes':
        print('Lets go!')
                   


# **elif statement**

# In[ ]:


a = int(input('Enter the number'))
if a > 0:
    print('The number is positive')
elif a < 0:
    print('The number is negative')


# In[ ]:


place = input('Enter the place')
bestfriend = input('Is my bestfriend going?')
sponsor = input('Are you sponsoring for the trip?')
if place == 'beach':
    print('I may go!')
    if bestfriend == 'yes':
        print('Lets go!')
elif sponsor == 'yes':
    print('Lets go!')


# **else**

# In[ ]:


a = int(input('Enter the number'))
if a > 0:
    print('The number is positive')
else:
    print('The number is negative')


# In[ ]:


a = int(input('Enter the number'))
if a > 0:
    print('The number is positive')
elif a < 0:
    print('The number is negative')
else:
    print('neither positive nor negative')


# In[ ]:


place = input('Enter the place')
bestfriend = input('Is my bestfriend going?')
sponsor = input('Are you sponsoring for the trip?')
if place == 'beach':
    print('I may go!')
    if bestfriend == 'yes':
        print('Lets go!')
elif sponsor == 'yes':
    print('Lets go!')
else:
    print('Let me see')


# **Assignment 1 Greatest among 3**

# In[ ]:


a = int(input('Enter the first number'))
b = int(input('Enter the second number'))
c = int(input('Enter the third number'))


# In[ ]:


maximum = a
if maximum < b:
    maximum = b
if maximum < c:
    maximum = c
print(maximum)
    

