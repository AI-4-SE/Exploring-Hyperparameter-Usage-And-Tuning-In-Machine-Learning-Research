#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Проаналізуємо оцінки студентів на єкзаменах з математики та письма.

# In[ ]:


df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv', sep=',')
df=df[['math score', 'writing score']]
df.head(20)


# Побудуємо діаграму розсіювання. На графіку бачимо чітку позитивну кореляцію між змінними.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(x=df['math score'], y=df['writing score'], color='#581845')
plt.xlabel('Math score')
plt.ylabel('Writing score')
plt.show()


# Розрахуємо коєфіцієнт кореляції між оцінками студентів на єкзаменах х математики та з письма.

# In[ ]:


n = len(x)
x_mean = round((np.mean(x)),2)
y_mean = round((np.mean(y)),2)
print(x_mean, y_mean)


# In[ ]:


SS_xy = round((np.sum(y*x) - n*y_mean*x_mean),2)
SS_xx = round((np.sum(x*x) - n*x_mean*x_mean),2)
SS_yy = round((np.sum(y*y) - n*y_mean*y_mean),2)
print(SS_xy, SS_xx, SS_yy)


# In[ ]:


SSE = SS_yy - b_1 * SS_xy
print(round(SSE,2))


# In[ ]:


x=np.array(df['math score'])
y=np.array(df['writing score'])
corr = SS_xy/math.sqrt(SS_xx*SS_yy)
print(round(corr,2))


# Отримали достатньо велику додатню кореляцію. Це може свідчити про стійкий лінійний зв'язок між випадковими величинами.
# 
# Розрахуємо коєфіцієнт детермінації з допомогою вбудованої функції.

# In[ ]:


det_y = (SS_yy-SSE)/SS_yy
print(round(det_y,2))


# Перевіримо, чи значуще відрізняється коефіцієнт кореляції від нуля при рівні значущості l=0.05.

# In[ ]:


t=corr/math.sqrt((1-corr**2)/(len(df['math score']-2)))
print(round(t,2))


# З таблиці розподілу Стьюдента знайдемо критичну область.

# In[ ]:


st_005 = 1.645


# In[ ]:


print(t>st_005)


# Так як t належить критичній області, то гіпотеза про рівність нулю коефіцієнта кореляції відхиляється.
# 
# Побудуємо рівняння лінійної регресії та його графік.

# In[ ]:


b_1 = round((SS_xy / SS_xx),2)
b_0 = round((y_mean - b_1*x_mean),2)
print(b_0, b_1)


# In[ ]:


print('y =', round(b_0,2), "+", round(b_1,2), "* x")


# Отримали рівняння.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
plt.scatter(x, y, color = "#581845", marker = "o", s = 30)
y_pred = b_0 + b_1*x
plt.plot(x, y_pred, color = "#FF5733", linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Оцінимо дисперсію похибки.

# In[ ]:


s = math.sqrt(SSE/n-2)
print(round(s,2))


# Перевіримо гіпотезу про значущість коєфієнта регресії за рівня значущості 0.05.

# In[ ]:


t = b_1/(s/math.sqrt(SS_xx))
print(round(t,2))


# In[ ]:


st_005 = 1.645


# In[ ]:


print(t>st_005)


# Коефіцієнт b_1 є статистично значущим.
# 
# Побудуємо довірчий інтервал для коефіцієнта регресії.

# In[ ]:


st_0025 = 1.96
a = b_1 - st_0025*(s/math.sqrt(SS_xx))
b = b_1 + st_0025*(s/math.sqrt(SS_xx))
print('(', round(a,2), ';', round(b,2), ')')

