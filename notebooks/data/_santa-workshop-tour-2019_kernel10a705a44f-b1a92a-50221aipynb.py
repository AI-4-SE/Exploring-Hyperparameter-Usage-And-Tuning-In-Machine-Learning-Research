#!/usr/bin/env python
# coding: utf-8

# Задача - дано 5000 семей, у каждой есть топ-10 приоритетных дней (choice_0 - choice_9) для посещения музея Санта Клауса. В день музей принимает от 125 до 300 посетителей, размеры семей даны. Если семья не попадает в свой первый приоритет, то она получает бонусы; чем дальше выбранный день от choice_0 семьи, тем больше бонус. Кроме того существуют дополнительные расходы в зависимости от заполненности музея в каждый конкретный день. Необходимо распределить семьи по дням так, чтобы минимизировать расходы музея

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#данные приведены в виде таблицы (id семьи, приоритеты, размер семьи)
fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

#стартовое решение, предоставленное kaggle
fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')

#вводим переменные min_occ, max_occ - минимальное и максимальное количество посетителей за день
min_occ = 125
max_occ = 300

#создаем словарь, где ключ - id семьи, а значение - ее размер
family_size_dict = data[['n_people']].to_dict()['n_people']

#создаем массив словарей. каждой записи соответствует список приоритетов соответствующей семьи
family_choices = []
for f in range(5000):
    family_choices.append(data.loc[f,'choice_0':'choice_9'].to_dict())

N_DAYS = 100
#списку дней задается обратный порядок, так как отсчет идет от рождества (1 день - 24 декабря, 2 день - 23 декабря итд)
days = list(range(N_DAYS,0,-1))

#словарь {id: размер семьи} сортируется по убыванию количества человек в семье
fam_forSort = data['n_people'].to_dict();
list_forSort = list(fam_forSort.items())
fam_sorted = sorted(list_forSort, key=lambda i: i[1], reverse=True)

#функция, возвращающая ключ по значению
def get_key(arr, value):
    for k, v in arr.items():
        if v == value:
            return k
    return 'none'

#функция, проверяющая, не превосходит ли число посетителей в данный день 200
#значение 200 выбрано с целью сбалансировать число посетителей между днями
g_daily_occupancy = {k:0 for k in days}
def get_daily_occupancy(day, n_people):
    if (g_daily_occupancy[day] + n_people <= 200):
        g_daily_occupancy[day] += n_people
        return True
    else:
        return False

#функция, считающая расходы для решения
def cost_function(prediction):
    penalty = 0

    #переменная, подсчитывающая число посетителей для каждого дня
    daily_occupancy = {k:0 for k in days}
    
    #f - id семьи, d - день, назначенный для этой семьи
    for f, d in enumerate(prediction):

       #n - размер рассматриваемой семьи 
        n = family_size_dict[f]
        daily_occupancy[d] += n
        
        #определяется, каким приоритетом назначенный день является для данной семьи
        d_name = get_key(family_choices[f], d)

        #считаются расходы в зависимости от приоритета
        if d_name == 'choice_0':
            penalty += 0
        elif d_name == 'choice_1':
            penalty += 50
        elif d_name == 'choice_2':
            penalty += 50 + 9 * n
        elif d_name == 'choice_3':
            penalty += 100 + 9 * n
        elif d_name == 'choice_4':
            penalty += 200 + 9 * n
        elif d_name == 'choice_5':
            penalty += 200 + 18 * n
        elif d_name == 'choice_6':
            penalty += 300 + 18 * n
        elif d_name == 'choice_7':
            penalty += 300 + 36 * n
        elif d_name == 'choice_8':
            penalty += 400 + 36 * n
        elif d_name == 'choice_9':
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    #проверяется, сответствует ли число посетителей рамкам, если нет - расход увеличивается настолько, чтобы превосходить любой рабочий вариант
    for _, v in daily_occupancy.items():
        if (v > max_occ) or (v < min_occ):
            penalty += 100000000
            
    #подсчет дополнительных расходов
    #так как в формуле для n-го дня используется значение для n+1-го, то день 100 подсчитывается отдельно
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    accounting_cost = max(0, accounting_cost)

    #подсчет для остальных дней с использованием значения предыдущего дня
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty, daily_occupancy


#считаем расходы для стартового решения
best = submission['assigned_day'].tolist()
start_score = cost_function(best)[0]
print(start_score)
new = best.copy()

#для каждой семьи ставим значение дня - 121 (т.е. не входящее в заданный диапазон, чтобы освободить все дни)
for i in range(5000):
    new[i] = 121;

#список семей, попавших в день последнего приоритета
unlucky_arr = []


# Каждой семье в порядке уменьшения (начиная с самых многочисленных) ставится в соответствие день их первого приоритета (choice_0). Затем проверяется, не превышает ли число поситетелей в этот день значение 200. Если превышает, то увеличиваем значение pick и присваиваем семье день их следующего приоритета. Если pick доходит до 9, то семье ставится день choice_9, даже если значение этого дня переполнено. Семья, дошедшая до choice_9 заносится в список unlucky_arr. Мы будем менять значения дней только для этих семей и этим избавимся от переполненных дней (т.к. они переполняются только за счет семей с choice_9).

# In[ ]:


for fam_id, f in enumerate(fam_sorted):
    pick=0
    while pick<10:
        new[f[0]] = family_choices[f[0]][f'choice_{pick}']
        ans = get_daily_occupancy(family_choices[f[0]][f'choice_{pick}'], f[1])
        if pick==9:
            unlucky_arr.append(f)
        if (ans==False):
            pick +=1
        else: 
            break;

#из списка дней выбираются в under_occ те, в которые посетителей меньше 125
res_daily_occ = list(cost_function(new)[1].items())
under_occ = []
for i, k in enumerate(res_daily_occ):
    if k[1]<125: under_occ.append(k)

#список unlucky_arr сортируется по возрастанию, так как семьи из него будут переноситься в дни, в которые не хватило посетителей
#таким образом, семьи, в которых больше людей, скорее останутся на своем 10 приоритете, а те, в которых меньше - попадут на случайные дни
unlucky_arr = sorted(unlucky_arr, key=lambda i: i[1])


# Для каждого дня из списка under_occ производится следующее: к нему приписывается семья из unlucky_arr, к количеству людей в этот день прибавляется ее размер, семья удаляется из unlucky_arr. Если число посетителей в данный день превысило 130, день удаляется из under_occ => переходим к следующему. Значение 130 выбрано затем, чтобы предупредить ситуацию, когда семья из дня under_occ[k] (который довели до 125 посетителей) переносится на under_occ[k+m] и таким образом в under_occ[k] снова не хватает посетителей

# In[ ]:


i=0
j=0
while (under_occ[j][1]<130) :
    new[unlucky_arr[i][0]] = under_occ[j][0]
    tmp=under_occ[j];
    under_occ.remove(under_occ[j])
    if (j<len(under_occ)):
        under_occ.insert(j, (tmp[0],tmp[1]+unlucky_arr[i][1]))
    else: under_occ.insert(0, (tmp[0],tmp[1]+unlucky_arr[i][1]))
    if (unlucky_arr):
        unlucky_arr.remove(unlucky_arr[i])
    else: break
    if (under_occ[j][1]>=130):
        under_occ.remove(under_occ[j])
    if (len(under_occ)==0):
        break

#считаем и выводим расходы для нового решения
res = cost_function(new)
print('penalty = ', res[0])
print(res[1])


# Сортируем список семей по возрастанию и запускаем генетический алгоритм. Зачем по возрастанию? На данный момент для больших семей учтены их пожелания, а маленькие семьи разнесены как придется. При этом сейчас большинство дней не переполнены, а следовательно, можно пробовать давать маленьким семьям их первые приоритеты. Если значение расходов уменьшается, то алгоритм запоминает это значение приоритета для рассматриваемой семьи и переходит к следующей. Данный перебор повторяется несколько раз, так как перенос семей на более высокие приоритеты освобождает дни, которые при повторном переборе могут оказаться более высокими приоритетами у других семей

# In[ ]:


fam_sorted.reverse();
for i in range(15):
    for fam_id, f in enumerate(fam_sorted):
        for pick in range(10):
            day = family_choices[f[0]][f'choice_{pick}']
            temp = new.copy()
            temp[f[0]] = day # add in the new pick
            if cost_function(temp)[0] < start_score:
                new = temp.copy()
                start_score = cost_function(new)[0]
                break;

#когда перебор заканчивается, найденное решение записывается в таблицу
submission['assigned_day'] = new
print(cost_function(new)[1])
score = cost_function(new)[0]
submission.to_csv(f'submission_{score}.csv')
print(f'Score: {score}')

