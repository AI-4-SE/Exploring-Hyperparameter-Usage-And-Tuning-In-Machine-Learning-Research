#!/usr/bin/env python
# coding: utf-8

# ## Operational Risk Loss Estimation :
# 
# Operational Risk losses stem from different domains like business continuity interruption (e.g. due to failure in process execution, natural catastrophes, large scale cyber attacks etc.), frauds (internal and external) etc. The main two dimensions which drive the OpRisk Loss are: Likelihood (probability of the event's happening in a year) and Severity (magnitude of loss amount in USD value) associated with the events. For the baseline estimation of OpRisk Loss amount, Likelihood is multiplied by the Severity of an incident.
# 
# $$OpRisk Loss = Likelihood * Severity$$
# 
# ## Sensitivity Analysis :
# 
# Sensitivity Analysis is the process of passing different inputs to a model to see how the outputs change. The purpose of Sensitivity Analysis is to understand how the outputs change over the full range of possible inputs. Sensitivity Analysis does not derive any expected outcome or a probability distribution of outcomes, instead returning a range of possible deterministic output values associated with each set of inputs.
# 
# 
# For the model expressed by:
# 
# $$y = f(X)$$
# $$X = [x_1, x_2, ..., x_n]$$
# 
# Where,
# 
# - $y$: Model output
# - $X$: Model input matrix
# - $x_i$ Value if $i$th $x$ variable
# 
# Follow the following steps:
# 
# - Choose a set of values for each $x_i$
# - Take the cartesian product of these values as $[X_1, X_2, ..., X_m]$
# - For each $X_i$, calculate $y_i = f(X_i)$
# - Store the values of $X_i$ mapped to $y_i$
# - Visualize $y_i$ versus $X_i$
# 
# ## Acknowledgement :
# 
# The methodology in Python is inspired by the workings done by **Nick DeRobertis** in his Financial Modelling course's Retirement Model which is [freely available here](https://nickderobertis.github.io/fin-model-course/)

# In[ ]:


import pandas as pd

def oprisk_model(likelihood, severity):
    """
    Represents f from above
    """
    return likelihood * severity


# In[ ]:


# Let's consider an event like earthquake which occurs once in every 10 years. 
# Let's assume: When it occurs, it results in 40 Mln USD loss to the organization
y = oprisk_model(0.1, 40) 
y


# In[ ]:


# Let's assume the events have 4 categories of likelihood to occur: 
# once in every 100 years, once in every 10 years, once in every 4 years, once in every 2 years
Likelihood_values = [0.01, 0.1, 0.25, 0.5]

# Let's assume the events have 4 categories of severity on the basis of loss impact on organization: 
# $ 10 Mln, $ 20 Mln, $ 25 Mln, $ 40 Mln
Severity_values = [10, 20, 25, 40]


# Now we have each $X_i$, we need to calculate $y_i = f(X_i)$.
# 
# Here, $X_1$ is Likelihood and $X_2$ is Severity

# In[ ]:


for x1 in Likelihood_values:
    for x2 in Severity_values:
        y_i = oprisk_model(x1, x2)
        print(y_i)


# In[ ]:


outputs = []
for x1 in Likelihood_values:
    for x2 in Severity_values:
        y_i = oprisk_model(x1, x2)
        outputs.append((x1, x2, y_i))
outputs


# Let's visualize the result.

# In[ ]:


df = pd.DataFrame(outputs, columns=['Likelihood', 'Severity', 'OpRisk_Loss_in_Mln_USD'])
df


# Let's highlight the high and low values for convenience.

# In[ ]:


df.style.background_gradient(subset='OpRisk_Loss_in_Mln_USD', cmap='RdYlGn_r')


# Let us display result with Hex-Bin Plot

# In[ ]:


df.plot.hexbin(x='Likelihood', y='Severity', C='OpRisk_Loss_in_Mln_USD', gridsize=4, cmap='RdYlGn_r', sharex=False)


# ## Using The Sensitivity Library
# 
# The [sensitivity](https://pypi.org/project/sensitivity/) package is designed to make this process easier. It is able to handle more than two varying inputs. 

# In[ ]:


get_ipython().system(' pip install sensitivity')


# In[ ]:


from sensitivity import SensitivityAnalyzer

sensitivity_dict = {
    'likelihood': [0.01, 0.1, 0.25, 0.5],
    'severity': [10, 20, 25, 40]
}

sa = SensitivityAnalyzer(sensitivity_dict, oprisk_model, reverse_colors=True)


# In[ ]:


sa.df


# In[ ]:


labels = {
    'likelihood': 'Likelihood',
    'severity': 'Severity'
}

sa = SensitivityAnalyzer(
     sensitivity_dict, oprisk_model, grid_size=4, reverse_colors=True, color_map='RdYlGn', labels=labels
)
plot = sa.plot()


# In[ ]:


styled = sa.styled_dfs(num_fmt='{:.2f}')


# ## Finetuned OpRisk Loss Model :
# 
# Let us consider a third dimension to the Operational Risk Loss model i.e. Velocity, apart from Likelihood and Severity. Velocity denotes how quickly the event spread and can make cascaded impacts on other risk domains. For e.g. : if a cyber attack occurs, it would not only cause business interruption, but will also result in secondary losses like attracting regulatory fines (if it is a breach), then other settlement and restoration costs etc. 
# 
# $$OpRisk Loss = (Likelihood * Severity)^{Velocity} $$

# In[ ]:


def oprisk_model_finetuned(likelihood, severity, velocity):
    return ((likelihood * severity) ** velocity)

sensitivity_dict = {
    'likelihood': [0.01, 0.1, 0.25, 0.5],
    'severity': [10, 20, 25, 40],
    'velocity': [1.2, 1.5, 2, 2.5]
}

sa = SensitivityAnalyzer(sensitivity_dict, oprisk_model_finetuned, reverse_colors=True, grid_size=4)


# In[ ]:


plot = sa.plot()


# In[ ]:


styled_dict = sa.styled_dfs(num_fmt='{:.2f}')


# We observe that the **probable maximum loss (worst case scenario)** comes as **678.35 Mln USD**, considering all our assumptions are valid. The approach used here is deterministic (formulaic) and not probabilistic.

# # Advanced Sensitivity Analysis :
# 
# **Sobol Sensitivity Analysis** is a Variance-based global sensitivity analysis which works in a probabilistic framework. It decomposes the variance of the output of the model into parts which can be attributed to different inputs (or set of inputs). For e.g.: given a model with two inputs and one output, it might be found that 60% of the output variance is caused by the variance in the 1st input, 30% by the variance in the 2nd input, and rest 10% due to interactions between the 1st and 2nd inputs.

# ## Acknowledgement :
# 
# Works using SAlib package in Python is inspired by the workings done by **Dr. Will Usher** in his GitHub repository which is [freely available here](https://github.com/willu47)

# In[ ]:


get_ipython().system('pip install SAlib')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import sobol_sequence
from SALib.analyze import sobol
from SALib.sample import saltelli


# We assume that the resultant loss curve (convoluting both components: Likelihood curve and Severity curve) is Exponential. An exponential curve is represented by the following equation:
# 
# $Resultant OpRisk Loss Curve (Exponential) = a*(1+r)^{x}$
# 
# Where:
# a = initial amount
# r = growth rate
# x = number of time intervals

# In[ ]:


def oprisk_loss_curve(x, a, r):
    return a*((1+r)**x)


# In[ ]:


oprisk_loss_curve_formation = {
    'num_vars': 2,
    'names': ['a', 'r'],
    'bounds': [[25, 40], [1.5, 3.0]] 
    # Let's assume: starting loss amount is bounded between 25 Mln USD to 40 Mln USD
    # Let's assume: growth rate of the loss is bounded by exponent 1.5 to 3.0
}


# In[ ]:


# sample generation using saltelli
# saltelli is a quasi-random sampling method
param_values = saltelli.sample(oprisk_loss_curve_formation, 2**6)

# evaluate
x = np.linspace(-1, 1, 100)
y = np.array([oprisk_loss_curve(x, *params) for params in param_values])

# analyze
sobol_indices = [sobol.analyze(oprisk_loss_curve_formation, Y) for Y in y.T]


# # Resultant Loss at 95%ile Confidence Limit

# In[ ]:


S1s = np.array([s['S1'] for s in sobol_indices])

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])

for i, ax in enumerate([ax1, ax2]):
    ax.plot(x, S1s[:, i],
            label=r'S1$_\mathregular{{{}}}$'.format(oprisk_loss_curve_formation["names"][i]),
            color='black')
    ax.set_xlabel("x")
    ax.set_ylabel("First-order Sobol index")

    ax.set_ylim(0, 1.04)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.legend(loc='upper right')

ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')

# in percent
prediction_interval = 95

ax0.fill_between(x,
                 np.percentile(y, 50 - prediction_interval/2., axis=0),
                 np.percentile(y, 50 + prediction_interval/2., axis=0),
                 alpha=0.5, color='black',
                 label=f"{prediction_interval} % prediction interval")

ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.legend(title=r"$y=a\cdot(1+r)^x$",
           loc='upper center')._legend_box.align = "left"

plt.show()


# **Result Interpretation:**  
# * At 95%ile confidence limit, the resultant loss comes around 105 Mln USD.
# * The entire confidence interval incorporates loss region starting from 72 Mln USD to 145 Mln USD.
# * First order Sobol indices for both "a" and "r" hover in the range [-1, 1] and follows symmetric pattern.

# # Resultant Loss at 99%ile Confidence Limit

# In[ ]:


S1s = np.array([s['S1'] for s in sobol_indices])

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])

for i, ax in enumerate([ax1, ax2]):
    ax.plot(x, S1s[:, i],
            label=r'S1$_\mathregular{{{}}}$'.format(oprisk_loss_curve_formation["names"][i]),
            color='black')
    ax.set_xlabel("x")
    ax.set_ylabel("First-order Sobol index")

    ax.set_ylim(0, 1.04)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.legend(loc='upper right')

ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')

# in percent
prediction_interval = 99

ax0.fill_between(x,
                 np.percentile(y, 50 - prediction_interval/2., axis=0),
                 np.percentile(y, 50 + prediction_interval/2., axis=0),
                 alpha=0.1, color='black',
                 label=f"{prediction_interval} % prediction interval")

ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.legend(title=r"$y=a\cdot(1+r)^x$",
           loc='upper center')._legend_box.align = "left"

plt.show()


# **Result Interpretation:**  
# * At 99%ile confidence limit, the resultant loss comes around 110 Mln USD. 
# * The entire confidence interval incorporates loss region starting from 70 Mln USD to 150 Mln USD.
# * First order Sobol indices for both "a" and "r" hover in the range [-1, 1] and follows symmetric pattern.

# # Resultant Loss at 99.5%ile Confidence Limit

# In[ ]:


S1s = np.array([s['S1'] for s in sobol_indices])

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])

for i, ax in enumerate([ax1, ax2]):
    ax.plot(x, S1s[:, i],
            label=r'S1$_\mathregular{{{}}}$'.format(oprisk_loss_curve_formation["names"][i]),
            color='black')
    ax.set_xlabel("x")
    ax.set_ylabel("First-order Sobol index")

    ax.set_ylim(0, 1.04)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.legend(loc='upper right')

ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')

# in percent
prediction_interval = 99.5

ax0.fill_between(x,
                 np.percentile(y, 50 - prediction_interval/2., axis=0),
                 np.percentile(y, 50 + prediction_interval/2., axis=0),
                 alpha=0.05, color='black',
                 label=f"{prediction_interval} % prediction interval")

ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.legend(title=r"$y=a\cdot(1+r)^x$",
           loc='upper center')._legend_box.align = "left"

plt.show()


# **Result Interpretation:**  
# 
# * At 99.5%ile confidence limit, the resultant loss comes around 110 Mln USD. Now it is behaving in almost stabilized way at high confidence limits. 
# * The entire confidence interval incorporates loss region starting from 68 Mln USD to 158 Mln USD.
# * First order Sobol indices for both "a" and "r" hover in the range [-1, 1] and follows symmetric pattern.

# In[ ]:




