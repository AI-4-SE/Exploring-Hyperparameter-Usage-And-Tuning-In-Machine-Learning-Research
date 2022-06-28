#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('python -m pip install git+https://github.com/NannyML/nannyml')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nannyml as nml # monitoring models
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[ ]:


# Load dummy data
reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
data = pd.concat([reference, analysis], ignore_index=True)


# In[ ]:


data


# In[ ]:


# Extract meta data
metadata = nml.extract_metadata(data = reference, model_type='classification_binary', exclude_columns=['identifier'])
metadata.target_column_name = 'work_home_actual'

# Choose a chunker or set a chunk size
chunk_size = 5000

# Estimate model performance
estimator = nml.CBPE(model_metadata=metadata, metrics=['roc_auc'], chunk_size=chunk_size)
estimator.fit(reference)
estimated_performance = estimator.estimate(data=data)


# In[ ]:


metadata


# In[ ]:


figure = estimated_performance.plot(metric='roc_auc', kind='performance')
figure.show()

# Detect multivariate feature drift
multivariate_calculator = nml.DataReconstructionDriftCalculator(model_metadata=metadata, chunk_size=chunk_size)
multivariate_calculator.fit(reference_data=reference)
multivariate_results = multivariate_calculator.calculate(data=data)

figure = multivariate_results.plot(kind='drift')
figure.show()

# Detect univariate feature drift
univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=chunk_size)
univariate_calculator.fit(reference_data=reference)
univariate_results = univariate_calculator.calculate(data=data)

# Rank features based on number of alerts
ranker = nml.Ranker.by('alert_count')
ranked_features = ranker.rank(univariate_results, model_metadata=metadata, only_drifting = False)

for feature in ranked_features.feature:
    figure = univariate_results.plot(kind='feature_distribution', feature_label=feature)
    figure.show()


# # Final
