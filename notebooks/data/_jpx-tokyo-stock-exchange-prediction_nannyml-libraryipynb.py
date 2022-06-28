#!/usr/bin/env python
# coding: utf-8

# ![](https://www.eu-startups.com/wp-content/uploads/2020/11/p3tsyjjm4se7h4gdi0gr-1.png)eu-startups.com

# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #001f3f;"><b style="color:orange;">NannyML</b></h1></center>
# 
# "NannyML is an open-source python library that allows you to estimate post-deployment model performance (without access to targets), detect data drift, and intelligently link data drift alerts back to changes in model performance. Built for data scientists, NannyML has an easy-to-use interface, interactive visualizations, is completely model-agnostic and currently supports all tabular classification use cases."
# 
# "The core contributors of NannyML have researched and developed a novel algorithm for estimating model performance: confidence-based performance estimation (CBPE). The nansters also invented a new approach to detect multivariate data drift using PCA-based data reconstruction."
# 
# https://github.com/NannyML/nannyml

# In[ ]:


get_ipython().system('pip install nannyml')


# In[ ]:


get_ipython().system('python -m pip install git+https://github.com/NannyML/nannyml')


# In[ ]:


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Code by https://github.com/NannyML/nannyml

import pandas as pd
import nannyml as nml

# Load dummy data
reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
data = pd.concat([reference, analysis], ignore_index=True)

# Extract meta data
metadata = nml.extract_metadata(data = reference, model_type='classification_binary', exclude_columns=['identifier'])
metadata.target_column_name = 'work_home_actual'

# Choose a chunker or set a chunk size
chunk_size = 5000

# Estimate model performance
estimator = nml.CBPE(model_metadata=metadata, metrics=['roc_auc'], chunk_size=chunk_size)
estimator.fit(reference)
estimated_performance = estimator.estimate(data=data)

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


# #Acknowledgement:
# 
# https://github.com/NannyML/nannyml
