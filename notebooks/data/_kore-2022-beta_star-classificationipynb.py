#!/usr/bin/env python
# coding: utf-8

# ## Importing modules

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark import SparkContext
from pyspark.sql.functions import *


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SQLContext, SparkSession
spark = SparkSession.Builder().appName('Project').getOrCreate()


# In[ ]:


from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation


# The dataset examined contains data from 100,000 space observations made by the SDSS
# (Sloan Digital Sky Survey). Each observation is described by 17 attributes and a variable class that identify it as a star, galaxy, or quasar. The dataset has the following
# FEATURES:
# 1. **obj_ID** = (Object Identifier) the unique value that identifies the object in the image catalog
# used by the CAS
# 2. **alpha** = Angle of ascension (RA - Celestial coordinates).
# 3. **delta** = angle of declination (DEC - Celestial coordinates)
# 4. **u** = Ultraviolet filter in the photometric system
# 5. **g** = Green filter in the photometric system
# 6. **r** = Red filter in the photometric system
# 7. **i** = Near-infrared filter in the photometric system
# 8. **z** = Infrared filter in the photometric system.
# 9. **run_ID** = run ID used to identify the specific scan
# 10. **rerun_ID** = Specifies the type of image processing.
# 11. **cam_col** = Identifies a specific scan line belonging to a single run
# 12. **field_ID** = Identifies the portion of the sky observed in a run.
# 13. **spec_obj_ID** = Unique ID used to identify each scan in the dataset.
# 14. **class** = Target variable: Galaxy, Star, Quasar.
# 15. **redshift** = Batochrome effect, based on the increase in the wavelength of the spectrum light detected
# 16. **plate** = ID identifying the spectograph used
# 17. **MJD** = Date of detection
# 18. **fiber_ID** = Identifies the optical fiber used on the spectograph of each observation

# # Data loading and understanding 

# In[ ]:


df = spark.read.csv('../input/stellar-classification-dataset-sdss17/star_classification.csv', header='true', inferSchema='true')


# ## **Exploratory Data Analysis (EDA)**

# In[ ]:


df.printSchema()


# ## **Descriptive Statistics**
# 

# In[ ]:


summary = df.describe().toPandas()
summary.T


# In[ ]:


rows = df.count()
cols = len(df.columns)
print(f'Dimension of the Dataframe is: {(rows,cols)}')


# In[ ]:


num_cols = [item[0] for item in df.dtypes if item[1] != 'string']
print('Le colonne numeriche sono')
print(num_cols)


# In[ ]:


non_num_cols = [item[0] for item in df.dtypes if item[1] == 'string']
print('Le colonne non numeriche sono')
print(non_num_cols)


# ## **Outliers**

# In[ ]:


sns.scatterplot(data=df.toPandas(), x="u", y="g")  


# In[ ]:


sns.scatterplot(data=df.toPandas(), x="u", y="z")  


# In[ ]:


sns.scatterplot(data=df.toPandas(), x="z", y="g")  


# In[ ]:


df_valorianomali=df.where("u=-9999 or g=-9999 or z=-9999")


# In[ ]:


df_valorianomali.show()


# We decided to drop the rows containing these values because, even assuming that they are not detection errors, they still represent outliers compared with the rest of the
# distribution of the 3 variables. We extracted observations containing values of u, g, or z equal to -9999,: this is a single observation

# In[ ]:


df=df.where("u!=-9999")


# In[ ]:


get_ipython().system('pip install vega_datasets')


# In[ ]:


import altair as alt
from vega_datasets import data

df_new = df.sample(withReplacement = False, fraction = 0.15
                   , seed=1234566)

alt.data_transformers.disable_max_rows()

alt.renderers.set_embed_options(theme='dark')


scatter = alt.Chart(df_new.toPandas()).mark_circle(size=20).encode(
    alt.X(field='u',type='quantitative', scale=alt.Scale(zero=False)),
    alt.Y(field='g',type='quantitative', scale=alt.Scale(zero=False)), 
    alt.Tooltip(['u', 'g', 'class']),                                                                  
    alt.Color(field='class',type='nominal', scale=alt.Scale(scheme='purpleorange'))
).properties(
    width=600,
    height=600
).interactive(
).configure_mark(
    opacity=0.8
)

scatter


# In[ ]:


alt.data_transformers.disable_max_rows()

alt.renderers.set_embed_options(theme='dark')


scatter = alt.Chart(df_new.toPandas()).mark_circle(size=20).encode(
    alt.X(field='u',type='quantitative', scale=alt.Scale(zero=False)),
    alt.Y(field='z',type='quantitative', scale=alt.Scale(zero=False)), 
    alt.Tooltip(['u', 'z', 'class']),                                                                  
    alt.Color(field='class',type='nominal', scale=alt.Scale(scheme='purpleorange'))
).properties(
    width=600,
    height=600
).interactive(
).configure_mark(
    opacity=0.8
)

scatter


# In[ ]:


alt.data_transformers.disable_max_rows()

alt.renderers.set_embed_options(theme='dark')


scatter = alt.Chart(df_new.toPandas()).mark_circle(size=20).encode(
    alt.X(field='z',type='quantitative', scale=alt.Scale(zero=False)),
    alt.Y(field='g',type='quantitative', scale=alt.Scale(zero=False)), 
    alt.Tooltip(['z', 'g', 'class']),                                                                  
    alt.Color(field='class',type='nominal', scale=alt.Scale(scheme='purpleorange'))
).properties(
    width=600,
    height=600
).interactive(
).configure_mark(
    opacity=0.8
)

scatter


# In[ ]:


summary = df.describe().toPandas()
summary.T


# We checked the count of distinct values for each variable: we see that the variable rerun_ID contains only one value (equal to 301) repeated in all observations, while the variable spec_obj_ID contains only distinct values (99999, equal to the number of observations) and is therefore an identifier
# of the observation. For this reason, the two variables were dropped.

# In[ ]:


from pyspark.sql.functions import isnan, when, count, col, isnull
missing = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()


# In[ ]:


for c in num_cols:
    print(f'column {c} contains {len(df.select(c).distinct().collect())} distinct values')


# In[ ]:


df = df.drop('rerun_ID', 'spec_obj_ID')


# In[ ]:


df.printSchema()


# ## **Data preparation**

# In[ ]:


df_noclass = df.drop('class')


# In[ ]:


features = df_noclass.schema.names

vectorassembler = VectorAssembler(inputCols = features, outputCol= 'assemblerfeatures')

output_dataset = vectorassembler.transform(df_noclass)  

pearsonCorr = Correlation.corr(output_dataset, 'assemblerfeatures', 'pearson').collect()[0][0]

#trasformo la DenseMatrix in un array numpy
correlation_array = pearsonCorr.toArray() #ritorna un numpy.ndarray

correlationDF = pd.DataFrame(
    correlation_array,
    index = features,
    columns = features
)

correlationDF


# In[ ]:


source_corr = df_noclass.toPandas().corr().reset_index().melt(id_vars='index')

# create dummy ordinal var
sort = {'obj_ID':0, 'alpha': 1, 'delta':2, 'u': 3, 'g': 4, 'r':5, 'i':6, 'z':7, 'run_ID':8,'cam_col':9, 'field_ID':10,'redshift':11, 'plate':12, 'MJD':13, 'fiber_ID':14}

heatmap = alt.Chart(source_corr)\
.mark_rect()\
.transform_calculate(
    order_rows='%s [datum.index]' % sort,
    order_cols='%s [datum.variable]' % sort
)\
.encode(
    alt.X('index:N', title=None, sort=list(sort.keys())),
    alt.Y('variable:N', title=None, sort=list(sort.keys())),
    alt.Color('value:Q', legend=None, scale=alt.Scale(scheme='purplered')),
    #alt.Legend()
)\
.properties(width=700, height=700
)

text = heatmap\
.mark_text(size=14)\
.encode(
    alt.Text('value:Q', format='.4f'),
    color=alt.condition(
        'datum.value > 0.5',
        alt.value('white'),
        alt.value('black'),
    )
)

heatmap + text


# The variables obj_ID, run_ID, field_ID, fiber_ID have been removed from the dataset as they are observation instrumentation identification codes. The variable cam_col was also removed as it identifies the row of survey instruments from which the bservation was taken and has no relevance to the  analysis of the light spectra.
# 
# The variables described above also have a very low correlation with those identifying the photometric filters, which are necessary for classification of the celestial body.
# 
# The plate variables, MjD, despite having a medium to high correlation with those identifying photometric filters, were removed because they provide information regarding the date of detection and the spectrograph used, which are not relevant to the classification of spectra
# light.

# In[ ]:


df= df.drop('obj_ID', 'run_ID', 'field_ID','fiber_ID', 'cam_col', 'plate','MjD' )


# In[ ]:


df_noclass = df.drop('class')


# In[ ]:


features = df_noclass.schema.names

vectorassembler = VectorAssembler(inputCols = features, outputCol= 'assemblerfeatures')

output_dataset = vectorassembler.transform(df_noclass)  

pearsonCorr = Correlation.corr(output_dataset, 'assemblerfeatures', 'pearson').collect()[0][0]

#trasformo la DenseMatrix in un array numpy
correlation_array = pearsonCorr.toArray() #ritorna un numpy.ndarray

correlationDF = pd.DataFrame(
    correlation_array,
    index = features,
    columns = features
)


# Below is the correlation matrix with the variables relevant to the analysis.

# In[ ]:


source_corr = df_noclass.toPandas().corr().reset_index().melt(id_vars='index')

# create dummy ordinal var
sort = {'alpha': 0, 'delta':1, 'u': 2, 'g': 3, 'r':4, 'i':5, 'z':6, 'redshift':7}

heatmap = alt.Chart(source_corr)\
.mark_rect()\
.transform_calculate(
    order_rows='%s [datum.index]' % sort,
    order_cols='%s [datum.variable]' % sort
)\
.encode(
    alt.X('index:N', title=None, sort=list(sort.keys())),
    alt.Y('variable:N', title=None, sort=list(sort.keys())),
    alt.Color('value:Q', legend=None, scale=alt.Scale(scheme='purplered'))
)\
.properties(width=600, height=600
)



text = heatmap\
.mark_text(size=15)\
.encode(
    alt.Text('value:Q', format='.4f'),
    color=alt.condition(
        'datum.value > 0.5',
        alt.value('white'),
        alt.value('black')
    )
)

heatmap + text


# Alpha and delta, despite their low correlation with those identifying the filters photometric filters, were not removed because scatter plot analysis shows that they could have a nonnegligible impact particularly in the classification of stars versus galaxies and quasars.

# In[ ]:


alt.data_transformers.disable_max_rows()

alt.renderers.set_embed_options(theme='dark')


scatter = alt.Chart(df_new.toPandas()).mark_circle(size=20).encode(
    alt.X(field='alpha',type='quantitative', scale=alt.Scale(zero=False)),
    alt.Y(field='delta',type='quantitative', scale=alt.Scale(zero=False)), 
    alt.Tooltip(['alpha', 'delta', 'class']),                                                                  
    alt.Color(field='class',type='nominal', scale=alt.Scale(scheme='purpleorange'))
).properties(
    width=600,
    height=600
).interactive(
).configure_mark(
    opacity=0.8
)

scatter


# ## **Variables Distribution**

# In[ ]:


from pyspark.sql.functions import isnan, when, count, col, isnull
from pyspark.sql import SQLContext 
get_ipython().run_line_magic('matplotlib', 'inline')

sqlCtx = SQLContext(spark)

df.createOrReplaceTempView("star") 

xy_columns = ['alpha'	,'delta',	'u'	,'g',	'r'	,'i'	,'z', 'redshift']

for col in xy_columns:
    
    print(f">>> plotting distribution for {col}")
    
    query = sqlCtx.sql(f'Select {col} from star') 
    
    h = query.select(col).rdd.flatMap(lambda x: x).histogram(50)


    dfh = pd.DataFrame(
        list(zip(*h)), 
        columns=['bin', 'frequency']
    )

    bins = dfh['bin']
    counts = dfh['frequency']

    plt.rcParams['axes.facecolor'] = '#363435'
    plt.figure(facecolor='#363435')
    
    plt.hist(bins, len(bins), weights=counts, color='#cc5c9c', alpha=0.8)

    plt.xlabel(f'{col}', size=13, color='white')
    plt.ylabel("Frequency", size=13, color='white')
    plt.xticks(color='white', size=11)
    plt.yticks(color='white', size=11)
    plt.grid(color='#8c8c8c')

    plt.show()


# ## **Target Variable Distribution**

# In[ ]:


alt.Chart(df.toPandas()).mark_bar().encode(
    alt.X("class:N", bin=False),
    y='count()',
    color= alt.Color(field='class',type='nominal', scale=alt.Scale(scheme='purplered'))
).properties(
    width=500,
    height=500
).interactive(
).configure_mark(
    opacity=0.8
).configure_axis(
    labelFontSize=13,
    titleFontSize=15
)


# More than half of the observations (nearly 60 percent) are categorized as galaxies, while the remaining 40% are divided between stars and quasars. Therefore, we perform oversampling of the STAR and QSO classes and an undersampling of the GALAXY class to make the distribution of the data more homogeneous.

# In[ ]:


df_star = df.where("class='STAR'")
df_galaxy = df.where("class='GALAXY'")
df_qso = df.where("class='QSO'")

ratio_star=float(df_star.count()/df.count())
ratio_galaxy=float(df_galaxy.count()/df.count())
ratio_qso=float(df_qso.count()/df.count())

print(ratio_star,ratio_galaxy,ratio_qso)


# In[ ]:


sampled_star = df_star.sample(withReplacement=True, fraction=0.85, seed=123)
sampled_star.count()


# In[ ]:


sampled_qso = df_qso.sample(withReplacement=True, fraction=1.0, seed=123)
sampled_qso.count()


# In[ ]:


sampled_galaxy = df_galaxy.sample(withReplacement=False, fraction=0.7, seed=123)
sampled_galaxy.count()


# In[ ]:


df_star_over = df_star.unionAll(sampled_star)
df_qso_over = df_qso.unionAll(sampled_qso)
df_star_qso_over = df_star_over.unionAll(df_qso_over)
df_over = df_star_qso_over.unionAll(sampled_galaxy)


# In[ ]:


df_over.count()


# In[ ]:


df_over.groupBy('class').count().orderBy('class').show()


# In[ ]:


alt.Chart(df_over.toPandas()).mark_bar().encode(
    alt.X("class:N", bin=False),
    y='count()',
    color= alt.Color(field='class',type='nominal', scale=alt.Scale(scheme='purplered'))
).properties(
    width=500,
    height=500
).interactive(
).configure_mark(
    opacity=0.8
).configure_axis(
    labelFontSize=13,
    titleFontSize=15
)


# # **Classification**

# In[ ]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# indexing e encoding del df
# indexing string feature columns
indexer = StringIndexer(inputCols = ['class'], outputCols = ['class_index']).setHandleInvalid("skip")

# converting indexed feature columns to One hot ecoded
onehotecoded = OneHotEncoder(inputCols = ['class_index'], outputCols = ['class_vect'])

# Stages of the pipeline
stages = [indexer, onehotecoded]
pipeline = Pipeline(stages=stages)

# when applying the pipeline
df_ohe = pipeline.fit(df_over).transform(df_over)


# In[ ]:


df_ohe.show(1)


# In[ ]:


df_ohe.select("class").distinct().collect()


# In[ ]:


df_ohe.select("class_index").distinct().collect()


# In[ ]:


df_ohe.select("class_vect").distinct().collect()


# In[ ]:


df=df_ohe


# In[ ]:


(trainingData, testData) = df.randomSplit([0.7, 0.3],seed=123)


# In[ ]:


trainingData.count(), testData.count()


# In[ ]:


trainingData.groupBy('class').count().orderBy('class').show()


# ## **Scaling** 

# In[ ]:


from pyspark.ml.feature import VectorAssembler, MinMaxScaler

assembler = VectorAssembler(inputCols=xy_columns, outputCol="features")

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures", min=0.0, max=1.0)

pipeline_preprocessing = Pipeline(stages=[assembler,scaler])

model_preprocessing = pipeline_preprocessing.fit(trainingData)

trainingData_scal = model_preprocessing.transform(trainingData)

testData_scal = model_preprocessing.transform(testData)


# In[ ]:


trainingData_scal.show(5)


# ## **Random Forest**

# In[ ]:


from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator


# In[ ]:


accuracy = list()
for i in range(10,201,10):
  rf = RandomForestClassifier(labelCol="class_index", featuresCol="scaledFeatures", numTrees=i)
  model_rf = rf.fit(trainingData_scal)
  predictions = model_rf.transform(testData_scal)
  acc=MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='class_index', metricName='accuracy')
  accuracy.append(acc.evaluate(predictions))


# In[ ]:


accuracy


# In[ ]:


labels=['10','20','30','40','50','60','70','80','90','100','110','120','130','140','150','160','170','180','190','200']


plt.rcParams['axes.facecolor'] = '#363435'

plt.figure(facecolor='#363435', figsize=(10,7))

plt.plot(labels,accuracy, color='pink')
plt.xlabel('Nr of trees', color='white', size= 14)
plt.ylabel('Accuracy', color='white', size=14)
plt.xticks(color='white', size=12)
plt.yticks(color='white', size=12)
plt.grid(color='#8c8c8c')


plt.plot()


# In[ ]:


rf = RandomForestClassifier(labelCol="class_index", featuresCol="scaledFeatures", numTrees=30)
model_rf = rf.fit(trainingData_scal)
predictions = model_rf.transform(testData_scal)

eval_f1 = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='class_index', metricName='f1')
print("RandomForest_n30", "f1", eval_f1.evaluate(predictions))
accuracy_f1 = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='class_index', metricName='accuracy')
print("RandomForest_n30", "acc", accuracy_f1.evaluate(predictions))


# In[ ]:


predictions.show(5)


# In[ ]:


from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np

predictionAndLabels = predictions.select("prediction", "class_index").rdd.map(lambda row: (row.prediction, float(row.class_index)))
metrics = MulticlassMetrics(predictionAndLabels)
cfmx = metrics.confusionMatrix()
cfarr = cfmx.toArray() 
confusionmatrix = pd.DataFrame(cfarr)

ax= plt.subplot()
sns.set(rc={'figure.figsize':(7,5), 'axes.facecolor':'#363435', 'figure.facecolor':'#363435','axes.labelcolor': 'white',
            'xtick.color': 'white', 'ytick.color': 'white'})
sns.heatmap(confusionmatrix/np.sum(confusionmatrix), annot=True,cmap='PuRd') 
ax.set_xlabel("Predicted labels");
ax.set_ylabel("True labels"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["GALAXY", "STAR", "QUASAR"])
ax.yaxis.set_ticklabels(["GALAXY", "STAR", "QUASAR"])
plt.show()


# ## **Naive Bayes**

# In[ ]:


from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create initial Naïve Bayes model

nb = NaiveBayes(labelCol="class_index", featuresCol="scaledFeatures", modelType='gaussian')
#model_nb = nb.fit(trainingData_scal)

# Create ParamGrid for Cross Validation
nbparamGrid = (ParamGridBuilder()
               .addGrid(nb.smoothing, [0.2, 0.4, 0.6, 0.8, 1.0])
               .build())

# Evaluate model
nbevaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol='class_index', metricName='accuracy') 

# Create 5-fold CrossValidator
nbcv = CrossValidator(estimator = nb,
                    estimatorParamMaps = nbparamGrid,
                    evaluator = nbevaluator,
                    numFolds = 8)

# Run cross validations
nbcvModel = nbcv.fit(trainingData_scal)
print(nbcvModel)

# Use test set here so we can measure the accuracy of our model on new data
pred = nbcvModel.transform(testData_scal)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
print('Accuracy:', nbevaluator.evaluate(pred))
#print('AUC:', BinaryClassificationMetrics(pred['class_index','prediction'].rdd).areaUnderROC)


# In[ ]:


pred.show(5)


# In[ ]:


predictionAndLabels = pred.select("prediction", "class_index").rdd.map(lambda row: (row.prediction, float(row.class_index)))
metrics = MulticlassMetrics(predictionAndLabels)
cfmx = metrics.confusionMatrix()
cfarr = cfmx.toArray() 
confusionmatrix = pd.DataFrame(cfarr)

ax= plt.subplot()
sns.set(rc={'figure.figsize':(7,5), 'axes.facecolor':'#363435', 'figure.facecolor':'#363435','axes.labelcolor': 'white',
            'xtick.color': 'white', 'ytick.color': 'white'})
sns.heatmap(confusionmatrix/np.sum(confusionmatrix), annot=True,cmap='PuRd') 
ax.set_xlabel("Predicted labels");
ax.set_ylabel("True labels"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["GALAXY", "STAR", "QUASAR"])
ax.yaxis.set_ticklabels(["GALAXY", "STAR", "QUASAR"])
plt.show()


# In[ ]:




