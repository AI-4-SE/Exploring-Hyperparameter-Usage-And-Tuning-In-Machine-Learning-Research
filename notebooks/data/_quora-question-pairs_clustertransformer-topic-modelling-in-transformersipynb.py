#!/usr/bin/env python
# coding: utf-8

# ## Cluster Transformer
# 
# 
# <img src="https://i.imgur.com/Fjm01Ca.png">
# 
# 
# [ClusterTransformer](https://github.com/abhilash1910/ClusterTransformer) is a topic clustering/modelling library built on top of [Huggingface Transformers](https://huggingface.co) to cluster/model semantically and contextually similar corpus into common clusters. The consequent loss is determined using cosine similarity after passing the corpus extracts through the forward pass of a generic Transformer module. The cosine metric is chosen as the main metric before applying agglomerative clustering strategies, based on similarity of neighborhood contexts. The library uses pytorch as the framework for the forward pass (dense feed forward network) construction of the embedding tensors.
# 
# <img src="https://venturebeat.com/wp-content/uploads/2019/09/hugging-face.png?w=1200&strip=all">
# 
# 
# 
# 
# ## Working Demo
# 
# 
# In this case, the Quora question pair similarity dataset is taken as the problem statement and instead of using the straightforward way, the Cluster transformer can be applied on a mixed set of the corpus. Instead of testing similarity based on row entries (i.e, question1 and question2), we will first merge all the questions and then try to create the clusters by passing them through the library. Semi supervised embedding weights created fromt the Transformer models can be used to map to the unsupervised clusters in this process. In a way, this is similar to zero shot classification on numeric labels. Since the classification of the clusters entirely depend on the neighborhood of cosine distance/similarity of an embedding eigen vector with respect to the others , the clusters may change for successive iterations.However the contextul similarity should prevail on a larger scale.
# 
# The steps to operate this library is as follows:
# 
# - Initialise the class: ClusterTransformer()
# - Provide the input list of sentences: In this case, the quora similar questions dataframe
#    has been taken for experimental purposes. 
# - Declare hyperparameters: 
#     - batch_size: Batch size for running model inference
#     - max_seq_length: Maximum sequence length for transformer to enable truncation
#     - convert_to_numpy: If  enabled will return the embeddings in numpy ,else will keep in torch.Tensor
#     - normalize_embeddings:If set to True will enable normalization of embeddings.
#     - neighborhood_min_size:This is used for neighborhood_detection method and determines the minimum number of entries in                             each cluster
#     - cutoff_threshold:This is used for neighborhood_detection method and determines the cutoff cosine similarity score to                        cluster the embeddings.
#     - kmeans_max_iter: Hyperparameter for kmeans_detection method signifying nnumber of iterations for convergence.
#     - kmeans_random_state:Hyperparameter for kmeans_detection method signifying random initial state.
#     - kmeans_no_cluster:Hyperparameter for kmeans_detection method signifying number of cluster.    
#     - model_name:Transformer model name ,any transformer from Huggingface pretrained library
#     
# -  Call the methods:
#     -  ClusterTransfomer.model_inference: For creating the embeddings by running inference through 
#        any Transformer library (BERT,Albert,Roberta,Distilbert etc.)Returns a torch.Tensor containing
#        the embeddings.
#     -  ClusterTransformer.neighborhood_detection: For agglomerative clustering from the embeddings created from the 
#         model_inference method.Returns a dictionary.
#     -  ClusterTransformer.kmeans_detection:For Kmeans clustering from the embeddings created from the 
#         model_inference method.Returns a dictionary.
#     - ClusterTransformer.convert_to_df: Converts the dictionary from the neighborhood_detection/kmeans_detection methods in                          a dataframe
#     - ClusterTransformer.plot_cluster:Used for simple plotting of the clusters for each text topic.
# 
# 
# The above mentioned steps are used in this kernel and the effect of different transformers are shown for comparison purposes.

# <img src="https://hips.hearstapps.com/digitalspyuk.cdnds.net/17/47/1511265525-justice-league-poster.jpg?resize=480:*">

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('../input/quora-question-pairs/train.csv.zip')
data.head()


# ## Merge the data into a unified list
# 
# Here we merge the data to provide an input list to the library without any prior context as to which question pair should be evaluated with which question pair. This randomization does not affect the model performance in any case, and is extremely useful to understand which pair shares a similar embedding space.
# 

# In[ ]:


merged_set=[]
for i in data.question1.tolist():
    merged_set.append(i)
for i in data.question2.tolist():
    merged_set.append(i)
merged_set[:5]


# In[ ]:


#Install the library
get_ipython().system('pip install ClusterTransformer==0.1')


# ## Method of working
# 
# The code steps provided in the tab below, represent all the steps required to be done for creating the clusters. The 'compute_topics' method has the following steps:
# 
# - Instantiate the object of the ClusterTransformer
# - Specify the transformer name from [pretrained transformers](https://huggingface.co/transformers/pretrained_models.html)
# - Specify the hyperparameters
# - Get the embeddings from 'model_inference' method
# - For agglomerative neighborhood detection use 'neighborhood_detection' method
# - For kmeans detection, use the 'kmeans_detection' method
# - For converting the dictionary to a dataframe use the 'convert_to_df' method
# - For optional plotting of the clusters w.r.t corpus samples, use the 'plot_cluster' method
# 
# A similar demonstration has also been provided in the [Google Colab](https://colab.research.google.com/drive/18HAoATFfuXGAGzPcOhWgZa0a9B6yOpKK?usp=sharing)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import ClusterTransformer.ClusterTransformer as cluster_transformer\n\ndef compute_topics(transformer_name):\n    \n    #Instantiate the object\n    ct=cluster_transformer.ClusterTransformer()\n    #Transformer model for inference\n    model_name=transformer_name\n    \n    #Hyperparameters\n    #Hyperparameters for model inference\n    batch_size=500\n    max_seq_length=64\n    convert_to_numpy=False\n    normalize_embeddings=False\n    \n    #Hyperparameters for Agglomerative clustering\n    neighborhood_min_size=3\n    cutoff_threshold=0.95\n    #Hyperparameters for K means clustering\n    kmeans_max_iter=100\n    kmeans_random_state=42\n    kmeans_no_clusters=8\n    \n    #Sub input data list\n    sub_merged_sent=merged_set[:200]\n    #Transformer (Longformer) embeddings\n    embeddings=ct.model_inference(sub_merged_sent,batch_size,model_name,max_seq_length,normalize_embeddings,convert_to_numpy)\n    #Hierarchical agglomerative detection\n    output_dict=ct.neighborhood_detection(sub_merged_sent,embeddings,cutoff_threshold,neighborhood_min_size)\n    #Kmeans detection\n    output_kmeans_dict=ct.kmeans_detection(sub_merged_sent,embeddings,kmeans_no_clusters,kmeans_max_iter,kmeans_random_state)\n    #Agglomerative clustering\n    neighborhood_detection_df=ct.convert_to_df(output_dict)\n    #KMeans clustering \n    kmeans_df=ct.convert_to_df(output_kmeans_dict)\n    return neighborhood_detection_df,kmeans_df \n    \n    \n')


# ## Testing with BERT
# 
# 
# <img src="https://miro.medium.com/max/1200/0*pA4-59_AWAYcEKqT.png">
# 
# ## BERT 
# 
# [BERT](https://arxiv.org/abs/1810.04805) is [bidirectional encoder Transformer model](https://github.com/google-research/bert)
# 
# 
# <img src="http://jalammar.github.io/images/distilBERT/bert-output-tensor.png">
# 
# 
# 
# The entire workflow can be designed as follows:
# 
# 
# This image can be used to describe the workflow:
# 
# 
# <img src="http://jalammar.github.io/images/distilBERT/bert-input-to-output-tensor-recap.png">
# 
# 
# Slicing the important part
# For sentence classification, we’re only only interested in BERT’s output for the [CLS] token, so we select that slice of the cube and discard everything else.
# 
# 
# <img src="http://jalammar.github.io/images/distilBERT/bert-output-tensor-selection.png">
# 
# 
# BERT Model
# 
# <img src="https://miro.medium.com/max/740/1*G6PYuBxc7ryP4Pz7nrZJgQ@2x.png">
# 
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nn_df,k_df=compute_topics('bert-large-uncased')\nkg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))\nrng = np.random.RandomState(0)\ns=1000*rng.rand(len(kg_df['Text']))\ns1=1000*rng.rand(len(ng_df['Text']))\nax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)\nax1.set_title('Kmeans clustering')\nax1.set_xlabel('No of clusters')\nax1.set_ylabel('No of topics')\nax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)\nax2.set_title('Agglomerative clustering')\nax2.set_xlabel('No of clusters')\nax2.set_ylabel('No of topics')\nplt.show()\n")


# In[ ]:


#Dataframe Clustered with Agglomerative method
n_df.head()


# In[ ]:


#Dataframe Clustered with Kmeans method
k_df.head()


# ## Implication of Clustering Graphs
# 
# The clustering graphs specify which cluster a particular topic belong to and also the count of topics present in a cluster.
# For this there are some different outputs:
# 
# - For the Kmeans clustering , it tries to produce all the input corpus extracts in the graph.
# - For the Agglomerative clustering, the number of clusters produced depends on the 'neighborhood_min_size'. If a default size of 1 is provided then the graph will try to accommodate all the topics in the graph. However if a value of greater than 1 is provided (e.g. 10), then only those clusters which have a size/count of 10 items inside it will be chosen and ranked accordingly in the graph. The graph in this case is strictly non increasing.In the above example,a min size of 3 items per cluster with a similarity score of 0.95 produces 2 clusters (denoted by the purple- label 0 containing 5 items and yellow - label 1 containing 3 items)

# ## DistilBERT Model
# 
# 
# The distilbert performs better than Bert in most cases owing to continuous feedback of attention weights from the teacher to the student network. Where the weights change by a large extent in case of Bert, this fails to happen in DistilBert.
# 
# 
# <img src="https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_391208%2Fimages%2FKD_figures%2Ftransformer_distillation.png">
# 
# 
# DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark. 2 significant benchmarks aspects of this Model:
# 
# - Quantization :This leads to approximation of internal weight vectors to a numerically smaller precision
# - Weights Pruning: Removing some connections from the network.
# 
# Knowledge distillation (sometimes also referred to as teacher-student learning) is a compression technique in which a small model is trained to reproduce the behavior of a larger model (or an ensemble of models). It was introduced by Bucila et al. and generalized by Hinton et al. a few years later. We will follow the latter method.Rather than training with a cross-entropy over the hard targets (one-hot encoding of the gold class), we transfer the knowledge from the teacher to the student with a cross-entropy over the soft targets (probabilities of the teacher). Our training loss thus becomes:
# 
# <img src="https://miro.medium.com/max/311/1*GZkQPjKC_Wqx1F4Uu3FdiQ.png">
# 
# This loss is a richer training signal since a single example enforces much more constraint than a single hard target.
# To further expose the mass of the distribution over the classes, Hinton et al. introduce a softmax-temperature:
# 
# <img src="https://miro.medium.com/max/291/1*BaVyKMXRWaudFvcI9So8MQ.png">
# 
# When T → 0, the distribution becomes a Kronecker (and is equivalent to the one-hot target vector), when T →+∞, it becomes a uniform distribution. The same temperature parameter is applied both to the student and the teacher at training time, further revealing more signals for each training example. At inference, T is set to 1 and recover the standard Softmax.
# 
# 
# Some resources:
# 
# - [Blog](https://medium.com/huggingface/distilbert-8cf3380435b5)
# - [Huggingface](https://huggingface.co/transformers/model_doc/distilbert.html)
# - [Paper](https://arxiv.org/abs/1910.01108)

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nn_df,k_df=compute_topics('distilbert-base-uncased')\nkg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))\nrng = np.random.RandomState(0)\ns=1000*rng.rand(len(kg_df['Text']))\ns1=1000*rng.rand(len(ng_df['Text']))\nax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)\nax1.set_title('Kmeans clustering')\nax1.set_xlabel('No of clusters')\nax1.set_ylabel('No of topics')\nax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)\nax2.set_title('Agglomerative clustering')\nax2.set_xlabel('No of clusters')\nax2.set_ylabel('No of topics')\nplt.show()\n")


# In[ ]:


#Dataframe Clustered with Agglomerative method
n_df.head()


# In[ ]:


#Dataframe Clustered with Kmeans method
k_df.head()


# ## XLM Roberta/Roberta
# 
# 
# [XLM](https://arxiv.org/pdf/1907.11692.pdf) builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.
# 
# <img src="https://camo.githubusercontent.com/f5c0d05eb0635cdd0e17e137265af23fa825b1d4/68747470733a2f2f646c2e666261697075626c696366696c65732e636f6d2f584c4d2f786c6d5f6669677572652e6a7067">Tips:
# 
# 
# This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained models.
# 
# RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.
# 
# RoBERTa doesn’t have token_type_ids, you don’t need to indicate which token belongs to which segment. Just separate your segments with the separation token tokenizer.sep_token (or </s>)
# 
# CamemBERT is a wrapper around RoBERTa.
# 
# 
# Resources:
# 
# - [FAIR](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)
# - [Pytorch](https://pytorch.org/hub/pytorch_fairseq_roberta/)
# - [Github](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
# - [Huggingface](https://huggingface.co/transformers/model_doc/roberta.html)

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nn_df,k_df=compute_topics('roberta-large')\nkg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))\nrng = np.random.RandomState(0)\ns=1000*rng.rand(len(kg_df['Text']))\ns1=1000*rng.rand(len(ng_df['Text']))\nax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)\nax1.set_title('Kmeans clustering')\nax1.set_xlabel('No of clusters')\nax1.set_ylabel('No of topics')\nax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)\nax2.set_title('Agglomerative clustering')\nax2.set_xlabel('No of clusters')\nax2.set_ylabel('No of topics')\nplt.show()\n")


# In[ ]:


#Dataframe Clustered with Agglomerative method
n_df.head()


# In[ ]:


#Dataframe Clustered with Kmeans method
k_df.head()


# ## Albert Transformer
# 
# 
# 
# <img src="data:image/jpeg;base64,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">
# 
# 
# [Albert](https://arxiv.org/abs/1909.11942) is a lightweight bert which introduces parameter sharing, caching, and intermediate repeated splitting of the embedding matrix for efficient modelling tasks.
# 
# According to the paper:
# 
# 
# 'The first one is a factorized embedding parameterization. By decomposing
# the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden
# layers from the size of vocabulary embedding. This separation makes it easier to grow the hidden
# size without significantly increasing the parameter size of the vocabulary embeddings. The second
# technique is cross-layer parameter sharing. This technique prevents the parameter from growing
# with the depth of the network. Both techniques significantly reduce the number of parameters for
# BERT without seriously hurting performance, thus improving parameter-efficiency. An ALBERT
# configuration similar to BERT-large has 18x fewer parameters and can be trained about 1.7x faster.
# The parameter reduction techniques also act as a form of regularization that stabilizes the training
# and helps with generalization.
# To further improve the performance of ALBERT, we also introduce a self-supervised loss for
# sentence-order prediction (SOP). SOP primary focuses on inter-sentence coherence and is designed
# to address the ineffectiveness (Yang et al., 2019; Liu et al., 2019) of the next sentence prediction
# (NSP) loss proposed in the original BERT.'
# 
# 
# Resources:
# 
# - [Github](https://github.com/google-research/albert)
# - [Huggingface](https://huggingface.co/transformers/model_doc/albert.html)
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nn_df,k_df=compute_topics('albert-base-v1')\nkg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))\nrng = np.random.RandomState(0)\ns=1000*rng.rand(len(kg_df['Text']))\ns1=1000*rng.rand(len(ng_df['Text']))\nax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)\nax1.set_title('Kmeans clustering')\nax1.set_xlabel('No of clusters')\nax1.set_ylabel('No of topics')\nax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)\nax2.set_title('Agglomerative clustering')\nax2.set_xlabel('No of clusters')\nax2.set_ylabel('No of topics')\nplt.show()\n")


# In[ ]:


#Dataframe Clustered with Agglomerative method
n_df.head()


# In[ ]:


#Dataframe Clustered with Kmeans method
k_df.head()


# ## BART Model
# 
# [This](https://arxiv.org/abs/1910.13461) is alternate SOTA model to denoise sentence2 sentence pretraining for natural language generation,comprehension etc. The most important points can be summarized as:
# 
# 
# - Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT).
# 
# - The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.
# 
# - BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.
# 
# The architecture contains these encoder -decoder modules :
# <img src="https://miro.medium.com/max/3138/1*Qss9gtS1nw_sgcG1pMAM2A.png">
# 
# <img src="https://miseciara.files.wordpress.com/2013/11/bart.gif">
# 
# 
# Some resources:
# - [Blog](https://medium.com/dair-ai/bart-are-all-pretraining-techniques-created-equal-e869a490042e)
# - [Blog-BART](https://medium.com/analytics-vidhya/revealing-bart-a-denoising-objective-for-pretraining-c6e8f8009564)

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nn_df,k_df=compute_topics('facebook/bart-large')\nkg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))\nrng = np.random.RandomState(0)\ns=1000*rng.rand(len(kg_df['Text']))\ns1=1000*rng.rand(len(ng_df['Text']))\nax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)\nax1.set_title('Kmeans clustering')\nax1.set_xlabel('No of clusters')\nax1.set_ylabel('No of topics')\nax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)\nax2.set_title('Agglomerative clustering')\nax2.set_xlabel('No of clusters')\nax2.set_ylabel('No of topics')\nplt.show()\n")


# In[ ]:


#Dataframe Clustered with Agglomerative method
n_df.head()


# In[ ]:


#Dataframe Clustered with Kmeans method
k_df.head()


# ## Longformer
# 
# 
# The Longformer model was presented in Longformer: [The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
# 
# The abstract from the paper is the following:
# 
# Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA.
# 
# Tips:
# 
# Since the Longformer is based on RoBERTa, it doesn’t have token_type_ids. You don’t need to indicate which token belongs to which segment. Just separate your segments with the separation token tokenizer.sep_token (or </s>).
# 
# - [Repository](https://github.com/allenai/longformer)
# 
# 
# <img src="https://t-dab.com/wp-content/uploads/2020/11/alex-300x269.png">

# In[ ]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\nn_df,k_df=compute_topics('allenai/longformer-base-4096')\nkg_df=k_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nng_df=n_df.groupby('Cluster').agg({'Text':'count'}).reset_index()\nfig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))\nrng = np.random.RandomState(0)\ns=1000*rng.rand(len(kg_df['Text']))\ns1=1000*rng.rand(len(ng_df['Text']))\nax1.scatter(kg_df['Cluster'],kg_df['Text'],s=s,c=kg_df['Cluster'],alpha=0.3)\nax1.set_title('Kmeans clustering')\nax1.set_xlabel('No of clusters')\nax1.set_ylabel('No of topics')\nax2.scatter(ng_df['Cluster'],ng_df['Text'],s=s1,c=ng_df['Cluster'],alpha=0.3)\nax2.set_title('Agglomerative clustering')\nax2.set_xlabel('No of clusters')\nax2.set_ylabel('No of topics')\nplt.show()\n")


# In[ ]:


#Dataframe Clustered with Agglomerative method
n_df.head()


# In[ ]:


#Dataframe Clustered with Kmeans method
k_df.head()


# ## Using with other Transformers
# 
# This library can be used with other variants of Transformers, present in this [link](https://huggingface.co/transformers/pretrained_models.html). Encoder Decoder models like T5 and Pegasus are not supported yet, and performance of Generative models like GPT can be tested as well.
# 
# Library details:
# 
# - [Github](https://github.com/abhilash1910/ClusterTransformer)
# - [Pypi](https://pypi.org/project/ClusterTransformer/)
# 
# <img src="https://redefined.s3.us-east-2.amazonaws.com/wp-content/uploads/2020/05/20151235/justiceleague_wbmoviestillsdb.jpg">
