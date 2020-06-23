# Welcome to *Clairvoyant Clustering of Consumer Complaints*!
This project is a multi-faceted approach to analyzing text and clustering companies identified in consumer complaints filed with the Consumer Financial Protection Bureau. Tools and techniques include natural language processing (SpaCy and CorEx) and unsupervised machine learning (K-Means and PCA)

The supporting video presentation can be found on [YouTube]([TBD]).

## Notebook #1 EDA and SpaCy
The dataset is downloaded from the Consumer Financial Protection Bureau's (CFPB) [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/search/?dataLens=Overview&dataNormalization=None&dateInterval=Month&dateRange=3y&date_received_max=2020-06-21&date_received_min=2017-06-21&from=0&page=1&searchField=all&size=25&sort=created_date_desc&tab=Map). Data filters used for this project:
- Date CFPB received the complaint: from 6/1/19 through 5/31/20
- Product: Debt collection

The main focus is on the consumer complaint text (also called 'consumer complaint narrative'). After an initial split and some EDA, the text is lightly cleaned. 

**SpaCy** is used for high-speed text pre-processing including: tokenization, lemmatization, part-of-speech tagging and dependency parsing. Looking at only nouns and adjectives in the complaint text does not provide as much detail as desired so the project proceeded with including words in all parts-of-speech. 

## Notebook #2 [CorEx](https://github.com/gregversteeg/corex_topic) (Correlation Explanation) for topic modeling 
In preparation for topic modeling, the complaint text is vectorized with CountVectorizer. Note that the **CorEx** topic model uses binary count vectors as input.

After a first round of topic modeling using CorEx in an unsupervised manner, some interesting words were selected to be used as 'anchor words' and the CorEx model was re-run in a semi-supervised fashion until topics were clean. 

A document-topic probability matrix is generated with the clean topics. CorEx is a discriminative model (whereas [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative model). This means that while LDA outputs a probability distribution over each document, CorEx instead estimates the probability a document belongs to a topic given that document's words. As a result, the probabilities across topics for a given document do not have to add up to 1. 

A company-topic probability matrix is then generated using an average of the estimated probabilities of topics for each document that belongs to a Company

## Notebook #3 K-Means for clustering 
Prior to clustering with the **K-Means** algorithm, a scree plot is created to determine the optimum number of clusters. 

Cluster centers are evaluated to determine strongest topics in each cluster.

## Notebook #4 PCA for visualization
**PCA** is used to visualize the K-Means clusters in two dimensions.
