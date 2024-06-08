# Data exploration

## Dataset

***"Oxford Parkinson's Disease Detection Dataset"***  
*"Little MA, McSharry PE, Hunter EJ, Ramig LO. Suitability of dysphonia measurements for telemonitoring of Parkinson's disease [dataset]. UC Irvine Machine Learning Repository. 2008."*

Repository: https://archive.ics.uci.edu/dataset/174/parkinsons  
Publication: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3051371/

 

This dataset contains vocal measurements of individuals with and without Parkinson's disease. This resource is valuable for studying the potential of acoustics in aiding the diagnosis of Parkinson's disease.

## Dysphonia
Dysphonia is a medical term that refers to difficulty or impairment in voice production, often resulting in changes in voice quality, pitch, loudness, or resonance.  

In individuals with Parkinson's disease, their vocal measurements may indicate dysphonia-related voice irregularities. Inn healthy individuals, the same measurements may contain data reflecting normal voice characteristics without dysphonia.

## Fetching 
***script: data_exploration.py*** 

147 rows (voice recordings) taken from 23 patients with Parkinson's disease.   
48 rows (voice recordings) taken from 8 individuals without Parkinson's disease.

![X_head.png](images/X_head.png)

## Processing 
***script: logic.data_processing.py*** 

There are no missing values. However, there are duplicated columns names.
First step is to rename the duplicates by adding "_2" to the column name.

## Correlation
***script: correlation.py***

![correlation.png](images/correlation.png)

## Decisions

Once the data was understood and explored the following decisions were made:

### Type of Machine Learning

Supervised learning is the chosen approach because the targets are clearly labeled as 0 (no Parkinson) or 1 (Parkinson's positive). The data is labeled, making it suitable for supervised learning techniques.

### Models for comparison

The following model were chosen for performance comparison:

- logistic regression
- random forest
- deep neural network
- maybe: support vector machines(SVM) 

