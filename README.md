# fintech-resampling-and-ensemble-learning
In this project I build and evaluate several machine learning models to predict credit risk using sample data (provided to me :)).  The 2 "techniques" used to build models which help with this imbalanced classification problem are Resampling and Ensemble Learning using the SkiKit Learn and Imbalaced Learn libraries.

---
## Instructions / Intro

Within this Project, there are two Jupyter Notebooks.
 - Credit-risk Resampling (`credit_risk_resampling.ipynb`)
 - Credit-risk Ensemble (`credit_risk_ensemble.ipynb`)

To run through either Notebook, execute the one you are interested in, and follow the analysis and conclusions for each.  These are not particularly mind-blowing in terms of visualisation.
They are a slightly more sophisticated look into the start of machine learning and binary (yes/no) decisions/predictions.

The data used in this Notebook is credit-risk related data typically used in determining the 'financial health' of an individual and therefore their ability to continue to service debt; the data is found in `./Resources/lending_data.csv`

---
## Credit-risk Resampling
This Notebook explores the benefits of using a number of resampling models and provides a direct comparison between them. One resamples the data when the data is imbalanced, i.e. the data being used to predict an outcome is highly skewed in one way or another.  
Resampling is a means of ensuring the data used - when "training" a machine learning model - is evaluated equally. This is essential when the "choices" (or put another way, the classification) we are hoping to make are binary, e.g. this or that; yes or no; black or white.

**Models Used**
* Simple Scaled sampling
  * a means of ensuring the data is equally scaled, e.g. numbers that could cause imbalance like 1 billion vs. 1 million, are brought into alignment by scaling them accordingly
* Naive Random Oversampling
  * a means of adding data to a sample to even it out randomly
* SMOTE Oversampling
  * a means of adding data to a sample to even it out more precisely, i.e. making the sampling more meaningful
* Cluster Centroids Undersampling
  * a means of *removing* data from a sample to even it out in a fairly sophisticated manner (the model requires quite a lot of processing power)
* SMOTE_ENN Combination sampling
  * a means of *both* adding and removing data to both "sides" of the sample to even it out

## Credit-risk Ensemble
This Notebook explores ...

---
## Acknowledgements
### Sources
- Data source provided by Trinity College as part of course work.
- Sci-kit Learn and Imbalanced Learn as referenced in the introduction and urls here: -
  - https://scikit-learn.org/stable/about.html
  - https://imbalanced-learn.org/stable/about.html 
- I found a lovely way of visualising the Confusion Matrix but with percentages.  I had used the Seaborn Heatmap in my code, but what I found useful was the way to display percentages (see example in [this](https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea) Medium article)


