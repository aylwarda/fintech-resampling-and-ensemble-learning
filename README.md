# fintech-resampling-and-ensemble-learning
In this project I build and evaluate several machine learning models to predict credit risk using sample data (provided to me :)).  The 2 "techniques" used to build models which help with this imbalanced classification problem are Resampling and Ensemble Learning using the SkiKit Learn and Imbalaced Learn libraries.

---
## Instructions / Intro

Within this Project, there are two Jupyter Notebooks.
 - Credit-risk Resampling (`credit_risk_resampling.ipynb`)
 - Credit-risk Ensemble (`credit_risk_ensemble.ipynb`)

To run through either Notebook, execute the one you are interested in, and follow the analysis and conclusions for each. As with other machine learning algorithms and models, they are not particularly mind-blowing in terms of visualisation.  However, they are a slightly more sophisticated look into the start of machine learning and binary (yes/no) decisions/predictions.

The data used in this Notebook is credit-risk related data typically used in determining the 'financial health' of an individual and therefore their ability to continue to service debt; the data is found in `./Resources/lending_data.csv`

## Typical Approach
Typically, when one wants to predict an outcome using the data one has at one's disposal is to: -
1. Assess the data and determine the "target"; that thing one wants to test a model against, i.e. is it a cat or is it a dog.
2. Clean and prepare the data, splitting it up into sets used for "training" the model and sets used to test the prediction
3. With reference to (1.), if the data is imbalanced (which is the case most of the time), then one needs to use a technique to balance it out.
   - resampling is one of these techniques
   - as is the use of more sophisticated models that take imbalance into account, such as ensemble learners
   - both of which are the subject of this project/repository/assignment
4. Train the model one has chosen.
5. Predict the outcome.
6. Evaluate the performance of the model.

---
## Credit-risk Resampling
This Notebook explores the benefits of using a number of resampling models and provides a direct comparison between them. One resamples the data when the data is imbalanced, i.e. the data being used to predict an outcome is highly skewed in one way or another.  
Resampling is a means of ensuring the data used - when "training" a machine learning model - is evaluated equally. This is essential when the "choices" (or put another way, the classification) we are hoping to make are binary, e.g. this or that; yes or no; black or white.  

The classification model used for all resampled data is Logistic Regression.

**Resampling Models Used**
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

## Credit-risk Ensemble Learners
This Notebook explores a slightly different approach as is performed with resampling in that the use of "ensemble learners" is in essence the use of a number of models and techniques that have weeknesses and others that have strengths, when combined, produce better outcomes than would have been produced should they have been used alone.  
In this case 2 classification models were used and in both cases, the models required that the training data be "scaled" as no resampling techniques (as was explored above) were used.  
Therefore these models inherently handle imbalanced data well.

**Classification Models Used**
* Balanced Random Forest Classifier (undersampling)
  * an ensemble model, "in which each tree of the forest [is] provided a balanced bootstrap sample" (see imbalanced learn [here](https://imbalanced-learn.org/stable/ensemble.html#forest)) and then randomly under-samples each bootstrap
* Easy Ensemble classifier (undersampling)
  * an ensemble model "of AdaBoost learners trained on different balanced boostrap samples [with] balancing achieved by random under-sampling" (see imbalanced learn [here](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html))

---
## Acknowledgements
### Sources
- Data source provided by Trinity College as part of course work.
- Sci-kit Learn and Imbalanced Learn as referenced in the introduction and urls here: -
  - https://scikit-learn.org/stable/about.html
  - https://imbalanced-learn.org/stable/about.html 
- I found a lovely way of visualising the Confusion Matrix but with percentages.  I had used the Seaborn Heatmap in my code during class, but what I found useful was a way to display percentages (see example in [this](https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea) Medium article, which has additional sophisticated approaches - which I did not use).


