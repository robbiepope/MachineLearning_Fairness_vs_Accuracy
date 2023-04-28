# MachineLearning_Fairness_vs_Accuracy
An investigation into the effect of regularisation on the fairness versus accuracy trade off for a logistic regression machine learning model. Completed as part of a Masters in Data Science
The notebook, Machine_Learning_Code_244804.ipynb was developed using Google Colab. 
For best results run this notebook using Google Colab. 

To run the notebook simply select ‘Run all’ - there is no additional code that needs to be input.
The notebook takes ~45 minutes to run all sections using a standard Google Colab account. 

# Introduction
Fairness studies in the field of machine learning have both empirically proven and quantified discrimination in models trained on data containing sensitive features [1]. When discrimination exists in the dataset, machine learning models can further exacerbate this disparity during classification, learning the pattern that sensitive features correspond to the most common or discriminatory outcomes [2]. As Cathy O’Neill said during her presentation at Talks at Google “Algorithms are nothing more than opinions embedded in code” [3]. This has led to the development of fair machine learning approaches that aim to mitigate discrimination by specific pre-processing, in-processing and post-processing techniques [1].

The investigstion explores how varying the hyperparameters of a standard logistic regression model and a fairness-based (reweighed) logistic regression model impacts the accuracy and fairness of the final outcome. The investigation also investigates the efficiency of the proposed criterion (trade-off metric) for hyperparameter selection of a model that accounts for both accuracy and fairness. Finally, the performance is analysed when the model is trained with data that has had the sensitive features suppressed.

# Methodology
The two datasets analysed were downloaded using the pre- processed functions supplied by AIF360. The datasets both contained two sensitive features and binary labels. The datasets were split into training (70%) and test (30%) subsets.
Adult Census Income Dataset [4]
- Predict whether a person makes over 50K a year 
- Sensitive features = Race and Sex
- N = 48,842

German Credit Dataset [5]
- Predict whether a person has good or bad credit 
- Sensitive features = Age and Sex
- N = 1,000

The logistic regression model was implemented as a fully connected linear layer with the sigmoid activation function using the PyTorch framework. 
The AIF360 toolbox was used to calculate the equality of opportunity fairness metric and apply the reweighing pre-processing technique.

# Results
Hyperparameter testing for a standard logistic regression model for both datasets showed that the more accurate models were more unfair. After reweighing the dataset, the model fairness was greatly improved whilst maintaining a high level of accuracy. Better results were observed with the Adult dataset compared to the German dataset which was much smaller in size. The German dataset is imbalanced which meant the model was able to achieve high accuracy simply by predicting the dominant label each time. One way to mitigate this with smaller datasets could be to augment the dataset before training.

# Conclusion 
The implementation of a fairness-based processing method when deploying a machine learning model is an important aspect to consider. This investigation demonstrates how fairness can be improved with minimum accuracy trade-offs either by reweighing or removing the sensitive feature(s) from the dataset. The proposed trade-off metric did not result in fairer models for the standard logistic regression model; however, it did mimic the hyperparameters for the most accurate fairness-based models. Future considerations may investigate additional fairness metrics to understand the impact across a range of metrics.

# References 
- [1] Scantamburlo, T., Non-empirical problems in fair machine learning. Ethics and Information Technology, 2021. 23(4): p. 703-712.
- [2] Nielsen, A. Fairness Pre-Processing. In: Practical Fairness. Sebastopol, California, United States: O'Reilly Media, Inc., 2020, pp.1-175. [Online] Available at: https://www.oreilly.com/library/view/practical- fairness/9781492075721/ch04.html [Accessed 10/05/2022]
- [3] Google, T.a., Weapons of Math Destruction | Cathy 552 O'Neil | Talks at Google. 2016. [Online] Available at: 553 https://www.youtube.com/watch?v=TQHs8SA1qpk 554 [Accessed 01/05/2022]
- [4] Kohavi, R. and B. Becker. Adult Census Income 578 Dataset. 1996 [Online] Available from: 579 https://archive.ics.uci.edu/ml/datasets/Adult [Accessed 18/04/2022]
- [5] Hofmann, H. Statlog (German Credit Data) Data Set. 581 1994 [Online] Available from: 582 https://archive.ics.uci.edu/ml/datasets/Statlog+%28Ger 583 man+Credit+Data%29 [Accessed 20/04/2022]
