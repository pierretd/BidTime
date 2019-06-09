# BidTime
Real time bidding framework for digital advertisements

In the online advertising industry, correctly predicting whether an impression will lead to a conversion can lead to massive profits. Google alone serves an average of almost 30 billion impressions per day, meaning that a minuscule improvement in predicting click through rate can create billions of revenue for the company over a year. The research presented in this paper aims to develop a flexible and robust click through rate prediction algorithm based on user data. Time series features pertaining to a data set of impressions on an eCommerce site with 40 million impressions are collected and assigned to two classes(conversion or no conversion) and modeled using a variety of methods. The models include logistic regression, decision trees, random forests, and boosting models. The classification performance of the models is then compared to determine the best method for predicting click through rate.


## Model Summaries
### Performance Metrics
| Model | AUC of ROC | AUC of PRC | MAE | RMSE |
|---|---|---|---|---|
| Logistic Regression(scikit-learn) | 0.717 | 0.335 | 0.255 |0.017  |
| SGD Classifier(scikit-learn) | 0.717 | 0.335 | 0.260 |0.017|

### Classification Reports(on Clicks)
| Model | Precision | Recall | F1 Score | 
|---|---|---|---|---|
|Decision Tree(scikit-learn)  |0.30 | 0.17 | 0.22 |0.017  |
| SGD Classifier(scikit-learn) | 0.717 | 0.335 | 0.260 |
## Index
* Data
  * Azuvu
* Resources
  * Relevant Publications
  * Videos
  * Articles

## Contact
ptdamiba at gmail.com

https://www.linkedin.com/in/pierredamiba/
