# Credit Risk Analysis 
*** Using Logistic Regression and Ensemble Learning to Predict Credit Risk ***
#### by Justin R. Papreck
---

## Overview

Analyzing credit risk is a difficult task, as there are a plethora of factors that can impact credit risk compounded with the fact that the number of good loans far outnumbers the number of risky loans, providing a very unbalanced dataset. Using different techniques in machine learning, this project attempts to find the best predictive method to determine which characteristics have the greatest correlation with high-risk loans to allow a lending company to better assess potential borrowers. The methods applied in this report are logistic regression with oversampling, undersampling, combinatorial sampling, random forests and bag of boosters. 

Since one of the biggest purchases many individuals make is a home, the analysis will consider the the financial impact of a loan based on the median cost and average 30-year mortgage rates of a home in California as of September, 2022. When analyzing high and low-risk applicants, the amount of risk the lender must consider is based on the lending amounts and interest. Erroneously a few high-risk clients $8,000 based on a bad model is not going to have the same impact as lending a single high-risk client $8,000,000 to purchase a modest home in La Jolla. The current median cost of a home in California is $900,000 with a current average mortgage interest rate of 6.33% on a 30-year loan.

---
## Methods
*** Logistic Regression with Different Sampling Applications ***

### Oversampling
---

Two oversampling algorithms were used to determine which performed better: a naive random oversampling algorithm and the SMOTE algorithm (Synthetic Minority Oversampling Technique). The naive random sampling was performed using RandomOverSampler from imbalanced-learn to resample the data, and the scikit-learn LogisticRegression model was used to fit the training data. For all logistic regressions, the solver used was the 'lbfgs' solver (default). A random state of 1

The data were analyzed using the balanced accuracy score and imbalanced classification score. (Data cleaning is shown in Index)

#### Naive Random Oversampling
---

Random oversampling takes the minority class, in this case the high-risk loans, and randomly selects records from the minority data and adds them to the training set until the majority and minority classes are balanced. 


```python
# Resample the training data with the RandomOversampler

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)

# Traininng the model
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Calculated the balanced accuracy score
print(f"The Balanced Accuracy Score is {balanced_accuracy_score(y_test, y_pred)*100:,.2f}%")

# Display confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(model, X_test, y_test, display_labels=["High Risk", "Low Risk"], cmap='Blues', values_format='d', ax=ax)
plt.title('Naive Random Oversampling Confusion Matrix')
plt.show()

# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))



```

    The Balanced Accuracy Score is 65.44%
    


    
![png](credit_risk_report_files/credit_risk_report_2_1.png)
    


                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.70      0.61      0.02      0.65      0.43       101
       low_risk       1.00      0.61      0.70      0.75      0.65      0.42     17104
    
    avg / total       0.99      0.61      0.70      0.75      0.65      0.42     17205
    
    

#### SMOTE Oversampling
---

SMOTE oversampling is intended to improve random oversampling by creating synthetic data points based on the original data points. It also sounds very impressive when you refer to it as the Synthetic Minority Oversampling TEchnique to people unfamiliar with it. This process uses a nearest-neighbors approach to interpolate the new data points, potentially improving accuracy. One drawback to this method is the vulnerability to outliers - as this process may create new data points near existing extreme outliers.  


```python
# SMOTE Resampling of the training data
X_resampled, y_resampled = SMOTE(
    random_state=1, sampling_strategy='auto').fit_resample(X_train, y_train)
Counter(y_resampled)

# Logistic Regression of the SMOTE resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Calculated the balanced accuracy score
print(
    f"The Balanced Accuracy Score is {balanced_accuracy_score(y_test, y_pred)*100:,.2f}%")

# print(confusion_matrix(y_test, y_pred))
# Display confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(model, X_test, y_test, display_labels=[
                      "High Risk", "Low Risk"], cmap='Blues', values_format='d', ax=ax)
plt.title('SMOTE Oversampling Confusion Matrix')
plt.show()


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

    The Balanced Accuracy Score is 65.81%
    


    
![png](credit_risk_report_files/credit_risk_report_4_1.png)
    


                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.63      0.68      0.02      0.66      0.43       101
       low_risk       1.00      0.68      0.63      0.81      0.66      0.43     17104
    
    avg / total       0.99      0.68      0.63      0.81      0.66      0.43     17205
    
    

### Undersampling
---

The process of oversampling amplified the minority to meet the size of the majority group. Undersampling does the opposite and curtails the size of the majority group to that of the size of the minority group. While this maintains the fidelity of the data by only using existing data, it is only very applicable if there is enough data in the minority set to work with. In very small datasets, too much of the majority set will be lost to make accurate predictions. Similar to the oversampling, there are different undersampling algorithms, including random sampling. In this case, a Cluster Centroid Undersampling technique was used, again intended to improve upon the inaccuracies of random sampling. This algorithm identifies cluseters of the majority class and generates synthetic data points, as did SMOTE, but in this case as an extrapolation of multiple existing data points to interpolate as a representative of the cluster. This allows the majority class to be sampled down to the size of the minority class. 


```python
# ClusterCentroid resampling of the training data 
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)

# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Calculated the balanced accuracy score
print(
    f"The Balanced Accuracy Score is {balanced_accuracy_score(y_test, y_pred)*100:,.2f}%")

# print(confusion_matrix(y_test, y_pred))
# Display confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(model, X_test, y_test, display_labels=[
                      "High Risk", "Low Risk"], cmap='Blues', values_format='d', ax=ax)
plt.title('Centroid Cluster Confusion Matrix')
plt.show()


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))

```

    The Balanced Accuracy Score is 52.83%
    


    <Figure size 432x360 with 0 Axes>



    
![png](credit_risk_report_files/credit_risk_report_6_2.png)
    


                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.64      0.41      0.01      0.52      0.27       101
       low_risk       0.99      0.41      0.64      0.58      0.52      0.26     17104
    
    avg / total       0.99      0.41      0.64      0.58      0.52      0.26     17205
    
    

### Combination Sampling
---

The SMOTEENN algorithm applies a combination of both undersampling and oversampling. SMOTEEN uses SMOTE and Edited Nearest Neighbors (ENN) as a two-step process, starting with the oversampling of the minority class with SMOTE and cleaning the resulting data by undersampling - if the two nearest neighbors of a data point belong to two different classes, the point is dropped. 


```python
# SMOTEENN resampling of the training data
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)

# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Calculated the balanced accuracy score
print(
    f"The Balanced Accuracy Score is {balanced_accuracy_score(y_test, y_pred)*100:,.2f}%")

# print(confusion_matrix(y_test, y_pred))
# Display confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(model, X_test, y_test, display_labels=[
                      "High Risk", "Low Risk"], cmap='Blues', values_format='d', ax=ax)
plt.title('SMOTEENN Confusion Matrix')
plt.show()


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))

```

    The Balanced Accuracy Score is 64.74%
    


    <Figure size 432x360 with 0 Axes>



    
![png](credit_risk_report_files/credit_risk_report_8_2.png)
    


                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.73      0.56      0.02      0.64      0.42       101
       low_risk       1.00      0.56      0.73      0.72      0.64      0.40     17104
    
    avg / total       0.99      0.56      0.73      0.71      0.64      0.40     17205
    
    

## *** Ensemble Learning with Random Forests and Boosting ***
---

### Balanced Random Forest Classifier
--- 

The Balanced Random Forest Classifier is a modification of the Random Forest Classifier. Random Forest Classification uses an ensemble of decision tree models to classify outcomes. While the individual decision trees may provide poor outcomes, the assembly of many of these trees into a 'forest' can accumulate the predictions from these weak learners to provide a collection of decision trees with controlled variance to generate reasonable predictions across a wide range of data, and thus this allows for the better predictions. Unfortunately, while the standard random forest algorithm can make good predictions even on imbalanced data, it has difficulty making predictions on extremely unbalanced data, since the probability of the decision trees even containing a predictor of the minority category is extremely low. 

The Balanced Random Forest Classifier uses undersampling of the majority class in each bootstrap sample to change the class distribution. This alleviates the issue of having such extreme unbalanced data, which is seen with the high-risk loans. This also lets us look at the features from the dataset and rank them in order of feature importance in the decision trees. 



```python
# Create the Balanced Random Forest Classifier and train the model
barf = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
barf.fit(X_train, y_train)
y_pred = barf.predict(X_test)

# Calculated the balanced accuracy score
print(
    f"The Balanced Accuracy Score is {balanced_accuracy_score(y_test, y_pred)*100:,.2f}%")

# print(confusion_matrix(y_test, y_pred))
# Display confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(barf, X_test, y_test, display_labels=[
                      "High Risk", "Low Risk"], cmap='Blues', values_format='d', ax=ax)
plt.title('Balanced Random Forest Confusion Matrix')
plt.show()


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))

# List the features sorted in descending order by feature importance
sorted(zip(barf.feature_importances_,X.columns), reverse=True)
```

    The Balanced Accuracy Score is 74.12%
    


    
![png](credit_risk_report_files/credit_risk_report_10_1.png)
    


                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.03      0.61      0.87      0.05      0.73      0.52       101
       low_risk       1.00      0.87      0.61      0.93      0.73      0.55     17104
    
    avg / total       0.99      0.87      0.62      0.92      0.73      0.55     17205
    
    




    [(0.08230994769510414, 'total_rec_prncp'),
     (0.07056965445399586, 'last_pymnt_amnt'),
     (0.06697574557210484, 'total_rec_int'),
     (0.0477260885777328, 'total_pymnt'),
     (0.04670837607659922, 'total_pymnt_inv'),
     (0.033015591804214164, 'issue_d'),
     (0.032811236582933494, 'int_rate'),
     (0.020352914833844552, 'mths_since_recent_inq'),
     (0.01878397328527505, 'avg_cur_bal'),
     (0.017975815460573304, 'out_prncp_inv'),
     (0.016145477977071974, 'tot_hi_cred_lim'),
     (0.016068713178803082, 'all_util'),
     (0.01584726050311816, 'out_prncp'),
     (0.01579798002260119, 'dti'),
     (0.0157533477018616, 'revol_bal'),
     (0.014690497594271377, 'mths_since_rcnt_il'),
     (0.014419789614312933, 'bc_open_to_buy'),
     (0.014325214246275421, 'installment'),
     (0.013896430363045737, 'mo_sin_old_rev_tl_op'),
     (0.013810675218325463, 'total_bc_limit'),
     (0.013666604450997676, 'total_rev_hi_lim'),
     (0.01366221345876921, 'tot_cur_bal'),
     (0.01336127669303002, 'annual_inc'),
     (0.013274027815055357, 'max_bal_bc'),
     (0.012877672946311107, 'mo_sin_old_il_acct'),
     (0.012775673514431742, 'il_util'),
     (0.012655711191830867, 'bc_util'),
     (0.01221732318662771, 'mths_since_recent_bc'),
     (0.0121802464150522, 'total_bal_ex_mort'),
     (0.011734883637619344, 'mo_sin_rcnt_tl'),
     (0.01170488213737501, 'total_bal_il'),
     (0.010897771188878462, 'total_il_high_credit_limit'),
     (0.010642777653893383, 'num_il_tl'),
     (0.01050655299170567, 'loan_amnt'),
     (0.010426668657475791, 'inq_fi'),
     (0.009941465855622058, 'mo_sin_rcnt_rev_tl_op'),
     (0.009768253128842013, 'acc_open_past_24mths'),
     (0.009709183155810978, 'num_rev_accts'),
     (0.00964393780978882, 'inq_last_12m'),
     (0.00952116913376898, 'num_actv_rev_tl'),
     (0.0092528039720755, 'total_acc'),
     (0.009167422028972605, 'pct_tl_nvr_dlq'),
     (0.008940691877737876, 'num_sats'),
     (0.008338851448864475, 'num_bc_tl'),
     (0.008071963526217607, 'total_cu_tl'),
     (0.008041810883117178, 'num_actv_bc_tl'),
     (0.007682141971822379, 'open_acc'),
     (0.007519405662023797, 'num_rev_tl_bal_gt_0'),
     (0.0074712768581656815, 'num_op_rev_tl'),
     (0.007362357456480017, 'percent_bc_gt_75'),
     (0.007193457679474712, 'total_rec_late_fee'),
     (0.006848369034211092, 'mort_acc'),
     (0.006833371674604838, 'open_act_il'),
     (0.005958990768199446, 'open_il_24m'),
     (0.0056203226103396465, 'open_acc_6m'),
     (0.005505924234849184, 'num_bc_sats'),
     (0.005293021724728219, 'num_tl_op_past_12m'),
     (0.005172444978051804, 'next_pymnt_d'),
     (0.005050226079710326, 'open_il_12m'),
     (0.005028134267923555, 'open_rv_12m'),
     (0.004855614104465087, 'open_rv_24m'),
     (0.004302470053544381, 'inq_last_6mths'),
     (0.004144286320078212, 'num_accts_ever_120_pd'),
     (0.0038569096067066107, 'tot_coll_amt'),
     (0.0027293344174567974, 'delinq_2yrs'),
     (0.0025686324105894053, 'home_ownership_RENT'),
     (0.0022140735742479612, 'home_ownership_OWN'),
     (0.002018771605501733, 'verification_status_Source Verified'),
     (0.0018166744274413973, 'application_type_Joint App'),
     (0.0014388734314764835, 'pub_rec'),
     (0.0013764781275519278, 'home_ownership_MORTGAGE'),
     (0.0013221111882989905, 'verification_status_Verified'),
     (0.001309899659193596, 'verification_status_Not Verified'),
     (0.0012085300073598, 'application_type_Individual'),
     (0.00111864450224645, 'initial_list_status_f'),
     (0.0010516445463903646, 'initial_list_status_w'),
     (0.0008884603922790834, 'num_tl_90g_dpd_24m'),
     (0.0008299129656903105, 'pub_rec_bankruptcies'),
     (0.0005481645065361378, 'collections_12_mths_ex_med'),
     (0.0004805700459645383, 'chargeoff_within_12_mths'),
     (0.00041392958446006704, 'home_ownership_ANY'),
     (0.0, 'tax_liens'),
     (0.0, 'recoveries'),
     (0.0, 'pymnt_plan_n'),
     (0.0, 'policy_code'),
     (0.0, 'num_tl_30dpd'),
     (0.0, 'num_tl_120dpd_2m'),
     (0.0, 'hardship_flag_N'),
     (0.0, 'delinq_amnt'),
     (0.0, 'debt_settlement_flag_N'),
     (0.0, 'collection_recovery_fee'),
     (0.0, 'acc_now_delinq')]



### Easy Ensemble AdaBoost Classifier
--- 

The Easy Ensemble Classifier creates balanced samples of the training data by collecting all of the data from the minority set and creates a random subset of equal size from the majority set, a process known as random undersampling. It then uses the AdaBoost algorithm to correct for errors made in each subsequent dataset, paying more attention to misclassified and less attention to correctly classified records. A new tree is added on the weighted dataset to correct the errors.

Similar to the Balanced Random Forests, the random undersampling of the Easy Ensemble allows for more accurate predictions in extremely unbalanced data.  


```python
# Create the Easy Ensemble Classifier and train the model
easy_e = EasyEnsembleClassifier(n_estimators=100, random_state=1)
easy_e.fit(X_train, y_train)
y_pred = easy_e.predict(X_test)

# Calculated the balanced accuracy score
print(
    f"The Balanced Accuracy Score is {balanced_accuracy_score(y_test, y_pred)*100:,.2f}%")

# print(confusion_matrix(y_test, y_pred))
# Display confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(easy_e, X_test, y_test, display_labels=[
                      "High Risk", "Low Risk"], cmap='Blues', values_format='d', ax=ax)
plt.title('Easy Ensemble Confusion Matrix')
plt.show()


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

    The Balanced Accuracy Score is 93.21%
    


    
![png](credit_risk_report_files/credit_risk_report_12_1.png)
    


                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
       low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104
    
    avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
    
    

## Results
---



***The Boring Financial Scenario***

The current median price of a home in California as of September, 2022 is $900,000 with an average mortgage rate 0f 6.33% on a 30-year fixed-rate loan. In this analysis, there are 17,205 applicants in the test group, 101 of which are high-risk borrowers. To set up the worst-case scenario - loans are provided to all 101 high-risk applicants, the question becomes how much money will the lender lose? 

Since this is setting up purchasing of homes, assuming that the buyer is required to put 10% down, the loan amount per borrower would be $810,000. If none of these borrowers pay back anything, effectively the lender would lose $81,810,000. The problem lies in the fact that while the lender will make interest on these loans, they will not make any money from any of these 30-year loans until after 15 years from any of the borrowers. At the end of the 30-year time mortgage, the lender stands to make $1,000,631 from each of these loans. In essence, instead of gaining, $1,000,000 per borrower, they are losing $810,000. That being said, from one month's loan payment from the remaining 17,104 borrowers the lender will collect $860,016,016 making up for the loss, but they will never make up for the $101,000,000 interest they should be collecting from the remaining borrowers.

Why is this important to consider? It is important for the lender to decide what is more important for their business model. Is it more important to filter out as many high-risk borrowers as possible, understanding that they may reject lending to potential clients? Or is it more important to get as many clients as possible understanding that a number of them will be high-risk and likely default on their loans? This relationship is complicated when the borrowed amount is different, hence the standardization of the $810,000 loan laid out in this analysis. 

## Test Results

The analyses performed were the Naive Random Oversampling, SMOTE Oversampling, Cluster Centroid Undersampling, SMOTEENN Combination Sampling, Balanced Random Forest, and Easy Ensemble AdaBoosting. In comparing these, the following parameters were investigated: balanced accuracy, precision, recall, and the harmonic mean of precision and recall (F1). The Balanced Accuracy is an accuracy measurement based on the true positive rate and the true negative rate, recall(sensitivity) and specificity, respectively. The precision indicates the positive predictive value, and the F1-measure indicates the balanced between the positive predictive value and the recall. The following table shows the results from each of the tests: 


```python
results = pd.DataFrame([[0.61, 0.01, 0.70, 0.02], 
    [0.66, 0.01, 0.63, 0.02], 
    [0.53, 0.01, 0.64, 0.01], 
    [0.65, 0.01, 0.73, 0.02], 
    [0.74, 0.03, 0.61, 0.05], 
    [0.93, 0.09, 0.92, 0.16]], 
    index=["Random Oversampling", "SMOTE", "Cluster Centroids", "SMOTEENN", "Random Forests", "Easy Ensemble"], 
    columns=["Accuracy", "Precision", "Recall", "F1"])
results = results.reindex(columns=["Precision", "Recall", "F1", "Accuracy"])
print(results)

```

                         Precision  Recall    F1  Accuracy
    Random Oversampling       0.01    0.70  0.02      0.61
    SMOTE                     0.01    0.63  0.02      0.66
    Cluster Centroids         0.01    0.64  0.01      0.53
    SMOTEENN                  0.01    0.73  0.02      0.65
    Random Forests            0.03    0.61  0.05      0.74
    Easy Ensemble             0.09    0.92  0.16      0.93
    

### Rejecting Good Applicants

The first four analyses utilized different methods of sampling to accommodate for the differences in the True Negative and the True Positives in the training group. These methods have worked in some cases, but multiple sources have suggested adjusting for sampling does not work well when there is a very unbalanced dataset. In the training set that was used, only 0.48% of the applicants were high-risk, and the slightly higher percentage of 0.59% were high risk in the test set. This indicates a very unbalanced dataset, which is further supported with the findings. 

Using the naive random oversampling, SMOTE oversampling, cluster centroids undersampling, and SMOTEENN combination sampling, the precision was measured to be 1%. This indicates that only 1% of applicants who were labeled as high-risk were actually high risk, due to the high number of false positives in each of the analyses. With the random oversampling 6,741 people were incorrectly labeled as high risk, which was slightly lower with the SMOTE analysis at 5,431 people: 39% and 32% of the total group of low-risk applicants were mislabeled as high-risk. Using cluster centroids in undersampling, there were actually more false positives than true negatives. 10,039 low-risk applicants were identified as high risk, and only 7,065 were correctly identified as low-risk: 59% were incorrectly mislabled as high-risk, making this the least effective of the 6 analyses. The SMOTEENN combination sampling fell between the oversampling and undersampling, yielding 7,488 false positives. 

The last two models utilized ensemble learning. The balanced random forest analysis produced substantially fewer false positives: 2249 low risk users (13%) were predicted to be high-risk. And the Easy Ensemble AdaBoost method produced the fewest, 968 (5.7%) were wrongly identified as high-risk. Due to the very unbalanced data, even the reduced number of false positives only amounted to a precision of 3% with balanced random forests and 9% with boosting.

### Accepting Bad Applicants

The number of false negatives in all but the Easy Ensemble Boosting algorithm identified 27% to 39% of high-risk applicants as low-risk. The Easy Ensemble was the only one to reduce this number to 8 of the 104, so about 8%. The recall, or sensitivity, of the different models gives a different aspect of how well each model worked. The recall indicates the true positive rate. Converse to the identification of the mislabled high-risk applicants, the Random Oversampling shows that 70% of the low-risk applicants were identified as low-risk applicants. While Random Forests was able to decrease the number of false positives, it yielded the highest number of false negatives, about 39% of the high-risk applicants were labeled low-risk. The Easy Ensemble had a recall of 92%, demonstrating that 92% of the low-risk applicants were identified as low risk. Thus, this model has the best results combining the highest precision and highest recall of any of the models. 

The F1-measure for all of the models except the Easy Ensemble were either 0.01 or 0.02 - supporting the notion that the unbalanced data may be too unbalanced for these tests to appropriately classify the high and low-risk applicants. Even the 0.16 F1-measure is not considerably high, it is just much higher than the other models. 

Finally, in considering the balanced accuracy, which accounts for the true positive rate and the true negative rate (sensitivity and specificity). The lowest accuracy came from the undersampling model, with 53% accuracy. Both of the ensemble methods outperformed the sampling analyses, with random forests yielding 74% and Easy Ensemble yielding 93% accuracy. 

--- 

## The Bottom Line

Considering the high values of the properties in California, the loans sought for the median household are about $810,000. If the lending company wants to minimize risk, the best of these models that they can use would be the Easy Ensemble AdaBoost model. This model had 93% accuracy and only mis-identified 8% of high-risk users for approval. The net loss would be minimized with this model. However, if the goal of the company is to maximize profits, then none of these models are exceptional, because even the Easy Ensemble misidentified 5.6% of good applicants as high-risk. Because the low-risk applicants far outnumber the high-risk applicants, this model would reduce their risk from 30% down to 8% of a population that represents only 0.5% of all applicants. Meanwhile, they are rejecting 5.6% of applicants who would otherwise be making them interest payments. From the numbers in the test set, comparing the worst-case scenario to the best model: 

*Worst-Case* 
Gain: 17,104 * 1,000,631 = 17,114,792,624
Lose: 101 * 810,000 =         -81,810,000
Profit:                    17,032,982,624

*Easy Ensemble*
Gain: 16,136 * 1,000,631 = 16,146,181,816
Lose: 8 * 810,000 =            -6,480,000
Profit:                    16,139,701,816

Thus, while the model can help reduce the loss, the lender is more likely to reject too many good applicants and will likely lose profits using any of these methods to identify high-risk applicants, due to the inability to reduce the misidentification of low-risk users. 

## Appendix - Data Cleaning


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
# Dependencies and Data Cleaning
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

```

In cleaning the data, several of the columns were removed that were not relevant to this analysis, such as id information and job title. Additionally, other aspects needed to be cleaned, such as the % symbol present in the interest rate field, and changing the Late/Default/Grace Period borrowers into a single high-risk category. There were still 10 non-numerical data types, which would make it difficult to do categorization. All of the months were changed to their numerical form using a dictionary. The other categories were changed using pd.get_dummies(). 


```python
# Data Cleaning
columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]
target = ["loan_status"]

# Load the data
file_path = Path('LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100

# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)',
                  'Default', 'In Grace Period'], 'high_risk')
df = df.replace(x)

df.reset_index(inplace=True, drop=True)
df.head()

# Determine the non-numerical Data types, which will need to be changed
df.select_dtypes(exclude='number').head(4)
# np.unique(df.loan_status)

# Encode the data from string objects to meaningful numerical forms
df_encoded = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'pymnt_plan',
                            'initial_list_status', 'application_type', 'hardship_flag', 'debt_settlement_flag'])

# loan_status_num = {
#     'low_risk': 0,
#     'high_risk': 1
# }

# df_encoded['loan_status'] = df_encoded['loan_status'].apply(lambda x: loan_status_num[x])

# Months dictionary
months_num = {
    "Jan-2019": 1,
    "Feb-2019": 2,
    "Mar-2019": 3,
    "Apr-2019": 4,
    "May-2019": 5
}

# Change the months of issue and next payment to a number corresponding to the month, rather than the string format
df_encoded["issue_d"] = df_encoded["issue_d"].apply(lambda x: months_num[x])
df_encoded["next_pymnt_d"] = df_encoded["next_pymnt_d"].apply(
    lambda x: months_num[x])

# Just verifying that all data has been encoded
# df_encoded.select_dtypes(exclude='number').head()

```


```python
# Split Data into Training and Testing
# Create our features
X = df_encoded.copy()
X = X.drop(target, axis=1)

# Create our target
y = df_encoded[target]

X.describe()

# Check the balance of our target values
y['loan_status'].value_counts()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
y_train['loan_status'].value_counts()

```




    low_risk     51366
    high_risk      246
    Name: loan_status, dtype: int64




```python
y_test['loan_status'].value_counts()
```




    low_risk     17104
    high_risk      101
    Name: loan_status, dtype: int64




```python
percent_high_train = 246/len(y_train)*100
percent_high_test = 101/len(y_test)*100
print(f"The percentage of high risk user in the training set is {percent_high_train:,.2f}%, and in the test set is {percent_high_test:,.2f}%.")
```

    The percentage of high risk user in the training set is 0.48%, and in the test set is 0.59%.
    
