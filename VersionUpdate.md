
# Deletion Pathogenicity Predictor – Version Update Jenny

## Overview

## **1. Handling Ambiguous Variants (`map_prob`)**

### **Original Behavior**

* Variants with ambiguous or conflicting clinical significance were dropped from the dataset.

### **Updated Behavior**

* Ambiguous variants are now assigned a probability of **0.5** instead of being removed.

### **Reason**

* Allows the model to retains more data → larger and more diverse training set => add some noise
* Reduces bias toward clearly pathogenic or benign variants.
* The additional 0.5 rather than 0 introduction allowed the model to produce **nuanced predictions** for uncertain cases.
* this helps the model generalize better to real-world variants with uncertain significance.

```python
def map_prob(text):
    t = (text or '').lower()
    if 'pathogenic' in t and 'likely' in t:
        return 0.9
    elif 'pathogenic' in t:
        return 1.0
    elif 'benign' in t and 'likely' in t:
        return 0.1
    elif 'benign' in t:
        return 0.0
    return 0.5  # Ambiguous variants I changed this from 0 to 0.5
```

---

## **2. One-Hot Encoding for Categorical Features**

### **Original Behavior**

* Used `LabelEncoder()` for categorical features (`gene`, `consequence`, `condition`).

### **Problem with Original Approach**

I am not 100% certain but it looked like that the model was making its answers by drawing too much confidence in the consequence (disease) of the given data

I believe label encoding assignes a unique int to every feature, but since we know stuff like gene consequence and condition is on eof the more imprtant features 

that the model should consider, I used one hot so that it avoids assigning arbitrary numeric order introduced by label encoding may mislead the model.

* 'Consequesce encoded' was a dominant feature observed and evaluated in the previous model
it weight of importance was really high (~92% for `consequence_encoded` previously) so I tried to tone it down a bit by cahnge the 0 ->0.5 in your map_prob()
and change the max depth, leaf nodes possible in the tree.

* In case you need to put this into report Brandon or a script for later presentation, I have include a more complete report below

### **Updated Behavior**

* Replaced label encoding with **one-hot encoding** for all categorical features & changes made to our map_prob().

### **Reason / Benefits**

* Reduces dominance of any single categorical feature, namely the featuer that referred to the specific diseas/consequence of the gene deletion 
* The changes made to map_prob() left more cases to the model to train on instead of dropping.  With more cases, I had hoped that 
this would help to preven t overfitting.
* Improves interpretability and allows numeric features (deletion size, chromosome position) to contribute more (hold more weight) to the
models predictions.


## **3. Model Regularization: Random Forest Hyperparameters**

### **Changes Made**

* `max_depth` reduced to limit tree depth.
* `min_samples_split` and `min_samples_leaf` increased to reduce overfitting to small subsets.
* These changes prevent overly complex trees and ensure the model learns **generalizable patterns**.

### **Reason / Benefits**

* I used this to reduced slight overfitting observed in train vs test metrics.
---

## **4. Model Evaluation Results (Post-Update)**

### **Hold-out Test Set Evaluation**

```
Test MSE:         0.0232       // Prev discord results 0.0181
Test Precision:   0.9637
Test Recall:      0.9865
Test Specificity: 0.9009

Train MSE:        0.0166        // Prev discord results: 0.0094 => 0.0181-0.0094 = 0.0087. The MSE of the model towards the test vs training data
                                // New Train MSE - NEW test MSE = 0.0232 - 0.0166 = 0.0066.  The gap closed (0.0087->0.0066) albeight not much suggests that
                                // preformance over train and test have become more consistent
Train Precision:  0.9716
Train Recall:     0.9877
Train Specificity:0.9320
```

### **Cross-Validation (10-fold)**

```
MSE: 0.0188
Precision: 0.9637
Recall: 0.9865
Specificity: 0.9009 // your numbers were better before, I'm not sure why this happened.  
```


## **5. Key Takeaways**

1. **0.5 Assignment for Ambiguous Variants**

   * Retains data and allows probabilistic learning for uncertain cases => more training cases kept yayy

2. **One-Hot Encoding**

   * Prevents arbitrary numeric ordering of categories, reduces feature dominance, and improves interpretability.

3. **Random Forest Regularization**

   * Adjusted `max_depth`, `min_samples_split`, and `min_samples_leaf` to reduce overfitting and increase generalizability.

4. **Improved Feature Representation**

   * Numeric features like deletion size and chromosomal position now contribute meaningfully, aligning predictions with biological knowledge.

6. Noticible differences: 

* Train vs test gap is modest => minimal overfitting.
* CV metrics match hold-out test metrics => model generalizes well although note that secificity is not quite as good as the previous model
* weights of feature importance shifted, increased for deletion location and chromosome ; decresed for consequences/disease feature
* Deletion size and chromosome position, previously underrepresented, now meaningfully influence predictions.

