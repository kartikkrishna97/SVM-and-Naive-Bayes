# Naive Bayes Classifier and SVM Implementations

This repository contains implementations of both Naive Bayes Classifier and Support Vector Machines (SVMs) for various classification tasks. The models are applied to Corona tweets data, and additional tasks like domain adaptation and feature extraction are explored.

## Naive Bayes Classifier

The Naive Bayes Classifier is implemented from scratch and includes several features to enhance its performance. Below is a breakdown of the functionality:

1. **Basic Naive Bayes Implementation:**
   - Trains a Naive Bayes classifier from scratch on the Corona tweets dataset.

2. **Baseline Classifiers:**
   - Implements two baseline classifiers:
     - A random classifier.
     - An all-1s classifier.
   - Computes and compares the accuracy of these baseline classifiers against Naive Bayes.

3. **Text Preprocessing:**
   - Applies stemming and stop word removal to the data.
   - Re-trains the Naive Bayes classifier on the preprocessed data and computes new accuracy and performance metrics.

4. **N-Gram Features:**
   - Incorporates both bigram and trigram features into the Naive Bayes model to improve performance.

5. **Domain Adaptation:**
   - Trains the Naive Bayes classifier on one domain and evaluates its performance with few-shot learning on a different domain.

### Word Cloud Visualization

- **Basic Word Cloud:**
  - Generates a word cloud based on the provided dataset.

- **Word Cloud with Stop Word Removal:**
  - Generates a word cloud after removing stop words from the data.

## SVM Implementations

The Support Vector Machines (SVMs) are implemented using the CVXOPT package, with multiple configurations for different tasks and kernels.

1. **Linear SVM (2 Classes):**
   - Implements a linear SVM for binary classification using CVXOPT.

2. **Gaussian Kernel SVM (2 Classes):**
   - Implements a Gaussian Kernel SVM for binary classification using CVXOPT.

3. **Multi-Class SVM (5 Classes):**
   - Extends the SVM implementation for multi-class classification (5 classes) using CVXOPT.

4. **K-Fold Cross Validation:**
   - Implements multi-class SVM with K-fold cross validation to evaluate the model’s robustness.

### SVM Comparisons

- **Linear SVM Comparison:**
  - Compares the performance of the custom linear SVM with the linear SVM from `sklearn`.

- **Gaussian Kernel SVM Comparison:**
  - Compares the performance of the custom Gaussian Kernel SVM with `sklearn`'s SVM using the RBF kernel.

## Usage Instructions

Each implementation can be run individually. Navigate to the corresponding directory and run the desired Python script:

```bash
python a.py  # For the Naive Bayes implementation
python b.py  # For SVM with Gaussian kernel
python c.py  # For multi-class SVM
# And so on...




