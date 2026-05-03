# Machine Learning – End-Term Examination Study Notes

---

## Table of Contents

1. [Introduction to Machine Learning](#1-introduction-to-machine-learning)
2. [Types of Machine Learning](#2-types-of-machine-learning)
3. [Linear Regression](#3-linear-regression)
4. [Logistic Regression](#4-logistic-regression)
5. [Support Vector Machines (SVM)](#5-support-vector-machines-svm)
6. [K-Nearest Neighbors (KNN)](#6-k-nearest-neighbors-knn)
7. [Decision Trees](#7-decision-trees)
8. [Ensemble Methods](#8-ensemble-methods)
9. [Unsupervised Learning](#9-unsupervised-learning)
10. [Dimensionality Reduction](#10-dimensionality-reduction)
11. [Neural Networks and Deep Learning](#11-neural-networks-and-deep-learning)
12. [Model Evaluation and Validation](#12-model-evaluation-and-validation)
13. [Bias–Variance Tradeoff](#13-biasvariance-tradeoff)
14. [Regularization](#14-regularization)
15. [Optimization Algorithms](#15-optimization-algorithms)
16. [Feature Engineering and Selection](#16-feature-engineering-and-selection)
17. [Bayesian Learning](#17-bayesian-learning)
18. [Reinforcement Learning (Basics)](#18-reinforcement-learning-basics)
19. [Natural Language Processing (NLP) Basics](#19-natural-language-processing-nlp-basics)
20. [Quick-Reference Formulas](#20-quick-reference-formulas)

---

## 1. Introduction to Machine Learning

### What is Machine Learning?
Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables systems to **learn from data** and improve their performance on tasks **without being explicitly programmed** for each task.

> **Arthur Samuel (1959):** "The field of study that gives computers the ability to learn without being explicitly programmed."

> **Tom Mitchell (1997):** "A computer program is said to **learn** from experience E with respect to task T and performance measure P, if its performance at T, as measured by P, improves with experience E."

### Key Concepts
| Concept | Description |
|---------|-------------|
| **Dataset** | Collection of examples (instances/samples) used for training or evaluation |
| **Feature (X)** | An input variable / attribute used to make predictions |
| **Label / Target (y)** | The output variable the model tries to predict |
| **Model** | A mathematical function mapping inputs to outputs |
| **Training** | Process of fitting a model to data by adjusting parameters |
| **Inference** | Using a trained model to make predictions on new data |
| **Generalization** | Model's ability to perform well on unseen data |

### ML vs Traditional Programming
| Traditional Programming | Machine Learning |
|------------------------|-----------------|
| Rules + Data → Output | Data + Output → Rules (learned) |
| Humans write logic explicitly | Computer discovers patterns automatically |

---

## 2. Types of Machine Learning

### 2.1 Supervised Learning
- **Definition:** The model is trained on **labeled data** (input–output pairs).
- **Goal:** Learn a mapping `f: X → y`.
- **Tasks:**
  - **Regression** – Predict a continuous value (e.g., house price).
  - **Classification** – Predict a discrete class label (e.g., spam/not spam).
- **Algorithms:** Linear Regression, Logistic Regression, SVM, Decision Trees, Neural Networks.

### 2.2 Unsupervised Learning
- **Definition:** The model is trained on **unlabeled data**.
- **Goal:** Discover hidden patterns or structure.
- **Tasks:**
  - **Clustering** – Group similar data points (e.g., customer segmentation).
  - **Dimensionality Reduction** – Compress data while preserving structure.
  - **Density Estimation / Generative modeling**.
- **Algorithms:** K-Means, DBSCAN, PCA, Autoencoders.

### 2.3 Semi-Supervised Learning
- **Definition:** Uses a **small amount of labeled data** + a large amount of unlabeled data.
- **Use Case:** Labeling data is expensive (e.g., medical images).

### 2.4 Reinforcement Learning (RL)
- **Definition:** An **agent** learns by interacting with an **environment** via trial-and-error to maximize **cumulative reward**.
- **Key Terms:** Agent, Environment, State, Action, Reward, Policy.
- **Algorithms:** Q-Learning, SARSA, Deep Q-Networks (DQN), Policy Gradient.

### 2.5 Self-Supervised Learning
- Labels are generated automatically from the input data itself (e.g., predicting the next word in a sentence).

---

## 3. Linear Regression

### 3.1 Simple Linear Regression
Models the relationship between one input feature `x` and a continuous target `y`:

```
ŷ = w₀ + w₁x
```

- `w₀` = intercept (bias)
- `w₁` = slope (weight)

### 3.2 Multiple Linear Regression
Extends to `n` features:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ  =  wᵀx  (in vector form)
```

### 3.3 Cost Function – Mean Squared Error (MSE)
```
MSE = (1/m) Σᵢ (yᵢ − ŷᵢ)²
```
where `m` is the number of training examples.

### 3.4 Training Methods
| Method | Description |
|--------|-------------|
| **Ordinary Least Squares (OLS)** | Closed-form solution: `w = (XᵀX)⁻¹ Xᵀy` |
| **Gradient Descent** | Iterative update: `w ← w − α ∇J(w)` |

### 3.5 Assumptions of Linear Regression (LIMHN)
1. **L**inearity – Relationship between X and y is linear.
2. **I**ndependence – Observations are independent.
3. **M**ulticollinearity absence – Features are not highly correlated with each other.
4. **H**omoscedasticity – Constant variance of residuals.
5. **N**ormality of residuals – Residuals are normally distributed.

### 3.6 Evaluation Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | `(1/m)Σ(y−ŷ)²` | Mean squared error (sensitive to outliers) |
| **RMSE** | `√MSE` | Same unit as y |
| **MAE** | `(1/m)Σ|y−ŷ|` | Mean absolute error (robust to outliers) |
| **R²** | `1 − SSres/SStot` | Proportion of variance explained (0–1, higher is better) |
| **Adjusted R²** | Adjusts R² for number of predictors | Penalizes unnecessary features |

---

## 4. Logistic Regression

### 4.1 Overview
- Used for **binary classification** (and extended to multiclass).
- Outputs a **probability** between 0 and 1 using the **sigmoid function**.

### 4.2 Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```
- Output ∈ (0, 1)
- Decision boundary: predict class 1 if `σ(z) ≥ 0.5`, else class 0.

### 4.3 Model
```
z = w₀ + w₁x₁ + ... + wₙxₙ
P(y=1 | x) = σ(z)
```

### 4.4 Cost Function – Binary Cross-Entropy (Log Loss)
```
J(w) = −(1/m) Σ [ yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ) ]
```

### 4.5 Multiclass Logistic Regression
- **One-vs-Rest (OvR):** Train one binary classifier per class.
- **Softmax (Multinomial):** Generalize sigmoid to multiple classes.

```
Softmax: P(y=k | x) = e^(zₖ) / Σⱼ e^(zⱼ)
```

---

## 5. Support Vector Machines (SVM)

### 5.1 Concept
SVM finds the **optimal hyperplane** that maximizes the **margin** between two classes.

- **Support Vectors:** Data points closest to the decision boundary.
- **Margin:** Distance between the hyperplane and the nearest support vectors. SVM maximizes this margin.

### 5.2 Hard Margin vs. Soft Margin
| Type | Description |
|------|-------------|
| **Hard Margin** | Assumes data is linearly separable; no misclassification allowed |
| **Soft Margin** | Allows some misclassification using slack variables `ξᵢ`; controlled by parameter `C` |

- **Large C** → Low bias, high variance (tries to classify all points correctly, narrower margin).
- **Small C** → High bias, low variance (wider margin, allows more misclassification).

### 5.3 Kernel Trick
For non-linearly separable data, map features to a higher-dimensional space using a **kernel function**:

| Kernel | Formula |
|--------|---------|
| Linear | `K(x, z) = xᵀz` |
| Polynomial | `K(x, z) = (xᵀz + c)^d` |
| RBF / Gaussian | `K(x, z) = exp(−γ‖x−z‖²)` |
| Sigmoid | `K(x, z) = tanh(αxᵀz + c)` |

### 5.4 SVM for Regression (SVR)
- Fits data within an **ε-tube** (epsilon-insensitive zone).
- Points inside the tube have zero loss.

---

## 6. K-Nearest Neighbors (KNN)

### 6.1 Algorithm
1. Store all training data.
2. For a new point `x`, compute distance to all training points.
3. Select the `K` nearest neighbors.
4. **Classification:** Majority vote among K neighbors.
5. **Regression:** Average of K neighbors' values.

### 6.2 Distance Metrics
| Metric | Formula |
|--------|---------|
| Euclidean | `√Σ(xᵢ−yᵢ)²` |
| Manhattan | `Σ|xᵢ−yᵢ|` |
| Minkowski | `(Σ|xᵢ−yᵢ|^p)^(1/p)` |
| Cosine | `1 − (x·y)/(‖x‖‖y‖)` |

### 6.3 Choosing K
- **Small K** → Complex boundary, low bias, high variance (overfitting).
- **Large K** → Smooth boundary, high bias, low variance (underfitting).
- Use cross-validation to select optimal K.

### 6.4 Properties
- **Lazy learner** – No explicit training; all computation at prediction time.
- **Instance-based** – Generalizes locally.
- Sensitive to **feature scaling** (normalize/standardize before using KNN).
- Computationally expensive for large datasets (O(n) per prediction).

---

## 7. Decision Trees

### 7.1 Structure
A decision tree is a flowchart-like tree where:
- **Internal nodes** represent feature tests.
- **Branches** represent outcomes of tests.
- **Leaf nodes** represent class labels (classification) or values (regression).

### 7.2 Splitting Criteria

#### For Classification:
| Criterion | Formula |
|-----------|---------|
| **Gini Impurity** | `G = 1 − Σ pᵢ²` |
| **Entropy (Information Gain)** | `H = −Σ pᵢ log₂(pᵢ)` |
| **Information Gain** | `IG = H(parent) − Σ (weight × H(child))` |

- `pᵢ` = proportion of class `i` in a node.
- Gini = 0 and Entropy = 0 → Pure node (all same class).

#### For Regression:
- Minimize **MSE** (Mean Squared Error) or **MAE** at each split.

### 7.3 Stopping Criteria
- Maximum tree depth reached.
- Minimum samples per leaf.
- No improvement in impurity.
- All samples belong to same class.

### 7.4 Pruning
Reduce overfitting by:
- **Pre-pruning (Early Stopping):** Stop growing tree early.
- **Post-pruning (Cost-Complexity Pruning):** Grow full tree, then prune branches with low gain.

### 7.5 Advantages and Disadvantages
| Advantages | Disadvantages |
|-----------|---------------|
| Interpretable and easy to visualize | Prone to overfitting |
| No feature scaling required | Unstable (small data changes → very different trees) |
| Handles both numeric and categorical features | Biased toward features with many levels |
| Non-parametric | Not globally optimal (greedy splits) |

---

## 8. Ensemble Methods

Combine multiple models ("weak learners") to create a stronger model.

### 8.1 Bagging (Bootstrap Aggregating)
- Train multiple models on **random subsets** (with replacement) of training data.
- **Combine:** Average (regression) or majority vote (classification).
- **Goal:** Reduce variance.
- **Algorithm: Random Forest**
  - Builds many decision trees on bootstrap samples.
  - At each split, considers only a **random subset of features** (√p or log₂p).
  - Final prediction = majority vote / average across all trees.
  - Feature importance can be computed via mean decrease in impurity.

### 8.2 Boosting
- Train models **sequentially**: each model corrects errors of previous ones.
- **Goal:** Reduce bias.
- **Algorithms:**

#### AdaBoost (Adaptive Boosting)
1. Initialize equal sample weights.
2. Train a weak learner.
3. Increase weights of misclassified samples.
4. Repeat for `T` rounds.
5. Final prediction = weighted vote of all weak learners.

#### Gradient Boosting
1. Start with a simple model (e.g., predict mean).
2. Compute **residuals** (errors).
3. Fit a new model to residuals.
4. Update prediction by adding new model (scaled by learning rate `η`).
5. Repeat.

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

#### XGBoost (eXtreme Gradient Boosting)
- Adds **L1 + L2 regularization**, column subsampling, and hardware optimization.
- Uses second-order Taylor expansion of loss for better optimization.
- Faster and often more accurate than standard gradient boosting.

### 8.3 Stacking
- Train a **meta-learner** on the outputs (predictions) of multiple base models.
- Base models make predictions → meta-learner learns how to combine them.

### 8.4 Bagging vs. Boosting
| | Bagging | Boosting |
|-|---------|---------|
| Models trained | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Weights | Equal | Adjusted by errors |
| Prone to | – | Overfitting if T too large |

---

## 9. Unsupervised Learning

### 9.1 K-Means Clustering

**Algorithm:**
1. Choose K cluster centroids randomly.
2. Assign each point to the nearest centroid.
3. Recompute centroids as the mean of assigned points.
4. Repeat steps 2–3 until convergence.

**Objective (Inertia):**
```
J = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖²
```

**Choosing K – Elbow Method:**
Plot inertia vs. K; choose K where the curve "elbows" (rate of decrease slows).

**Limitations:**
- Must specify K in advance.
- Assumes spherical, equal-sized clusters.
- Sensitive to outliers and initialization.
- Use **K-Means++** initialization for better results.

### 9.2 DBSCAN (Density-Based Spatial Clustering)
- Groups points in **dense regions** and labels sparse regions as noise.
- **Parameters:** `ε` (neighborhood radius), `MinPts` (minimum points in ε-neighborhood).
- **Point types:** Core, Border, Noise.
- **Advantages:** Finds arbitrary-shaped clusters; detects outliers; no need to specify K.

### 9.3 Hierarchical Clustering
**Agglomerative (bottom-up):**
1. Start: each point is its own cluster.
2. Merge the two closest clusters.
3. Repeat until one cluster remains.

**Linkage Methods:**
| Method | Distance Between Clusters |
|--------|--------------------------|
| Single | Minimum distance between any two points |
| Complete | Maximum distance between any two points |
| Average | Average distance between all pairs |
| Ward | Minimize within-cluster variance |

Visualized using a **dendrogram**.

### 9.4 Gaussian Mixture Models (GMM)
- Assumes data is generated from a mixture of `K` Gaussian distributions.
- Uses **Expectation-Maximization (EM)** algorithm.
- **Soft clustering** – Each point has a probability of belonging to each cluster.

---

## 10. Dimensionality Reduction

### 10.1 Why Reduce Dimensions?
- Combat the **curse of dimensionality**.
- Reduce computation time and memory.
- Improve model performance by removing noise.
- Enable 2D/3D visualization.

### 10.2 Principal Component Analysis (PCA)

**Steps:**
1. Standardize the data (zero mean, unit variance).
2. Compute the **covariance matrix** `Σ`.
3. Compute **eigenvectors** and **eigenvalues** of `Σ`.
4. Sort eigenvectors by eigenvalue (descending) → these are the principal components.
5. Project data onto top `k` eigenvectors.

**Key Points:**
- Principal components are **orthogonal** (uncorrelated).
- Each PC captures maximum remaining variance.
- **Explained variance ratio** = eigenvalue / sum of all eigenvalues.
- Choose `k` such that the cumulative explained variance ≥ 95%.

### 10.3 t-SNE (t-distributed Stochastic Neighbor Embedding)
- **Non-linear** dimensionality reduction for **visualization** (typically 2D/3D).
- Preserves **local structure** (nearby points stay nearby).
- **Not suitable** for general dimensionality reduction or new data projection.
- Hyperparameter: **perplexity** (balances local vs. global structure, typically 5–50).

### 10.4 Autoencoders
- Neural network with **encoder** (compress to bottleneck) and **decoder** (reconstruct).
- Bottleneck layer provides compressed representation.
- Used for: anomaly detection, denoising, generative modeling.

### 10.5 Linear Discriminant Analysis (LDA)
- **Supervised** dimensionality reduction.
- Finds axes that **maximize between-class variance** and **minimize within-class variance**.
- At most `C−1` components (C = number of classes).
- Assumes features are normally distributed with equal class covariances.

---

## 11. Neural Networks and Deep Learning

### 11.1 Biological Inspiration
Modeled after neurons in the brain. Each artificial neuron:
1. Receives weighted inputs.
2. Sums them.
3. Passes through an **activation function**.

### 11.2 Perceptron
Simplest neural network (single neuron):
```
output = f(w₀ + w₁x₁ + ... + wₙxₙ)
```

### 11.3 Multilayer Perceptron (MLP)
- **Input Layer** – Receives features.
- **Hidden Layers** – Learn intermediate representations.
- **Output Layer** – Produces final prediction.

### 11.4 Activation Functions
| Function | Formula | Range | Used In |
|----------|---------|-------|---------|
| **Sigmoid** | `1/(1+e^{-x})` | (0, 1) | Binary output |
| **Tanh** | `(e^x−e^{-x})/(e^x+e^{-x})` | (−1, 1) | Hidden layers |
| **ReLU** | `max(0, x)` | [0, ∞) | Hidden layers (most common) |
| **Leaky ReLU** | `max(αx, x)` where α≪1 | (−∞, ∞) | Avoids dying ReLU |
| **Softmax** | `e^{xᵢ}/Σe^{xⱼ}` | (0,1), sums to 1 | Multiclass output |
| **GELU** | `x·Φ(x)` | (−∞, ∞) | Transformers |

**Dying ReLU Problem:** ReLU neurons can "die" (always output 0) if weights push inputs to negative region. Leaky ReLU / ELU / PReLU address this.

### 11.5 Backpropagation

**Forward Pass:** Compute output and loss.

**Backward Pass:** Compute gradients using **chain rule**:
```
∂L/∂wᵢ = ∂L/∂ŷ · ∂ŷ/∂zₗ · ∂zₗ/∂zₗ₋₁ · ... · ∂z₁/∂wᵢ
```

**Weight Update (Gradient Descent):**
```
w ← w − α · ∂L/∂w
```

### 11.6 Loss Functions for Neural Networks
| Task | Loss Function |
|------|--------------|
| Binary Classification | Binary Cross-Entropy |
| Multiclass Classification | Categorical Cross-Entropy |
| Regression | MSE or MAE |

### 11.7 Convolutional Neural Networks (CNN)
- Designed for **grid-structured data** (images, audio spectrograms).
- **Convolution Layer:** Applies filters to extract local features (edges, textures).
- **Pooling Layer:** Reduces spatial dimensions (Max Pooling, Average Pooling).
- **Fully Connected Layer:** Final classification/regression.
- Key property: **Parameter sharing** and **local connectivity** → fewer parameters than FC layers.

**CNN Architecture Example (LeNet-5):**
```
Input → Conv → Pool → Conv → Pool → FC → FC → Output
```

### 11.8 Recurrent Neural Networks (RNN)
- Designed for **sequential data** (text, time series, audio).
- Has **hidden state** `h_t` that carries information from previous time steps.
- **Problem:** Vanishing/exploding gradients over long sequences.

**LSTM (Long Short-Term Memory):**
- Adds **cell state** and gating mechanisms (forget gate, input gate, output gate) to preserve long-term dependencies.

**GRU (Gated Recurrent Unit):**
- Simpler than LSTM with reset gate and update gate; often comparable performance.

### 11.9 Transformers (Attention Mechanism)
- Replaces recurrence with **self-attention**.
- **Attention Score:** `Attention(Q, K, V) = softmax(QKᵀ/√dₖ) · V`
- Processes all tokens in **parallel** (vs. sequential in RNN).
- Foundation of modern NLP: BERT, GPT, T5.

### 11.10 Regularization in Neural Networks
| Technique | Description |
|-----------|-------------|
| **Dropout** | Randomly zero out neurons during training (rate p) |
| **Batch Normalization** | Normalize activations within a mini-batch |
| **Early Stopping** | Stop training when validation loss stops improving |
| **Weight Decay (L2)** | Add L2 penalty on weights to loss |
| **Data Augmentation** | Artificially expand training set (flips, crops, noise) |

---

## 12. Model Evaluation and Validation

### 12.1 Train / Validation / Test Split
- **Training Set (~60–70%):** Used to fit the model.
- **Validation Set (~10–20%):** Used to tune hyperparameters.
- **Test Set (~10–20%):** Used once for final evaluation.

### 12.2 Cross-Validation

**K-Fold Cross-Validation:**
1. Split data into K equal folds.
2. Train on K−1 folds, validate on 1 fold.
3. Repeat K times (each fold is the validation set once).
4. Average performance across K runs.

**Stratified K-Fold:** Preserves class proportions in each fold (important for imbalanced data).

**Leave-One-Out Cross-Validation (LOOCV):** K = m (number of samples). Very expensive but low-bias estimate.

### 12.3 Classification Metrics

**Confusion Matrix:**
```
                 Predicted Positive   Predicted Negative
Actual Positive       TP                    FN
Actual Negative       FP                    TN
```

| Metric | Formula | Notes |
|--------|---------|-------|
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` | Misleading for imbalanced data |
| **Precision** | `TP/(TP+FP)` | Of all predicted positives, how many are truly positive |
| **Recall (Sensitivity)** | `TP/(TP+FN)` | Of all actual positives, how many did we catch |
| **Specificity** | `TN/(TN+FP)` | True Negative Rate |
| **F1 Score** | `2·(P·R)/(P+R)` | Harmonic mean of Precision and Recall |
| **F-beta** | `(1+β²)·P·R / (β²·P+R)` | β>1: favor recall; β<1: favor precision |
| **MCC** | Complex formula | Balanced metric even for imbalanced classes |

**ROC Curve (Receiver Operating Characteristic):**
- Plots **TPR (Recall) vs. FPR** at various thresholds.
- **AUC (Area Under Curve):** 1.0 = perfect classifier; 0.5 = random.

**Precision-Recall Curve:**
- More informative than ROC for **imbalanced datasets**.

### 12.4 Regression Metrics
| Metric | Formula |
|--------|---------|
| MSE | `(1/m)Σ(y−ŷ)²` |
| RMSE | `√MSE` |
| MAE | `(1/m)Σ|y−ŷ|` |
| MAPE | `(1/m)Σ|y−ŷ|/|y| × 100%` |
| R² | `1 − SSres/SStot` |

### 12.5 Hyperparameter Tuning
| Method | Description |
|--------|-------------|
| **Grid Search** | Exhaustively try all combinations |
| **Random Search** | Randomly sample from parameter space (faster) |
| **Bayesian Optimization** | Model the objective function, sample intelligently |
| **Automated ML (AutoML)** | Automates model selection and hyperparameter tuning |

---

## 13. Bias–Variance Tradeoff

### 13.1 Key Definitions
- **Bias:** Error from **wrong assumptions** – model is too simple (underfitting).
- **Variance:** Error from **sensitivity to training data fluctuations** – model is too complex (overfitting).
- **Irreducible Error (noise):** Cannot be reduced regardless of model.

### 13.2 Total Expected Error
```
Expected Error = Bias² + Variance + Irreducible Noise
```

### 13.3 Tradeoff
| | Bias | Variance | Example |
|-|------|----------|---------|
| Simple model | High | Low | Linear Regression on complex data |
| Complex model | Low | High | Deep neural network on small dataset |
| Ideal model | Low | Low | Properly regularized model |

### 13.4 Diagnosing Issues
| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| High training error AND high test error | Underfitting (High Bias) | More complex model, add features |
| Low training error, HIGH test error | Overfitting (High Variance) | Regularize, more data, simpler model |
| Both errors low | Good fit | – |

---

## 14. Regularization

### 14.1 Why Regularize?
Prevent overfitting by adding a **penalty term** to the loss function that discourages large weights.

### 14.2 L2 Regularization (Ridge)
```
J(w) = Loss(w) + λ Σ wᵢ²
```
- Shrinks all weights toward zero (but not exactly to zero).
- **Closed-form solution:** `w = (XᵀX + λI)⁻¹ Xᵀy`
- Handles multicollinearity well.

### 14.3 L1 Regularization (Lasso)
```
J(w) = Loss(w) + λ Σ |wᵢ|
```
- Drives some weights **exactly to zero** → automatic **feature selection**.
- Produces sparse models.
- No closed-form solution (use subgradient or coordinate descent).

### 14.4 Elastic Net
```
J(w) = Loss(w) + λ₁ Σ|wᵢ| + λ₂ Σwᵢ²
```
- Combines L1 (sparsity) and L2 (grouping effect).
- Useful when features are correlated.

### 14.5 Dropout (Neural Networks)
- During training, randomly set each neuron's output to 0 with probability `p`.
- At inference, scale weights by `(1−p)`.
- Equivalent to training an ensemble of sub-networks.

---

## 15. Optimization Algorithms

### 15.1 Gradient Descent Variants
| Variant | Batch Size | Update Frequency | Notes |
|---------|-----------|-----------------|-------|
| **Batch GD** | All data | Once per epoch | Stable but slow for large datasets |
| **Stochastic GD (SGD)** | 1 sample | After each sample | Noisy, fast updates, can escape local minima |
| **Mini-Batch GD** | B samples | After each mini-batch | Balance of stability and speed (most common) |

### 15.2 Advanced Optimizers

#### Momentum
```
v ← β·v − α·∇J
w ← w + v
```
- Accumulates velocity in direction of gradient.
- Reduces oscillations and speeds convergence.

#### RMSProp
```
s ← β·s + (1−β)·(∇J)²
w ← w − (α / √(s+ε)) · ∇J
```
- Adapts learning rate per parameter.
- Good for non-stationary objectives (RNNs).

#### Adam (Adaptive Moment Estimation)
```
m ← β₁·m + (1−β₁)·∇J        (first moment)
v ← β₂·v + (1−β₂)·(∇J)²     (second moment)
m̂ = m/(1−β₁ᵗ)               (bias correction)
v̂ = v/(1−β₂ᵗ)
w ← w − α·m̂/√(v̂+ε)
```
- Combines Momentum + RMSProp.
- Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001.
- **Most widely used** optimizer in deep learning.

### 15.3 Learning Rate Scheduling
| Schedule | Description |
|----------|-------------|
| Step Decay | Reduce LR by factor every N epochs |
| Exponential Decay | `α = α₀ · e^(−kt)` |
| Cosine Annealing | LR follows cosine curve |
| Warm Restarts | Periodically reset LR (SGDR) |
| OneCycleLR | Ramp up then down (fast training trick) |

---

## 16. Feature Engineering and Selection

### 16.1 Feature Engineering Techniques
| Technique | Description |
|-----------|-------------|
| **Normalization (Min-Max)** | Scale to [0, 1]: `x' = (x − xₘᵢₙ)/(xₘₐₓ − xₘᵢₙ)` |
| **Standardization (Z-score)** | Scale to mean=0, std=1: `x' = (x − μ)/σ` |
| **Log Transform** | Reduce skewness for right-skewed distributions |
| **One-Hot Encoding** | Convert categorical to binary dummy variables |
| **Label Encoding** | Assign integers to categories |
| **Binning** | Group continuous values into discrete bins |
| **Polynomial Features** | Create interaction terms `x₁x₂`, `x₁²` |
| **Date/Time Features** | Extract hour, day, month, weekday, etc. |

### 16.2 Feature Selection Methods

#### Filter Methods (Model-Independent)
- **Correlation:** Remove highly correlated features.
- **Chi-Square Test:** For categorical features vs. categorical targets.
- **ANOVA F-test:** For continuous features vs. categorical targets.
- **Mutual Information:** Measures statistical dependency.

#### Wrapper Methods
- **Forward Selection:** Start empty; add best feature iteratively.
- **Backward Elimination:** Start full; remove worst feature iteratively.
- **Recursive Feature Elimination (RFE):** Fit model, rank features, remove least important.

#### Embedded Methods
- **L1 Regularization (Lasso):** Zero-weight features are eliminated.
- **Tree-based Feature Importance:** Mean decrease in impurity.
- **Gradient Boosting Importance:** Feature importance from SHAP values.

### 16.3 Handling Missing Values
| Strategy | Description |
|----------|-------------|
| **Drop rows** | Remove samples with missing values |
| **Drop columns** | Remove features with too many missing values |
| **Mean/Median/Mode Imputation** | Fill with statistical summary |
| **KNN Imputation** | Use nearby samples to impute |
| **Multiple Imputation** | Statistical approach accounting for uncertainty |
| **Indicator variable** | Add binary column: `feature_was_missing` |

### 16.4 Handling Imbalanced Data
| Method | Description |
|--------|-------------|
| **Oversampling (SMOTE)** | Synthetically generate minority class samples |
| **Undersampling** | Reduce majority class samples |
| **Class Weights** | Increase penalty for minority class in loss |
| **Threshold Moving** | Adjust decision threshold |
| **Ensemble Methods** | BalancedBaggingClassifier, EasyEnsemble |

---

## 17. Bayesian Learning

### 17.1 Bayes' Theorem
```
P(H | D) = P(D | H) · P(H) / P(D)
```
- `P(H)` = **Prior** – belief about hypothesis before seeing data.
- `P(D | H)` = **Likelihood** – probability of data given hypothesis.
- `P(H | D)` = **Posterior** – updated belief after seeing data.
- `P(D)` = **Evidence** (normalizing constant).

### 17.2 Maximum Likelihood Estimation (MLE)
- Find parameters `θ` that **maximize the likelihood** of observed data:
```
θ̂_MLE = argmax_θ P(D | θ)  =  argmax_θ log P(D | θ)
```
- Does **not** use a prior.

### 17.3 Maximum A Posteriori (MAP)
- Find parameters that **maximize the posterior**:
```
θ̂_MAP = argmax_θ P(θ | D) = argmax_θ [log P(D | θ) + log P(θ)]
```
- Incorporates prior; equivalent to MLE + regularization.

### 17.4 Naïve Bayes Classifier

**Assumption:** Features are **conditionally independent** given the class label.

```
P(y | x₁,...,xₙ) ∝ P(y) · Π P(xᵢ | y)
```

**Variants:**
| Variant | Feature Type |
|---------|-------------|
| Gaussian NB | Continuous (assumes Gaussian distribution) |
| Multinomial NB | Count data (word counts in text) |
| Bernoulli NB | Binary features |

**Laplace Smoothing:** Add 1 (or α) to all counts to avoid zero probabilities.

**Advantages:** Very fast, works well with small data, good for text classification.
**Disadvantage:** Strong independence assumption often violated in practice.

### 17.5 Bayesian Networks
- **Directed Acyclic Graph (DAG)** representing conditional dependencies.
- Node = random variable; Edge = conditional dependency.
- Efficient representation of joint distribution.

---

## 18. Reinforcement Learning (Basics)

### 18.1 Framework
```
Agent → (Action aₜ) → Environment → (State sₜ₊₁, Reward rₜ) → Agent
```

| Term | Description |
|------|-------------|
| **State (s)** | Current situation of the agent |
| **Action (a)** | Choice made by agent |
| **Reward (r)** | Scalar feedback from environment |
| **Policy (π)** | Strategy: `π(a|s)` = probability of action a in state s |
| **Value Function V(s)** | Expected cumulative reward from state s |
| **Q-Function Q(s,a)** | Expected cumulative reward from state s taking action a |
| **Episode** | Sequence of states, actions, rewards until terminal state |
| **Discount Factor (γ)** | Importance of future rewards: `0 ≤ γ ≤ 1` |

### 18.2 Return (Cumulative Reward)
```
Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... = Σₖ₌₀^∞ γᵏ rₜ₊ₖ
```

### 18.3 Bellman Equation
```
V(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [r(s,a,s') + γV(s')]
```

### 18.4 Q-Learning (Off-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]
```
- Model-free; learns optimal Q without needing environment model.

### 18.5 Exploration vs. Exploitation
- **Exploration:** Try new actions to gain information.
- **Exploitation:** Use best known action.
- **ε-greedy:** With probability ε, pick random action; else pick best action.

---

## 19. Natural Language Processing (NLP) Basics

### 19.1 Text Preprocessing Pipeline
```
Raw Text → Tokenization → Lowercasing → Stop Word Removal → 
Stemming/Lemmatization → Feature Extraction
```

### 19.2 Text Representation

| Method | Description |
|--------|-------------|
| **Bag of Words (BoW)** | Document = vector of word counts |
| **TF-IDF** | Term Frequency × Inverse Document Frequency |
| **N-grams** | Sequences of N consecutive words |
| **Word Embeddings (Word2Vec, GloVe)** | Dense vectors preserving semantic relationships |
| **Contextual Embeddings (BERT, GPT)** | Vectors vary based on surrounding context |

**TF-IDF Formula:**
```
TF-IDF(t, d) = TF(t, d) × log(N / df(t))
```
- `TF(t,d)` = frequency of term t in document d.
- `N` = total documents; `df(t)` = documents containing term t.

### 19.3 Word2Vec
- **CBOW:** Predict target word from context words.
- **Skip-gram:** Predict context words from target word.
- Captures semantic relationships (king − man + woman ≈ queen).

### 19.4 Common NLP Tasks
| Task | Description | Example |
|------|-------------|---------|
| Text Classification | Assign category to text | Spam detection |
| Named Entity Recognition (NER) | Identify entities in text | "Apple Inc." → ORG |
| Sentiment Analysis | Detect sentiment | Positive/Negative/Neutral |
| Machine Translation | Translate between languages | EN → FR |
| Question Answering | Answer questions from context | Reading comprehension |
| Text Summarization | Shorten text preserving meaning | Abstractive / Extractive |

---

## 20. Quick-Reference Formulas

### Probability & Statistics
```
Mean:               μ = (1/n) Σxᵢ
Variance:           σ² = (1/n) Σ(xᵢ − μ)²
Std Dev:            σ = √σ²
Covariance:         Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)]
Pearson Correlation: ρ = Cov(X,Y) / (σₓ σᵧ)
Bayes Theorem:      P(A|B) = P(B|A)P(A)/P(B)
```

### Linear Algebra
```
Matrix Multiply:    (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
Transpose:          (Aᵀ)ᵢⱼ = Aⱼᵢ
Inverse:            AA⁻¹ = I
Eigenvalue:         Av = λv
```

### Key ML Formulas
```
Linear Regression:  ŷ = wᵀx + b
Logistic (sigmoid): σ(z) = 1/(1+e⁻ᶻ)
Softmax:            P(k) = e^zₖ / Σⱼ e^zⱼ
MSE Loss:           L = (1/m)Σ(yᵢ−ŷᵢ)²
Cross-Entropy:      L = −(1/m)Σ[yᵢlog(ŷᵢ)+(1−yᵢ)log(1−ŷᵢ)]
Gradient Descent:   w ← w − α∇L
Ridge:              L_ridge = L + λ‖w‖²₂
Lasso:              L_lasso = L + λ‖w‖₁
Precision:          TP/(TP+FP)
Recall:             TP/(TP+FN)
F1:                 2PR/(P+R)
```

---

## Exam Tips & Common Mistakes

### Things Often Tested
1. **Distinguish** supervised / unsupervised / RL with examples.
2. **Derive** or interpret gradient descent update rule.
3. **Explain** bias-variance tradeoff with diagrams.
4. **Calculate** confusion matrix metrics (Precision, Recall, F1).
5. **Compare** Gini impurity vs. Information Gain.
6. **Explain** the kernel trick in SVM.
7. **Describe** backpropagation step-by-step.
8. **Select** appropriate metric for imbalanced dataset.
9. **Explain** when to use L1 vs. L2 regularization.
10. **Compare** Bagging vs. Boosting.

### Common Mistakes to Avoid
- Confusing **training error** and **test error** when diagnosing bias/variance.
- Applying **feature scaling** to test data using test set statistics (always use training statistics).
- Using **accuracy** as the only metric for imbalanced data.
- Forgetting **data leakage** when preprocessing before splitting.
- Misunderstanding **gradient descent** step direction (move against gradient).
- Confusing **precision** and **recall** formulas.
- Forgetting that **PCA requires standardization** before application.
- Confusing **parameters** (learned from data) with **hyperparameters** (set before training).

---

*Good luck on your end-term examination!*
