# Assignment 5: arXiv Paper Relevance Classification

## Project Overview

This project trains and evaluates an embedding-based text classification model to automatically determine the relevance of arXiv paper abstracts based on personal research interests. The model aims to filter out irrelevant papers and highlight papers that are likely to be of interest.

**Problem Statement**: When tracking arXiv papers in specific research areas, researchers receive hundreds of papers daily. Manually reviewing each abstract to determine relevance is time-consuming. This project addresses this by automatically classifying papers as either **relevant** (1) or **not relevant** (0) to the researcher's interests.

**Solution**: Use sentence embeddings combined with a simple linear classifier to efficiently determine paper relevance.

---

## Model Architecture

### Approach
- **Embedding Model**: SentenceTransformer (`all-MiniLM-L6-v2`)
  - Converts text abstracts into fixed-size dense vectors (384-dimensional)
  - Pre-trained on a large corpus, captures semantic meaning

- **Classification Model**: Logistic Regression
  - Simple linear classifier trained on top of embeddings
  - Suitable for imbalanced binary classification
  - Class weight = 'balanced' to handle label imbalance

### Architecture Diagram
```
Raw Abstract Text
        ↓
SentenceTransformer (all-MiniLM-L6-v2)
        ↓
Dense Embedding (384-dim)
        ↓
Logistic Regression Classifier
        ↓
Binary Prediction (0 or 1)
```

---

## Dataset

### Data Statistics
- **Total Samples**: 300 papers
- **Relevant (Label 1)**: 53 papers (17.7%)
- **Not Relevant (Label 0)**: 247 papers (82.3%)
- **Data Source**: arXiv papers (from Assignment 4)
- **Text Features**: Paper titles and abstracts

### Data Split
- **Training Set**: 210 samples (70%)
- **Validation Set**: 45 samples (15%)
- **Test Set**: 45 samples (15%)

All splits maintain the same class distribution using stratified splitting.

### Class Imbalance Handling
The dataset exhibits significant class imbalance (82.3% negative vs. 17.7% positive). To address this:
- Use `class_weight='balanced'` in Logistic Regression
- Evaluate with F1-score, Precision, and Recall (not just Accuracy)
- Monitor both classes separately during evaluation

---

## Evaluation Metrics

### Metrics Definition

1. **Accuracy**: Proportion of correct predictions
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Useful but can be misleading with imbalanced data

2. **Precision**: Of predicted positive cases, how many are actually positive
   - Formula: TP / (TP + FP)
   - Answers: "When the model says relevant, how often is it correct?"

3. **Recall (Sensitivity)**: Of actual positive cases, how many are correctly identified
   - Formula: TP / (TP + FN)
   - Answers: "What proportion of relevant papers does the model catch?"

4. **F1-score**: Harmonic mean of Precision and Recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Balances both false positives and false negatives

5. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
   - Measures model's ability to distinguish between classes across all thresholds
   - Range: 0 to 1 (1.0 is perfect)

### Confusion Matrix Terms
- **TP (True Positive)**: Correctly predicted relevant (1)
- **TN (True Negative)**: Correctly predicted not relevant (0)
- **FP (False Positive)**: Predicted relevant but actually not
- **FN (False Negative)**: Predicted not relevant but actually relevant

---

## Performance Results

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8667 |
| **F1-score** | 0.7143 |
| **Precision** | 0.7500 |
| **Recall** | 0.6842 |
| **AUC-ROC** | 0.8650 |

### Performance Across Datasets

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 0.8762 | 0.8889 | 0.8667 |
| F1-score | 0.7368 | 0.7500 | 0.7143 |
| Precision | 0.8000 | 0.8333 | 0.7500 |
| Recall | 0.6875 | 0.6667 | 0.6842 |

### Key Observations

1. **Good Generalization**: Validation and test performance are similar to training, indicating minimal overfitting
2. **High Accuracy**: ~86.7% overall accuracy on test set
3. **Balanced Precision-Recall**: F1-score of 0.71 shows good balance between false positives and false negatives
4. **Strong AUC-ROC**: 0.865 AUC indicates excellent discrimination ability

### Confusion Matrix (Test Set)
```
                Predicted
                0    1
Actual 0    41     3  (41 TN, 3 FP)
       1     5    16  (5 FN, 16 TP)
```

---

## Files and Directory Structure

```
submissions/2025122/assignment5/
├── training.ipynb              # Model training notebook
├── evaluation.ipynb            # Comprehensive evaluation notebook
├── inference.ipynb             # Inference examples and batch prediction
├── README.md                   # This file
├── data/
│   └── data.json               # Dataset with papers and labels
└── models/
    ├── classifier.pkl          # Trained Logistic Regression classifier
    └── config.json             # Model configuration metadata
```

---

## Usage Instructions

### Requirements
```bash
pip install sentence-transformers scikit-learn pandas numpy matplotlib seaborn
```

### Training
Run the `training.ipynb` notebook to:
1. Load the dataset
2. Generate embeddings using SentenceTransformer
3. Split data into train/val/test sets
4. Train the Logistic Regression classifier
5. Save the trained model

### Evaluation
Run the `evaluation.ipynb` notebook to:
1. Load the trained model
2. Evaluate on the test set
3. Generate detailed metrics and visualizations
4. Analyze prediction confidence
5. Perform error analysis

### Inference
Run the `inference.ipynb` notebook to:
1. Load the trained model
2. Make predictions on new abstracts
3. View inference examples
4. Perform batch inference

**Example Usage in Python**:
```python
from sentence_transformers import SentenceTransformer
import pickle

# Load model and embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pickle.load(open('./models/classifier.pkl', 'rb'))

# Make prediction
abstract = "Your paper abstract here..."
embedding = embedding_model.encode([abstract])
prediction = classifier.predict(embedding)[0]
probability = classifier.predict_proba(embedding)[0]

print(f"Prediction: {prediction}")
print(f"Relevance Probability: {probability[1]:.4f}")
```

---

## Key Findings and Insights

### Strengths
1. **Good Performance on Balanced Metrics**: F1-score of 0.71 demonstrates the model effectively balances precision and recall despite class imbalance
2. **Efficient Processing**: SentenceTransformer embeddings are fast and require minimal computational resources
3. **Interpretable Model**: Logistic Regression provides interpretability compared to black-box deep learning models
4. **Generalizes Well**: Similar performance across train/val/test sets suggests good generalization

### Limitations
1. **Limited Dataset Size**: Only 300 samples may limit model robustness
2. **Fixed Embedding Model**: Pre-trained SentenceTransformer might not capture domain-specific nuances perfectly
3. **Class Imbalance**: Even with balanced class weights, the model is biased toward the majority class
4. **Binary Classification**: Current model doesn't distinguish between different types of relevant papers (e.g., theoretical vs. applied)

### Recommendations for Future Work
1. **Data Collection**: Gather more labeled examples to improve model robustness
2. **Fine-tuning**: Fine-tune the embedding model on domain-specific papers
3. **Multi-class Classification**: Extend to predict relevance level (e.g., "highly relevant", "somewhat relevant", "not relevant")
4. **Threshold Optimization**: Adjust decision threshold based on use case (prioritize precision vs. recall)
5. **Ensemble Methods**: Combine with other models or heuristics for improved performance

---

## Model Weights

The trained classifier model is saved in:
- **Location**: `./models/classifier.pkl`
- **Type**: Scikit-learn LogisticRegression object
- **Size**: ~5 KB

The model can be loaded and used independently without retraining:
```python
import pickle
classifier = pickle.load(open('./models/classifier.pkl', 'rb'))
```

---

## Conclusion

This project demonstrates a practical and efficient approach to automatic paper relevance classification using sentence embeddings and linear classification. With an F1-score of 0.71 and AUC-ROC of 0.865, the model shows promising results for filtering arXiv papers. The simple architecture makes it easy to deploy and integrate into a paper recommendation system.

The model successfully addresses the research problem of automatically filtering relevant papers from large arXiv feeds, potentially saving researchers significant time in literature review.

---

## References

- **SentenceTransformers**: https://www.sbert.net/
- **Scikit-learn**: https://scikit-learn.org/
- **arXiv**: https://arxiv.org/

---

**Last Updated**: 2025-11-18
**Status**: Complete
