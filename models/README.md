# 📁 Models Directory

This directory contains the machine learning models and training notebooks.

## 📄 Files

### `classification_models.ipynb`
Complete Jupyter notebook containing:
- Data loading and preprocessing
- Implementation of all 6 classification models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Gaussian Naive Bayes
  5. Random Forest (Ensemble)
  6. XGBoost (Ensemble)
- Training and evaluation for each model
- Calculation of 6 evaluation metrics per model
- Visualizations and comparative analysis

## 🔧 Usage

### Running the Notebook

```bash
# Navigate to project directory
cd streamlit-app

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook models/classification_models.ipynb
```

### Training Models

Execute all cells in sequence:
1. Import libraries
2. Load dataset
3. Preprocess data
4. Train each model
5. Evaluate and compare results

## 📊 Model Files

If you train and save models, you can store them here:

```
models/
├── classification_models.ipynb    (training notebook)
├── logistic_regression.pkl        (saved model)
├── decision_tree.pkl              (saved model)
├── knn.pkl                        (saved model)
├── naive_bayes.pkl                (saved model)
├── random_forest.pkl              (saved model)
└── xgboost.pkl                    (saved model)
```

### Saving Models

```python
import pickle

# After training
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
```

### Loading Models

```python
import pickle

# In your Streamlit app
with open('models/logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 📝 Notes

- Notebook contains all required implementations for the assignment
- Models are trained on 10 monkey species dataset
- All evaluation metrics are calculated and displayed
- Results are used in the Streamlit application

## ⚠️ Important

- Keep this directory in your GitHub repository
- Required for assignment submission
- Demonstrates your model implementation work
- Shows commit history for plagiarism check
