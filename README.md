# 🐵 Monkey Species Classification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A comprehensive machine learning system for classifying 10 different monkey species using 6 different classification algorithms with detailed performance analysis.

---

## 📋 Problem Statement

The objective of this project is to develop and evaluate multiple machine learning models capable of accurately classifying images of monkeys into one of 10 distinct species. This multi-class image classification problem addresses challenges in:

- **Wildlife Conservation**: Automated species identification for monitoring populations
- **Educational Purpose**: Teaching tool for learning about primate diversity
- **Research Applications**: Supporting zoological and behavioral studies
- **Real-world ML Application**: Demonstrating comparative analysis of classical ML algorithms on image data

The challenge involves handling high-dimensional image data (12,288 features from 64x64 RGB images) and distinguishing between visually similar species with varying performance across different algorithmic approaches.

---

## 📊 Dataset Description

### Overview
- **Dataset Name**: 10 Monkey Species Image Classification Dataset
- **Source**: https://www.kaggle.com/datasets/slothkong/10-monkey-species/data
- **Total Classes**: 10 different monkey species
- **Data Split**:
  - Training Images: 1,370 images
  - Validation Images: 272 images
  - Total Images: 1,642 images

### Class Distribution

| Class ID | Latin Name | Common Name | Training Images | Validation Images |
|----------|-----------|-------------|-----------------|-------------------|
| n0 | *Alouatta palliata* | Mantled Howler | 131 | 26 |
| n1 | *Erythrocebus patas* | Patas Monkey | 139 | 28 |
| n2 | *Cacajao calvus* | Bald Uakari | 137 | 27 |
| n3 | *Macaca fuscata* | Japanese Macaque | 152 | 30 |
| n4 | *Cebuella pygmaea* | Pygmy Marmoset | 131 | 26 |
| n5 | *Cebus capucinus* | White-headed Capuchin | 141 | 28 |
| n6 | *Mico argentatus* | Silvery Marmoset | 132 | 26 |
| n7 | *Saimiri sciureus* | Common Squirrel Monkey | 142 | 28 |
| n8 | *Aotus nigriceps* | Black-headed Night Monkey | 133 | 27 |
| n9 | *Trachypithecus johnii* | Nilgiri Langur | 132 | 26 |

### Data Preprocessing

1. **Image Loading**: Images loaded from directory structure organized by class labels
2. **Resizing**: All images standardized to 64×64 pixels
3. **Color Space**: RGB format (3 channels)
4. **Normalization**: Pixel values scaled to [0, 1] range by dividing by 255
5. **Feature Extraction**: Images flattened into 1D vectors (64×64×3 = 12,288 features)
6. **Feature Scaling**: StandardScaler applied for zero mean and unit variance
7. **Class Balance**: Dataset is relatively balanced with 131-152 training samples per class

### Dataset Characteristics

- **Feature Dimensionality**: 12,288 features per sample
- **Data Type**: Continuous (normalized pixel values)
- **Challenge Level**: High - requires distinguishing subtle visual differences between species
- **Quality**: Professional wildlife photography with varying backgrounds and poses

---

## 🤖 Models Used

### Comprehensive Model Comparison

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---------------|----------|-----------|-----------|--------|----------|-----------|
| **Logistic Regression** | 0.5735 | 0.8672 | 0.5777 | 0.5735 | 0.5660 | 0.5256 |
| **Decision Tree** | 0.3971 | 0.6622 | 0.4058 | 0.3971 | 0.3910 | 0.3300 |
| **K-Nearest Neighbor** | 0.5074 | 0.8186 | 0.5241 | 0.5074 | 0.5062 | 0.4527 |
| **Gaussian Naive Bayes** | 0.3382 | 0.7584 | 0.3768 | 0.3382 | 0.3251 | 0.2660 |
| **Random Forest (Ensemble)** | 0.5588 | 0.8705 | 0.5813 | 0.5588 | 0.5539 | 0.5280 |
| **XGBoost (Ensemble)** | 0.5441 | 0.8930 | 0.5644 | 0.5441 | 0.5416 | 0.4939 |

### Performance Highlights

**Best Performers by Metric:**
- **Accuracy**: Logistic Regression (57.35%)
- **AUC Score**: XGBoost (89.30%)
- **Precision**: Random Forest (58.13%)
- **Recall**: Logistic Regression (57.35%)
- **F1 Score**: Logistic Regression (56.60%)
- **MCC Score**: Random Forest (52.80%)

---

## 🔍 Model Performance Observations

### 1. Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 0.5735 |
| AUC Score | 0.8672 |
| Precision | 0.5777 |
| Recall | 0.5735 |
| F1 Score | 0.5660 |
| MCC Score | 0.5256 |

**Observations:**
- ✅ **Best overall accuracy** (57.35%) and F1 score among all models
- ✅ Demonstrates excellent balance between precision and recall
- ✅ Strong AUC score (0.8672) indicates good class separation ability
- ✅ Fast training time and low computational requirements
- ⚠️ Linear decision boundaries limit ability to capture complex non-linear patterns in image data
- ⚠️ Treats all features independently, missing spatial relationships between pixels
- **Conclusion**: Excellent baseline model that performs surprisingly well despite simplicity. Recommended for production where interpretability and speed are priorities.

---

### 2. Decision Tree Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 0.3971 |
| AUC Score | 0.6622 |
| Precision | 0.4058 |
| Recall | 0.3971 |
| F1 Score | 0.3910 |
| MCC Score | 0.3300 |

**Observations:**
- ❌ **Poorest performance** across nearly all metrics
- ⚠️ High tendency to overfit with deep trees on high-dimensional data
- ⚠️ Single tree cannot capture complex patterns in 12,288-dimensional space
- ⚠️ Susceptible to noise in pixel-level features
- ✅ Fast prediction time once trained
- ✅ Provides interpretable decision rules (though impractical with many features)
- **Conclusion**: Not suitable for image classification with raw pixels. Performance significantly improves when used in ensemble methods (Random Forest). Better suited for structured tabular data with fewer features.

---

### 3. K-Nearest Neighbor Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 0.5074 |
| AUC Score | 0.8186 |
| Precision | 0.5241 |
| Recall | 0.5074 |
| F1 Score | 0.5062 |
| MCC Score | 0.4527 |

**Observations:**
- ⚠️ **Moderate performance** placing in middle of the pack
- ✅ Non-parametric approach effectively captures local patterns and similarities
- ✅ Good AUC score (0.8186) shows reasonable discriminative ability
- ⚠️ Suffers from curse of dimensionality with 12,288 features
- ⚠️ Computationally expensive for predictions (searches entire training set)
- ⚠️ Memory-intensive as it requires storing all training samples
- ⚠️ Sensitive to choice of distance metric in high-dimensional spaces
- **Conclusion**: Better performance could be achieved with dimensionality reduction (PCA, feature selection) or distance metric learning. Not recommended for real-time applications due to prediction latency.

---

### 4. Gaussian Naive Bayes Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 0.3382 |
| AUC Score | 0.7584 |
| Precision | 0.3768 |
| Recall | 0.3382 |
| F1 Score | 0.3251 |
| MCC Score | 0.2660 |

**Observations:**
- ❌ **Second-worst performer** with lowest accuracy (33.82%)
- ⚠️ Naive independence assumption severely violated by spatially correlated pixels
- ⚠️ Gaussian distribution assumption doesn't hold well for pixel intensities
- ✅ Surprisingly decent AUC score (0.7584) despite low accuracy
- ✅ Extremely fast training and prediction
- ✅ Minimal memory footprint
- ⚠️ Better suited for text classification or datasets with truly independent features
- **Conclusion**: Not appropriate for image classification with raw pixels. The strong feature independence assumption makes it fundamentally unsuited for spatial data. Consider only if extreme speed is the only priority and low accuracy is acceptable.

---

### 5. Random Forest Classifier (Ensemble)

| Metric | Score |
|--------|-------|
| Accuracy | 0.5588 |
| AUC Score | 0.8705 |
| Precision | 0.5813 |
| Recall | 0.5588 |
| F1 Score | 0.5539 |
| MCC Score | 0.5280 |

**Observations:**
- 🏆 **Best precision** (58.13%) and **best MCC score** (52.80%)
- ✅ Strong second-place accuracy, close behind Logistic Regression
- ✅ Ensemble approach effectively reduces overfitting seen in single decision tree
- ✅ Robust to outliers and handles high-dimensional data well
- ✅ High AUC score (0.8705) demonstrates excellent ranking ability
- ✅ Provides feature importance rankings
- ⚠️ Computationally intensive training with 100 trees
- ⚠️ Larger model size requires more memory
- **Conclusion**: Excellent choice for robust predictions with good generalization. Best balance between performance and interpretability among ensemble methods. Recommended when computational resources are available and precision is critical.

---

### 6. XGBoost Classifier (Ensemble)

| Metric | Score |
|--------|-------|
| Accuracy | 0.5441 |
| AUC Score | 0.8930 |
| Precision | 0.5644 |
| Recall | 0.5441 |
| F1 Score | 0.5416 |
| MCC Score | 0.4939 |

**Observations:**
- 🏆 **Highest AUC score** (89.30%) - best class separation and probability estimates
- ✅ Advanced gradient boosting with built-in regularization
- ✅ Handles complex patterns better than individual trees
- ✅ Efficient implementation with parallel processing
- ✅ Best model for ranking and probability calibration tasks
- ⚠️ Slightly lower accuracy (54.41%) than Logistic Regression and Random Forest
- ⚠️ More hyperparameters to tune compared to simpler models
- ⚠️ Less interpretable than Random Forest
- **Conclusion**: Best choice when probability estimates are critical (e.g., ranking predictions by confidence). Superior AUC indicates excellent discriminative ability. Recommended for applications where predicting class probabilities is more important than hard classifications.

---

## 📈 Overall Analysis

### Key Findings

1. **Linear vs Non-linear Models**:
   - Logistic Regression (linear) surprisingly outperforms most non-linear models in accuracy
   - Suggests that complex patterns may not be fully captured with current feature representation

2. **Ensemble Performance**:
   - Ensemble methods (Random Forest, XGBoost) show more consistent performance across metrics
   - XGBoost achieves best AUC, indicating superior probability estimates

3. **Feature Representation Challenge**:
   - All models achieve moderate accuracy (34-57%), indicating limitation of raw pixel features
   - High dimensionality (12,288 features) poses challenges for distance-based and tree-based methods

4. **Trade-offs Identified**:
   - **Speed vs Accuracy**: Naive Bayes fastest but least accurate
   - **Interpretability vs Performance**: Decision Tree interpretable but poor performance
   - **Simplicity vs Robustness**: Logistic Regression simple yet competitive

### Recommendations for Future Work

1. **Feature Engineering**:
   - Apply dimensionality reduction (PCA, t-SNE, UMAP)
   - Extract hand-crafted features (HOG, SIFT, color histograms)
   - Use pre-trained CNN features (transfer learning)

2. **Model Improvements**:
   - Implement Convolutional Neural Networks (CNNs)
   - Try transfer learning with ResNet, VGG, or EfficientNet
   - Expected accuracy improvement: 70-95%

3. **Data Augmentation**:
   - Rotation, flipping, brightness/contrast adjustments
   - Increase effective training data size

4. **Hyperparameter Optimization**:
   - Grid search or Bayesian optimization
   - Cross-validation for robust parameter selection

---

## 🚀 Streamlit Application Features

### Interactive Web Interface

✅ **Dataset Upload Option** - Upload CSV files with test data for predictions

✅ **Model Selection Dropdown** - Choose from 6 different trained models

✅ **Evaluation Metrics Display** - Comprehensive metrics for each model

✅ **Confusion Matrix Visualization** - Visual representation of classification performance

✅ **Classification Report** - Detailed per-class performance metrics

✅ **Performance Comparison** - Interactive charts comparing all models

✅ **Feature Importance** - For tree-based models (Decision Tree, Random Forest, XGBoost)

---

## 📦 Installation & Deployment

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/monkey-classification.git
cd monkey-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Prepare Repository**:
   - Ensure `app.py`, `requirements.txt`, and `README.md` are in root or specified folder
   - Commit and push to GitHub

2. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository, branch, and `app.py`
   - Click "Deploy"

3. **Monitor**:
   - Check deployment logs for any errors
   - App will be live at `https://your-app-name.streamlit.app`

---

## 📁 Project Structure

```
streamlit-app/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Directory for saved models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
└── notebooks/                      # Jupyter notebooks
    └── classification_models.ipynb  # Model training notebook
```

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Pillow** - Image processing

---

## 📊 Usage

### Running Predictions

1. Launch the application
2. Navigate to "Make Predictions" page
3. Select desired model from dropdown
4. Upload CSV file with test data
5. Click "Run Prediction"
6. View results including:
   - Predicted classes
   - Confidence scores
   - Confusion matrix
   - Classification report

### Expected CSV Format

```csv
feature_0,feature_1,feature_2,...,feature_12287,label
0.234,0.456,0.789,...,0.123,n0
0.567,0.890,0.123,...,0.456,n1
```

- **Features**: 12,288 columns of normalized pixel values
- **Label** (optional): True class label for evaluation (n0-n9)

---

## 🎯 Performance Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Proportion of correct predictions | Higher is better (0-1) |
| **AUC Score** | Area Under ROC Curve | Measures ranking quality (0-1) |
| **Precision** | Ratio of true positives to predicted positives | Minimizes false alarms |
| **Recall** | Ratio of true positives to actual positives | Minimizes missed cases |
| **F1 Score** | Harmonic mean of precision and recall | Balanced performance metric |
| **MCC Score** | Matthews Correlation Coefficient | Accounts for class imbalance (-1 to 1) |

---

## ⚠️ Known Limitations

1. **Feature Representation**: Raw pixel features are suboptimal for complex images
2. **Model Complexity**: Classical ML models limited compared to deep learning
3. **Computational Resources**: Limited by Streamlit Cloud free tier
4. **Image Resolution**: 64×64 pixels may lose important details
5. **Generalization**: Models trained on specific dataset may not generalize to all monkey images

---

## 🔮 Future Enhancements

- [ ] Implement CNN-based models (ResNet, EfficientNet)
- [ ] Add transfer learning capabilities
- [ ] Real-time image upload and prediction
- [ ] Model explainability with LIME/SHAP
- [ ] API endpoint for programmatic access
- [ ] Mobile-responsive design improvements
- [ ] Batch prediction support
- [ ] Model retraining interface

---

## 📄 License

This project is created for educational purposes as part of a machine learning course assignment.

---

## 👥 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset: 10 Monkey Species Classification Dataset
- Streamlit Community for deployment platform
- Scikit-learn and XGBoost development teams
- Course instructors and teaching assistants

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact via email

---

**Last Updated**: February 2026

**Version**: 1.0.0
