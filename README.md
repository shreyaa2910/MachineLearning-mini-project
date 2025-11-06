# 📊 Comparative Analysis of ML and DL Models for IMDb Sentiment Analysis

## 1. Description
This project conducts a *comparative study* of four different Machine Learning (ML) and Deep Learning (DL) models for binary text classification.  
The task is *Sentiment Analysis* on the IMDb movie review dataset, classifying reviews as either Positive or Negative.

Sentiment analysis is a crucial task in *Natural Language Processing (NLP)* with applications ranging from customer feedback analysis to social media monitoring.  
This project benchmarks classical bag-of-words models against sequential (RNN) and transformer-based (DistilBERT) architectures to evaluate trade-offs between *performance, complexity, and training time*.

> 🏆 Results show that a well-tuned *Logistic Regression* provides the best balance of high accuracy and efficiency, outperforming deep learning models under current training conditions.

---

## 2. Dataset Source and Preprocessing

### 📚 Dataset
- *Source:* IMDb Movie Reviews dataset (from tensorflow_datasets)
- *Size:* 50,000 movie reviews  
- *Split:*
  - Training: 20,000 reviews  
  - Validation: 5,000 reviews  
  - Test: 25,000 reviews  

---

### 🧹 Preprocessing
A “Single Source of Truth” approach was used by loading the full dataset into a Pandas DataFrame first.  
Two distinct preprocessing pipelines were applied for classical and deep learning models.

#### A. For Classical Models (Naive Bayes & Logistic Regression)
1. *Text Cleaning:* Removed HTML tags (<br />) and non-alphabetic characters using regex.  
2. *Normalization:* Converted all text to lowercase.  
3. *Tokenization & Filtering:* Split text into words and removed stopwords (from NLTK).  
4. *Lemmatization:* Reduced words to their base form using WordNetLemmatizer.  
5. *Vectorization:* Used TfidfVectorizer with:
   - max_features = 20,000
   - ngram_range = (1, 2) (unigrams and bigrams)

#### B. For Deep Learning Models (Bi-LSTM & DistilBERT)
1. *Bi-LSTM:*  
   - Used tf.keras.layers.TextVectorization for preprocessing (tokenization, integer encoding, padding).  
   - Parameters: VOCAB_SIZE = 20,000, MAX_SEQUENCE_LENGTH = 250.  

2. *DistilBERT:*  
   - Tokenized using the pre-trained DistilBertTokenizer.  
   - Added special tokens ([CLS], [SEP]), truncated, and padded to max_length = 250.

---

## 3. Methods

Four models were selected to represent a range of NLP techniques:

| Model | Description |
|--------|--------------|
| *Multinomial Naive Bayes (MNB)* | Fast probabilistic baseline that performs well with TF-IDF features. Assumes feature independence. |
| *Logistic Regression (LR)* | Strong, interpretable linear model; performs well on text classification with TF-IDF n-grams. |
| *Bidirectional LSTM (Bi-LSTM)* | RNN designed to capture sequential dependencies by reading text both forward and backward. |
| *DistilBERT* | A distilled, faster version of BERT using attention mechanisms for deep contextual understanding. Fine-tuned for this task. |

---

## 4. Steps to Run the Code

1. *Environment:*  
   - Run in *Google Colab*.  
   - Enable GPU: Runtime > Change runtime type > GPU (T4).

2. *Installation:*  
   Install required libraries:
   ```bash
   !pip install transformers datasets -q

## 5. 🧪 Experiments and Results Summary

### 📈 Quantitative Benchmarking
The comparison between classical and deep learning models was based on *Test Accuracy, **F1-Score, **Training Time, and **Inference Time*.

| Model | Test Accuracy | Test F1-Score | Training Time (s) | Inference Time (s/review) |
|--------|----------------|---------------|-------------------|----------------------------|
| *Logistic Regression* | *0.8837* | *0.8843* | 0.97 | 0.0000 |
| *Multinomial Naive Bayes* | 0.8540 | 0.8514 | *0.06* | 0.0000 |
| *Bidirectional LSTM* | 0.8430 | 0.8298 | 59.16 | 0.0002 |
| *DistilBERT* | 0.5000 | 0.6667 | 1055.10 | 0.0081 |

---

### 🔍 Analysis of Results

#### 1. Classical Models (LR & MNB)
- *Logistic Regression* achieved the *highest accuracy (88.4%)* and *F1-score (88.4%)*.  
- This strong performance can be attributed to the *TF-IDF bigrams*, which effectively capture sentiment context.  
- *Naive Bayes* was the *fastest to train (0.06s)* while maintaining a strong *85.4% accuracy*.  
- These results emphasize that *simple models can outperform complex ones* when well-tuned.

#### 2. Bi-LSTM Model
- The *Bi-LSTM* achieved *84.3% accuracy*, lower than both classical models.  
- This suggests that the *sequential dependencies* captured by RNNs in this dataset were less informative than TF-IDF features.  
- Training for *only 5 epochs* might have limited the model’s potential.

#### 3. DistilBERT Model (Training Failure)
- The *DistilBERT* model achieved *50.0% accuracy*, equivalent to random guessing.  
- Qualitative analysis revealed it predicted “Positive” for almost all inputs — a *training failure*.  
- Causes likely include:
  - *Insufficient training time* (only 2 epochs).  
  - *Improper learning rate* during fine-tuning.  
- This highlights that transformer models are *not plug-and-play* — they demand careful hyperparameter tuning to perform well.

---

### 📊 Visualizations

#### 1. Model Performance (Accuracy & F1)
Visual bar charts confirm that *Logistic Regression* led in both accuracy and F1-score.

> 📍 Suggestion: Insert the bar chart titled  
> *“Model Performance (Accuracy & F1)”*  
> (Generated in Notebook Cell 41)

---

#### 2. ROC Curve Comparison
The *ROC curves* demonstrate that *Logistic Regression* and *Naive Bayes* have curves closest to the *top-left corner*, indicating higher AUC values than Bi-LSTM.

> 📍 Suggestion: Insert the ROC plot titled  
> *“Combined ROC Curve Comparison”*  
> (Generated in Notebook Cell 56)

---

## 6. 🧾 Conclusion

The experiment reveals a powerful insight — *baselines are not to be underestimated*.

- *Logistic Regression* with *TF-IDF features* outperformed both the Bi-LSTM and a poorly-tuned DistilBERT model.  
- It was also *exponentially faster* to train and deploy.  
- *Classical models* remain strong contenders for text classification tasks, offering:
  - High accuracy  
  - Low computational cost  
  - Interpretability  

While *transformers* like DistilBERT have enormous potential, this project underscores that they require *careful fine-tuning* and *longer training* to reach their state-of-the-art potential.

> ⚡ *Takeaway:* A well-designed baseline can beat complex architectures when implemented and optimized efficiently.

---

## 7. 📚 References

- *Dataset:* [TensorFlow Datasets – IMDb Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)  
- *Libraries Used:*
  - Scikit-learn  
  - TensorFlow / Keras  
  - Hugging Face Transformers  

---

### 📘 Project Outline Recap
1. Description  
2. Dataset Source and Preprocessing  
3. Methods  
4. Steps to Run the Code  
5. Experiments and Results Summary  
6. Conclusion  
7. References

   
