# 📊 Comparative Analysis of ML and DL Models for IMDb Sentiment Analysis

## 1. Description
This project compares four ML/DL models on a binary sentiment analysis task, classifying "Positive" or "Negative" reviews from the IMDb dataset. This type of analysis is a critical NLP function used for social media monitoring and customer feedback.
The study benchmarks classical "bag-of-words" models (like Logistic Regression) against sequential (RNN) and transformer (DistilBERT) architectures, evaluating trade-offs in performance, complexity, and training speed. The findings indicate that, for this specific dataset, an optimized Logistic Regression model provided the best mix of accuracy and efficiency, surpassing the deep learning models under the tested conditions.
 
> 🏆 Results show that a well-tuned *Logistic Regression* provides the best balance of high accuracy and efficiency, outperforming deep learning models under current training conditions.

---

## 2. Dataset Source and Preprocessing

### 📚 Dataset
- *Source:* IMDb Movie Reviews dataset, loaded directly from tensorflow_datasets (TFDS).
- *Size:* 50,000 movie reviews.  
- *Split:* The dataset is split into:
  - Training: 20,000 reviews  
  - Validation: 5,000 reviews  
  - Test: 25,000 reviews  

---

### 🧹 Preprocessing
The methodology began by loading the full dataset into a Pandas DataFrame, establishing a "Single Source of Truth." This initial step was followed by the creation of two separate preprocessing pipelines, each tailored for a specific category of models.

#### A. For Classical Models (Naive Bayes & Logistic Regression)
This process involved a sequence of five steps:
•	First, the text was cleaned using regular expressions to remove HTML tags and non-alphabetic characters.
•	Second, all text was normalized to lowercase.
•	Third, the text was tokenized (split into words), and standard NLTK stopwords were filtered out.
•	Fourth, a WordNetLemmatizer was applied to standardize words to their base form.
•	Finally, the preprocessed text was vectorized using TfidfVectorizer. This vectorizer was limited to 20,000 features and set to include both unigrams and bigrams (ngram_range=(1, 2))


#### B. For Deep Learning Models (Bi-LSTM & DistilBERT)
1. *Bi-LSTM:*  
   This model's pipeline ingested the raw review text, letting an initial tf.keras.layers.TextVectorization layer perform all preprocessing. This layer was set to create a 20,000-word vocabulary and pad or truncate sequences to 250 tokens, handling all tokenization and integer mapping internally.  

2. *DistilBERT:*  
   For this model, the DistilBertTokenizer was used to tokenize the raw text. The process involved appending special [CLS] and [SEP] tokens and standardizing all sequences to a 250-token length through padding or truncation.

---

## 3. Methods

Four models were selected to represent a range of NLP techniques:

| Model | Description |
|--------|--------------|
| *Multinomial Naive Bayes (MNB)* | A fast, probabilistic baseline that pairs well with TF-IDF and assumes feature independence. |
| *Logistic Regression (LR)* | A powerful, interpretable linear model that often excels in text classification, especially with TF-IDF n-grams. |
| *Bidirectional LSTM (Bi-LSTM)* | An RNN that processes text in both directions (forwards and backwards) to capture richer sequential context. |
| *DistilBERT* | A smaller, faster version of the BERT transformer. It was pre-trained and then fine-tuned for this task, using attention mechanisms for deep contextual analysis. |

---
## 4. Steps to Run the Code

1. *Environment:*  
   - Run in *Google Colab*.  
   - Enable GPU: `Runtime > Change runtime type > GPU (T4).`

2. *Installation:*  
   Install required libraries:
   ```bash
   !pip install transformers datasets -q
   ```
3. *▶ Execute All*
Run all cells *sequentially from top to bottom*.

- 🕒 *Data loading and preprocessing (Parts 1–2)* may take a few minutes.  
- ⚙ *The Bi-LSTM model (Part 3)* will take approximately *1–2 minutes* to train (with a GPU).  
- 🧠 *The DistilBERT model (Part 4)* is computationally intensive and will take *~15–20 minutes* to train for *2 epochs*.

4. *📊 Results*
The final *quantitative comparison table, **performance graphs, and **qualitative analysis* are generated in *Part 5*.
---
6. *🧪 Experiments and Results Summary*

## 5. 📈 Quantitative Benchmarking
The primary comparison was based on Test Accuracy and Test F1-Score. Training and inference times were also recorded for a practical comparison of efficiency.

| Model | Test Accuracy | Test F1-Score | Training Time (s) | Inference Time (s/review) |
|--------|----------------|---------------|-------------------|----------------------------|
| *Logistic Regression* | *0.8837* | *0.8843* | 0.97 | 0.0000 |
| *Multinomial Naive Bayes* | 0.8540 | 0.8514 | *0.06* | 0.0000 |
| *Bidirectional LSTM* | 0.8430 | 0.8298 | 59.16 | 0.0002 |
| *DistilBERT* | 0.5000 | 0.6667 | 1055.10 | 0.0081 |

---

### 🔍 Analysis of Results

#### 1. Classical Models (LR & MNB)
  *Logistic Regression* (LR) emerged as the top performer, yielding the best accuracy ($\mathbf{88.4\%}$) and F1-score ($\mathbf{88.4\%}$). This superiority is likely attributable to the strong predictive capabilities of TF-IDF bigrams. Naive Bayes (MNB) established a very strong baseline, achieving $\mathbf{85.4\%}$ accuracy, and was the quickest model to train ($\mathbf{0.06s}$).

#### 2. Bi-LSTM Model
  The Bi-LSTM model showed lower performance than both classical approaches ($\mathbf{84.3\%}$ accuracy). This indicates that the sequential context learned by the RNN (trained over 5 epochs) provided less valuable information for this dataset than the TF-IDF n-gram features

#### 3. DistilBERT Model (Training Failure)
The DistilBERT model resulted in a clear training failure, achieving only $\mathbf{50.0\%}$ accuracy—the level of random chance for a balanced binary task. Section 5.2's qualitative review supports this, showing the model almost exclusively predicted "Positive." This poor outcome likely stems from inadequate training time (only 2 epochs) or an unsuitable learning rate for fine-tuning. This outcome underscores that transformer models are not instantly deployable and require substantial hyperparameter optimization to reach state-of-the-art performance

---

### 📊 Visualizations

#### 1. Model Performance (Accuracy & F1)
The performance hierarchy is visually reinforced by the bar charts generated in Section 5.1 of the notebook, which clearly show Logistic Regression as the leading model.
This conclusion is further supported by the ROC Curve Comparison plot (found in Section 5.2). The curves for Logistic Regression and Naive Bayes are situated closer to the top-left corner—indicating a higher Area Under the Curve (AUC)—compared to the LSTM model.

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

This project's key finding is a validation of the effectiveness of classical models. A meticulously configured Logistic Regression model, leveraging TF-IDF features, was found to be the optimal choice. It not only outperformed a Bi-LSTM and a suboptimally-tuned DistilBERT architecture but also demonstrated orders-of-magnitude faster training and deployment.
Although transformer models, like DistilBERT, represent the cutting edge of NLP, this experiment underscores the fact that achieving state-of-the-art results is contingent upon extensive and meticulous hyperparameter optimization. In its absence, strong classical baselines often retain their competitive edge due to their inherent efficiency and high effectiveness in many text classification scenarios


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

   
