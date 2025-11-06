# DOWNLOAD IPYNB FILE TO SEE, AS IT IS NOT AVAILABLE FOR PREVIEW

# 📊 Comparative Analysis of ML and DL Models for IMDb Sentiment Analysis

## 1. Description
The current project compares four ML/DL models to a binary sentiment analysis task, i.e., defining Positive or Negative reviews with regards to IMDb data. It is a critical NLP task that is applied in the social media monitoring and customer feedback. The authors compare classical bag-of-words (e.g. Logistic Regression) models with sequential (RNN) and transformer (DistilBERT) architectures, in terms of trade-offs between their performance, complexity and training speed. The results show that there is an optimized Logistic Regression model that is the most effective in terms of accuracy and efficiency, and it was outperforming the deep learning models in the conditions in which the experiments were conducted.
 
> 🏆 Results demonstrates that a properly tuned Logistic Regression gives optimal balance between high accuracy and high efficiency, and beats deep learning models in present training circumstances.

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
The first step of methodology was loading the entire dataset in a Pandas DataFrame, which defined a Single Source of Truth. This first step was succeeded by the development of two different preprocessing pipelines that were designed to be used in a particular category of models.


#### A. For Classical Models (Naive Bayes & Logistic Regression)
This was done in five steps, which were: • Second, text was put into lower case form. Thirdly, the text was tokenized (cut into words) and common NLTK stopwords were removed.

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
  * The best performing was the Logistic Regression (LR) with the highest accuracy (88.4) and F1-score (88.4). This dominance can probably be explained by high predictive abilities of TF-IDF bigrams. Naive Bayes (MNB) had a very high baseline with a 85.4 accuracy, and is the fastest to train (0.06s).

#### 2. Bi-LSTM Model
 * The Bi-LSTM model was comparatively worse than the classical ones (84.3 accuracy). It suggests that the sequential context that was trained on the RNN (5 epochs) offered less meaningful information to this dataset compared with the TF-IDF n-gram features.

#### 3. DistilBERT Model (Training Failure)
 * The DistilBERT model found that it trained a clear flop that only had 50.0 accuracy the chance of a task with a balanced binary. This is reinforced in the qualitative review of section 5.2, which indicates that the model generally only predicted "Positive." This weak performance is probably due to lack of sufficient training time (2 epochs only) or improper learning rate to fine-tune. This result highlights the fact that transformer models cannot be immediately utilized and need significant hyperparameter optimization until they can reach state-of-the-art.

---

### 📊 Visualizations

#### 1. Model Performance (Accuracy & F1)
The bar charts created in Section 5.1 of the notebook, which obviously indicate the most influential model to be Logistic Regression, are the visual reinforcement of the performance hierarchy.
This is further confirmed by the plot of the ROC Curve Comparison (located in Section 5.2). The Logistic Regression and Naive Bayes curves are placed nearer to the top-left corner (which means that they have a larger Area Under the Curve (AUC)).

<img width="1206" height="585" alt="1" src="https://github.com/user-attachments/assets/1f6237de-34a8-4820-98b8-49ee9c39a68d" />

---

#### 2. ROC Curve Comparison
The ROC curves indicate that the curve of the Logistic Regression and the Naive Bayes is nearest to the upper left part of the curve, which means that these two classifiers show greater values of the AUC when compared to Bi-LSTM.

<img width="850" height="707" alt="2" src="https://github.com/user-attachments/assets/74eed1cd-a36c-460d-936b-cf02b32d4f28" />


---

## 6. 🧾 Conclusion

The main finding of this project is the approval of the validity of classical models. The best option turned out to be a carefully-tuned Logistic Regression model, which makes use of TF-IDF features. Not only did it outperform a Bi-LSTM and a suboptimally-tuned DistilBERT architecture, but also it had orders-of-magnitudes faster training and deployment. Even though transformer models, such as DistilBERT, are the state-of-the-art in NLP, this experiment highlights the fact that to be able to achieve state-of-the-art results, careful, and thorough hyperparameter optimization is essential. Without it, powerful classical baselines will still have competitive advantage as their efficiency and high effectiveness in most of the text classification cases can be inbuilt.


> ⚡ *Takeaway:* Well-designed baselines can be used in complex architectures to succeed when done and optimised.

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

   
