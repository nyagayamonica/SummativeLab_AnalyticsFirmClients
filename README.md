# Customer Reviews Analysis

## Summative Lab – Analytics Firm Clients

This project analyzes customer review data using three analytics approaches:

1. **Natural Language Processing (NLP)** to clean review text, identify common language patterns, and analyze sentiment.
2. **Time Series Analysis** to examine changes in review activity and customer ratings over time.
3. **Neural Networks** to predict customer star ratings from review text.

The analysis uses the `amazon_reviews_lab.csv` dataset and demonstrates how text analytics, temporal analysis, and deep learning can be combined to better understand customer experiences.

---

## Business Problem

Customer reviews contain valuable information about product experience, satisfaction, and areas of concern. However, manually reviewing large amounts of text is time-consuming.

The purpose of this project is to explore how analytics can be used to:

- Identify common topics and product features discussed by customers.
- Understand the overall sentiment expressed in customer reviews.
- Examine whether customer review activity and ratings change over time.
- Build a neural network capable of predicting star ratings from written review content.

---

## Dataset

The dataset contains **997 customer reviews** and **9 variables**, including:

- `product_id` – product identifier
- `rating` – customer rating from 1 to 5 stars
- `review_text` – full customer review
- `review_summary` – short review summary
- `review_time_raw` – original review date
- `unix_review_time` – Unix-format review time
- `timestamp` – review timestamp
- `sentiment_label` – sentiment category
- `review_length_words` – review length in words

### Rating Distribution

| Rating | Number of Reviews |
|---|---:|
| 1 Star | 87 |
| 2 Stars | 48 |
| 3 Stars | 94 |
| 4 Stars | 188 |
| 5 Stars | 580 |

The dataset is therefore **imbalanced**, with 5-star reviews representing the majority of observations.

---

## Part 1: Natural Language Processing

### Text Preprocessing

The review text was cleaned before further analysis. The preprocessing pipeline included:

- Converting text to lowercase
- Removing URLs
- Removing punctuation
- Tokenizing the text
- Keeping alphabetic tokens
- Removing English stopwords
- Lemmatizing words

The processed tokens were then combined to calculate word and bigram frequencies.

### Most Common Words

The ten most frequent words after preprocessing were:

| Word | Frequency |
|---|---:|
| nook | 1,509 |
| book | 913 |
| one | 569 |
| kindle | 561 |
| screen | 496 |
| like | 459 |
| read | 454 |
| work | 446 |
| get | 424 |
| great | 422 |

These words show that many reviews focus on e-readers, reading, screens, and product performance.

### Most Common Bigrams

The most common two-word combinations included:

- `barnes noble`
- `nook color`
- `battery life`
- `touch screen`
- `sd card`
- `nook hd`
- `work great`
- `nook tablet`
- `work well`
- `read book`

The bigrams provide more context than individual words. Phrases such as **“work great”** and **“work well”** suggest positive product experiences, while **“battery life,” “touch screen,”** and **“sd card”** highlight product features customers frequently discuss.

### Sentiment Analysis

VADER sentiment analysis was used to classify reviews as positive, neutral, or negative.

Before the additional text preprocessing, the sentiment distribution was:

- **Positive:** 637 reviews
- **Neutral:** 321 reviews
- **Negative:** 39 reviews

After preprocessing, the sentiment distribution shifted substantially toward positive and negative classifications, with fewer reviews remaining neutral. This suggests that removing noise such as stopwords and punctuation made sentiment-bearing words more prominent.

A sentiment-based star-rating estimate was also created from VADER compound scores. The predicted ratings matched the original customer ratings only **32.1%** of the time, showing that sentiment scores alone are not sufficient for accurate five-class star-rating prediction.

---

## Part 2: Time Series Analysis

The timestamp variable was converted to a datetime index and customer activity was resampled by month.

The dataset covers reviews from:

- **Earliest review:** 12 May 2000
- **Latest review:** 17 July 2014

The analysis calculated:

- Monthly review count
- Monthly average rating
- Three-month rolling average of monthly ratings

### Review Volume

Review activity was initially very low and irregular. Activity began increasing around 2010, followed by a substantial rise between approximately 2011 and 2013. Review volume later declined toward 2014.

This pattern suggests that customer engagement with the reviewed products increased significantly during the middle of the observed period before slowing later.

### Average Customer Rating

Monthly average ratings fluctuate considerably, particularly during months with fewer reviews. From roughly 2010 onward, ratings became more stable and generally remained between about **3.5 and 4.5 stars**.

### Three-Month Rolling Average

The three-month rolling average reduces short-term monthly fluctuations and provides a clearer view of the underlying trend.

The smoothed results suggest a slight improvement in customer ratings over time, followed by stabilization at a relatively high level. This indicates that overall customer satisfaction remained fairly strong despite month-to-month variation.

---

## Part 3: Neural Network Rating Prediction

A multi-class neural network was developed to predict customer ratings from review text.

### Text Vectorization

Review text was transformed into numerical features using **TF-IDF** with:

- `max_features=5000`
- English stopword removal
- Unigrams and bigrams using `ngram_range=(1, 2)`

The resulting TF-IDF matrix had a shape of:

`(997, 5000)`

### Train-Test Split

The data was split into:

- **797 training observations**
- **200 testing observations**

A stratified split was used to preserve the original rating distribution.

Ratings were converted from **1–5** to **0–4** for TensorFlow compatibility and then one-hot encoded.

### Neural Network Architecture

The model consisted of:

- Input layer with 5,000 TF-IDF features
- Dense hidden layer with 128 neurons and ReLU activation
- Dropout layer with a rate of 0.30
- Dense hidden layer with 64 neurons and ReLU activation
- Output layer with 5 neurons and Softmax activation

The model contained **648,709 trainable parameters**.

### Model Training

The neural network was compiled using:

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Metric:** Accuracy
- **Epochs:** 10
- **Batch size:** 32
- **Validation split:** 20%

---

## Model Performance

The final model achieved:

- **Test Accuracy:** 56.5%
- **Test Loss:** 1.50
- **Macro F1-score:** 0.23
- **Weighted F1-score:** 0.48

### Classification Performance

| Rating Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 1 Star | 0.29 | 0.12 | 0.17 | 17 |
| 2 Stars | 0.00 | 0.00 | 0.00 | 10 |
| 3 Stars | 0.14 | 0.05 | 0.08 | 19 |
| 4 Stars | 0.33 | 0.11 | 0.16 | 38 |
| 5 Stars | 0.61 | 0.91 | 0.73 | 116 |

The model performs best on the dominant **5-star class**, correctly identifying **106 of 116** 5-star reviews. It failed to correctly identify any of the 10 two-star reviews.

The confusion matrix shows that many reviews from the minority classes were incorrectly predicted as 5 stars. This confirms that **class imbalance is the main limitation of the model**.

---

## Hyperparameter Experiment

The TF-IDF vocabulary size was initially set to **1,000 features**. Increasing it to **5,000 features** allowed the model to capture more information from the reviews.

This adjustment:

- Increased test accuracy from approximately **52.5% to 56.5%**
- Reduced test loss from approximately **4.00 to 1.50**

The richer text representation therefore improved model performance, although it did not solve the class-imbalance problem.

---

## Key Findings

The analysis produced several important findings:

- Customer discussions are strongly centered on e-reader functionality, reading experience, screen performance, battery life, and storage.
- Common phrases such as **“work great”** and **“work well”** provide evidence of positive customer experiences.
- Text preprocessing made sentiment classifications more distinct by reducing the number of neutral classifications.
- Review activity increased substantially around 2011–2013 before declining toward 2014.
- Average ratings show a slight upward long-term trend followed by stabilization at relatively high levels.
- The neural network achieved moderate overall accuracy but performed poorly across minority rating classes.
- The strong imbalance toward 5-star reviews causes the model to favor predicting the majority class.
- Increasing TF-IDF vocabulary size improved performance but was not sufficient to create balanced predictions.

---

## Recommendations for Model Improvement

Future versions of the project could improve predictive performance by:

- Applying **class weights** so errors on minority rating classes receive greater importance.
- Experimenting with oversampling or other techniques for handling class imbalance.
- Tuning the TF-IDF vocabulary size and n-gram range.
- Using **early stopping** to reduce overfitting.
- Tuning the learning rate, dropout rate, batch size, number of epochs, and hidden-layer sizes.
- Comparing the neural network with baseline machine-learning models such as Logistic Regression, Naive Bayes, or Support Vector Machines.
- Evaluating macro F1-score alongside accuracy because macro F1 gives equal importance to all rating classes.

---

## Technologies and Libraries

The project was developed in Python using:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- TensorFlow / Keras
- Statsmodels
- SciPy

---

## Repository Structure

```text
customer-reviews-analysis/
│
├── C09_M08.ipynb
├── amazon_reviews_lab.csv
└── README.md
```

## How to Run the Project

1. Clone or download the repository.
2. Install the required Python libraries.
3. Place `amazon_reviews_lab.csv` in the project directory.
4. Open `C09_M08.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
5. Run the notebook cells in order from top to bottom.
6. Ensure the required NLTK resources are downloaded when prompted.

Example installation:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow statsmodels scipy
```

---

## Conclusion

This project demonstrates how NLP, time series analysis, and neural networks can provide complementary views of customer review data. NLP identifies the language, product features, and sentiment expressed by customers, while time series analysis shows how review activity and satisfaction change over time. The neural network demonstrates the potential to predict star ratings directly from review text, achieving **56.5% test accuracy**.

However, the classification results also demonstrate why accuracy should not be considered in isolation. Because 5-star reviews dominate the dataset, the model performs well on that class but poorly on less common ratings. Future work should therefore prioritize class-imbalance handling and balanced evaluation metrics before the model is considered suitable for reliable automated rating prediction.

---

## Author

**Monica Achieng Nyagaya**

*Data Science Summative Lab – Customer Reviews Analysis*
