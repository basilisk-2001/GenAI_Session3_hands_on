# GenAI_Session3_hands_on
This repository contains an intermediate hands-on that can be used to understand the functioning of Advanced NLP techniques

### 1. Sentiment Analysis Techniques
**Example: Analyzing Song Lyrics from Recent Hits**

**Objective:** Demonstrate sentiment analysis techniques using lyrics from recent popular songs.

**Activity:**
1. **Data Collection:**
   - Use lyrics from popular songs such as "Flowers" by Miley Cyrus and "Kill Bill" by SZA.
2. **Sentiment Analysis:**
   - Analyze the sentiment of these lyrics using a pre-trained model.

**Demonstration:**
```python
from transformers import pipeline

# Load a pre-trained sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis')

# Lyrics from recent popular songs
lyrics = [
    "I can buy myself flowers, write my name in the sand - Flowers by Miley Cyrus",
    "I might kill my ex, not the best idea - Kill Bill by SZA"
]

# Analyze the sentiment of each lyric
for lyric in lyrics:
    result = sentiment_analyzer(lyric)[0]
    print(f"Lyrics: {lyric}\nSentiment: {result['label']} (Score: {result['score']})\n")
```

### 2. Deep Dive into Sentiment Analysis - Polarity, Emotion Detection, etc.
**Example: Detecting Emotions in Oscar Acceptance Speeches**

**Objective:** Dive deeper into sentiment analysis by detecting emotions in Oscar acceptance speeches.

**Activity:**
1. **Data Collection:**
   - Use transcripts of recent Oscar acceptance speeches.
2. **Emotion Detection:**
   - Detect emotions (joy, gratitude, surprise, etc.) in the speeches.

**Demonstration:**
```python
from transformers import pipeline

# Load a pre-trained emotion detection model
emotion_analyzer = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# Sample excerpts from Oscar acceptance speeches
speeches = [
    "Thank you to the Academy, this is the greatest honor of my life!",
    "I never thought I would be standing here, thank you to everyone who believed in me."
]

# Detect emotions in the speeches
for speech in speeches:
    emotions = emotion_analyzer(speech)
    print(f"Speech: {speech}\nEmotions: {emotions}\n")
```

### 3. Text Classification Fundamentals - Topic Labeling, Spam Detection, etc.
**Example: Classifying Movie Reviews by Genre**

**Objective:** Demonstrate text classification by labeling movie reviews according to their genre.

**Activity:**
1. **Data Collection:**
   - Use a dataset of movie reviews from different genres.
2. **Topic Classification:**
   - Classify the reviews into genres (e.g., drama, comedy, action).

**Demonstration:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample movie reviews and genres
reviews = [
    "This movie was a heart-wrenching drama that kept me in tears.",
    "A hilarious comedy that had the whole theater laughing.",
    "An action-packed thriller that kept me on the edge of my seat."
]
genres = ['Drama', 'Comedy', 'Action']

# Create a training pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model on the reviews
model.fit(reviews, genres)

# Test the model on new reviews
test_reviews = [
    "The storyline was so emotional, I couldn't stop crying.",
    "The jokes were spot on and made everyone laugh out loud."
]
labels = model.predict(test_reviews)

print(f"Predicted genres: {labels}")
```

### 4. Feature Engineering & Model Optimization for NLP - Advanced Feature Selection & the Tuning of NLP Models
**Example: Optimizing a Model for Classifying Social Media Posts about a Real-Life Incident**

**Objective:** Show advanced feature selection and model tuning using social media posts about a real-life incident (e.g., a major sporting event like the FIFA World Cup).

**Activity:**
1. **Data Collection:**
   - Use a dataset of social media posts about the FIFA World Cup.
2. **Feature Engineering:**
   - Create advanced features (e.g., n-grams, TF-IDF scores).
3. **Model Optimization:**
   - Tune hyperparameters to improve model performance.

**Demonstration:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Sample social media posts and labels
posts = [
    "What an incredible goal! Best match of the World Cup so far.",
    "The referee's decision was totally unfair. Ruined the game for everyone.",
    "Can't believe my team made it to the finals! So excited!"
]
labels = ['Positive', 'Negative', 'Positive']

# Create a pipeline for feature extraction and model training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC())
])

# Define a grid of hyperparameters to search
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(posts, labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")
```

### 5. Sentiment-Driven Text Generation, a Crossover between Sentiment Analysis and Generative Modeling
**Example: Generating Movie Reviews for Recent Blockbusters with Specific Sentiments**

**Objective:** Combine sentiment analysis and generative modeling to create movie reviews with specific sentiments for recent blockbusters.

**Activity:**
1. **Sentiment Analysis:**
   - Analyze the sentiment of existing movie reviews.
2. **Text Generation:**
   - Generate new reviews with a specified sentiment (positive or negative).

**Demonstration:**
```python
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

# Define the desired sentiment
desired_sentiment = "positive"

# Create a prompt to generate a movie review with the desired sentiment
prompt = f"Write a {desired_sentiment} review for the recent blockbuster 'Spider-Man: No Way Home'."

# Generate the review using GPT-4
response = openai.Completion.create(
    engine="text-davinci-004",
    prompt=prompt,
    max_tokens=100,
    temperature=0.7
)
print(f"Generated Review: {response.choices[0].text.strip()}")
```
