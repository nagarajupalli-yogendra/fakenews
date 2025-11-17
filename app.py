import streamlit as st
import pandas as pd
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# Using Multinomial Naive Bayes (MNB) for fast text classification
from sklearn.naive_bayes import MultinomialNB 
import numpy as np
import warnings

# Suppress harmless future warnings from scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 0. Initial Setup & Constants ---

# User requested threshold: 0.70 (70%)
FINAL_THRESHOLD = 0.55

@st.cache_resource
def download_nltk_resources():
    """Ensures NLTK stopwords are downloaded once."""
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        nltk.download('stopwords')

download_nltk_resources()

# Initialize Stemmer and Stopwords globally 
port_stem = PorterStemmer()
nltk_stopwords = stopwords.words('english')

# --- 1. Data Cleaning and Preprocessing Function ---
def text_preprocessing(text):
    """Cleans and preprocesses the text for the model."""
    
    if not isinstance(text, str):
        return ''
        
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in nltk_stopwords]
    text = ' '.join(text)
    
    return text

# --- 2. Model Training and Caching (Multinomial Naive Bayes) ---
@st.cache_resource(show_spinner="Training ultra-fast Multinomial Naive Bayes model... Please wait (seconds).") 
def train_model():
    """Loads and preprocesses data, trains the vectorizer, and trains the MNB model."""
    
    try:
        # Load the datasets
        fake_df = pd.read_excel(r'C:\Users\Admin\OneDrive\Desktop\Fake.xlsx')
        fake_df['label'] = 1
        true_df = pd.read_excel(r'C:\Users\Admin\OneDrive\Desktop\True.xlsx')
        true_df['label'] = 0
        
    except FileNotFoundError as e:
        st.error(f"Error: Could not find required file **{e.filename}**. Ensure both 'Fake.csv' and 'True.csv' are in the directory.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return None, None
        
    news_df = pd.concat([fake_df, true_df], ignore_index=True)
    # Combining title and text for content feature
    news_df['content'] = news_df['title'].astype(str).fillna('') + ' ' + news_df['text'].astype(str).fillna('')
    news_df['content'] = news_df['content'].apply(text_preprocessing)
    news_df = news_df[news_df['content'].str.strip().astype(bool)]
    
    X = news_df['content'].values
    Y = news_df['label'].values
    
    if len(np.unique(Y)) < 2:
        st.error("Training Error: Not enough unique classes (Real/Fake) left in the dataset after cleaning.")
        return None, None

    # Splitting data (ensures Y_train is defined for training)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # FINAL MODEL: Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_vectorized, Y_train)

    return model, vectorizer

# --- 3. Streamlit Application Interface (main) ---
def main():
    """Main function to run the Streamlit app."""
    
    # 1. Train/Load the model 
    classifier, vectorizer = train_model()

    # 2. Stop if training failed
    if classifier is None:
        st.stop()
        
    # 3. Main Application Content (No sidebar content)
    st.title("📰 Real-Time Fake News Detector")
    st.markdown("A **Multinomial Naive Bayes** classifier optimized for speed and text accuracy.")
    st.markdown("Enter a news article title and/or body text below to check its authenticity.")
    
    st.markdown("---")
    
    # Form to handle input and button press reliably
    with st.form(key='prediction_form'):
        user_input = st.text_area(
            "Paste News Article Content Here:",
            height=250,
            placeholder="e.g., 'WASHINGTON (Reuters) - The U.S. Congress reached an agreement to avert a government shutdown at the last minute.'"
        )
        
        submit_button = st.form_submit_button("Check Authenticity", type="primary")

    if submit_button:
        if user_input:
            with st.spinner('Analyzing Text...'):
                processed_input = text_preprocessing(user_input)
                
                if not processed_input:
                    st.warning("Input text is too short or resulted in an empty string after cleaning. Please enter more substantive text.")
                    return
                
                input_vector = vectorizer.transform([processed_input])
                prediction_proba = classifier.predict_proba(input_vector)[0]
                
                # Ensure robust float comparison 
                fake_proba = float(prediction_proba[1])
                THRESHOLD = FINAL_THRESHOLD # 0.70
                
                # CLASSIFICATION LOGIC (0.70 threshold)
                if fake_proba > THRESHOLD:
                    label = "FAKE NEWS"
                    confidence = fake_proba * 100 
                    color = "red"
                    icon = "❌"
                else:
                    label = "REAL NEWS"
                    confidence = prediction_proba[0] * 100
                    color = "green"
                    icon = "✅"
                
                # Display Results
                st.subheader(f"{icon} Prediction Result: {label}")
                st.markdown(f"**Confidence Score:** **<span style='color:{color}; font-size: 24px;'>{confidence:.2f}%</span>**", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 📊 Details & Confidence Breakdown (Explainability)")
                st.markdown(f"- **Model Used:** Multinomial Naive Bayes")
                st.markdown(f"- The model required **{THRESHOLD*100:.0f}%** certainty to flag this as Fake.")
                st.markdown(f"- Probability of **Real News (0)**: `{prediction_proba[0]:.4f}`")
                st.markdown(f"- Probability of **Fake News (1)**: `{prediction_proba[1]:.4f}`")
                
        else:
            st.warning("Please paste some text into the box and press the button.")

if __name__ == '__main__':
    main()