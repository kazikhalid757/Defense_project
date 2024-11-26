import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google_play_scraper import app, reviews
import re
import streamlit as st

# Function to extract app ID from URL
def extract_app_id(url):
    """
    Extracts the app ID from a full Google Play Store URL.
    """
    match = re.search(r'id=([a-zA-Z0-9\._-]+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not extract app ID from URL")

# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), weights_only=True)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

# Scrape Google Play Store reviews using Google Play Scraper
def scrape_reviews(app_id, num_reviews=500):
    """
    Scrape reviews from the Google Play Store for a given app using the google-play-scraper.
    """
    reviews_list = []
    result, _ = reviews(app_id, count=num_reviews)  # Get reviews for the given app ID
    for review in result:
        reviews_list.append(review['content'])
    return reviews_list

# Preprocess and predict
def predict_review_authenticity(reviews, tokenizer, model):
    """
    Predict if an app is genuine or fake based on reviews.
    """
    inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)  # Get the predicted class for each review
    return predictions

# Streamlit UI for input and display results
def streamlit_app():
    st.title("Detecting Fack Apps throught User Review Analysis in Google play Store")
    
    # User input for app URL
    app_url = st.text_input("Enter Google Play Store App URL: ")
    
    model_path = "distilbert_model_pt.pt"  # Path to your saved model
    
    # Add some custom styling for Streamlit
    st.markdown("""
    <style>
    .title {
        color: #ff6347;
        font-weight: bold;
    }
    .subtitle {
        color: #4caf50;
        font-size: 20px;
    }
    .fake {
        color: #f44336;
        font-weight: bold;
    }
    .genuine {
        color: #4caf50;
        font-weight: bold;
    }
    .summary {
        font-size: 18px;
        background-color: #f0f4c3;
        padding: 10px;
        border-radius: 10px;
    }
    .results {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Detect"):
        if app_url:
            try:
                # Extract the app ID from the URL
                app_id = extract_app_id(app_url)
                st.markdown(f"**Extracted App ID**: {app_id}", unsafe_allow_html=True)
                
                # Load the model and tokenizer
                st.write("Loading model and tokenizer...")
                tokenizer, model = load_model_and_tokenizer(model_path)

                # Scrape reviews
                st.write("Scraping reviews from the Google Play Store...")
                reviews_list = scrape_reviews(app_id)
                
                # Predict authenticity
                st.write("Analyzing user reviews to determine the authenticity of the app...")
                predictions = predict_review_authenticity(reviews_list, tokenizer, model)
                
                # Count genuine and fake reviews
                fake_count = 0
                genuine_count = 0
                review_results = []
                
                for idx, prediction in enumerate(predictions):
                    if prediction.item() == 0:
                        fake_count += 1
                        
                    else:
                        genuine_count += 1
                        
                
                # Display overall result with colors
                if fake_count > genuine_count:
                    st.markdown(f"<h3 class='fake'>The app is likely FAKE based on reviews</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 class='genuine'>The app is likely GENUINE based on reviews</h3>", unsafe_allow_html=True)
                
                # Display the count of fake and genuine reviews with summary styling
                st.markdown(f"<div class='summary'>Genuine reviews: {genuine_count}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='summary'>Fake reviews: {fake_count}</div>", unsafe_allow_html=True)
                
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a valid app URL.")
    
if __name__ == "__main__":
    streamlit_app()
