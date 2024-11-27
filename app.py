import torch
from transformers import DistilBertTokenizerFast
from google_play_scraper import reviews
import streamlit as st
import re

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

# Load the full model and tokenizer
def load_model_and_tokenizer(model_path):
    """
    Load the full saved model and tokenizer.
    """
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()  # Set the model to evaluation mode
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Error loading model or tokenizer: {str(e)}")

# Scrape Google Play Store reviews
def scrape_reviews(app_id, num_reviews=500):
    """
    Scrape reviews from the Google Play Store for a given app using the google-play-scraper.
    """
    reviews_list = []
    result, _ = reviews(app_id, count=num_reviews)  # Get reviews for the given app ID
    for review in result:
        reviews_list.append(review['content'])
    return reviews_list

# Predict review authenticity
def predict_review_authenticity(reviews, model, tokenizer):
    """
    Predict if an app is genuine or fake based on reviews.
    """
    inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)  # Get the predicted class for each review
    return predictions

# Streamlit app
def streamlit_app():
    st.title("Fake App Detection Based on User Reviews")

    # Input for Google Play Store App URL
    app_url = st.text_input("Enter Google Play Store App URL:")

    # Model path (update with the correct path to your saved model)
    model_path = "distilbert_model_pt.pt"

    # Style for Streamlit UI
    st.markdown("""
    <style>
    .fake { color: red; font-weight: bold; }
    .genuine { color: green; font-weight: bold; }
    .summary { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Detect"):
        if app_url:
            try:
                # Extract app ID from URL
                app_id = extract_app_id(app_url)
                st.markdown(f"**Extracted App ID**: {app_id}")

                # Load the model and tokenizer
                st.write("Loading model and tokenizer...")
                model, tokenizer = load_model_and_tokenizer(model_path)

                # Scrape reviews
                st.write("Scraping reviews from the Google Play Store...")
                reviews_list = scrape_reviews(app_id)

                # Make predictions
                st.write("Analyzing reviews...")
                predictions = predict_review_authenticity(reviews_list, model, tokenizer)

                # Count results
                fake_count = (predictions == 0).sum().item()
                genuine_count = (predictions == 1).sum().item()

                # Display results
                if fake_count > genuine_count:
                    st.markdown("<h3 class='fake'>The app is likely FAKE based on reviews</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 class='genuine'>The app is likely GENUINE based on reviews</h3>", unsafe_allow_html=True)

                # Display counts
                st.markdown(f"<div class='summary'>Genuine Reviews: {genuine_count}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='summary'>Fake Reviews: {fake_count}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a valid app URL.")

if __name__ == "__main__":
    streamlit_app()
