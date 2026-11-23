# Social Media Impact Predictor

A Streamlit web application powered by a Random Forest Classifier to predict the overall impact of social media on a student's life.

## Repository Files
- `app.py`: Main Streamlit application interface and prediction pipeline.
- `requirements.txt`: Python dependencies pinned to `scikit-learn==1.6.1`.
- `social_media_impact_rf_model.pkl`: Trained Random Forest classification model.
- `label_encoders.pkl`: Saved `LabelEncoder` mappings for categorical features.

## Deployment Steps
1. Upload `app.py`, `requirements.txt`, `social_media_impact_rf_model.pkl`, and `label_encoders.pkl` to the root of your GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **New app**, select your repository and branch, set `app.py` as the main file path, and click **Deploy!**.
