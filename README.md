# YouTube Trending Analysis and Prediction

An extended machine learning project for predicting whether a YouTube video belongs to the high-view group, using video metadata and text features.

## Original Project

This project builds upon the original work by Nancy and collaborators:

- Original repository: https://github.com/nancy1404/youtube-trending-analysis

The original project is acknowledged here. This repository focuses on my local modifications and extensions.

## My Contributions

- Updated and ran the model training and testing workflow.
- Used TF-IDF features for video titles, descriptions, and tags.
- Trained and evaluated a Logistic Regression classifier.
- Saved the trained model and text vectorizers for reuse.
- Developed a Streamlit dashboard for interactive predictions.
- Evaluated performance on a held-out test dataset.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Jupyter Notebook
- Streamlit

## Model Evaluation

The Logistic Regression model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 82.09% |
| Precision | 25.09% |
| Recall | 39.77% |
| F1-score | 30.77% |

Accuracy alone does not tell the full story. The relatively low precision means many videos predicted as high-view may not actually belong to the high-view group.

## Dataset and Prediction Target

The dataset file is not included in this repository while its redistribution terms are being checked.

The prediction target represents videos in the top 10% by views in the dataset. It is **not** a prediction of YouTube's official Trending designation.

## Project Structure

- `models/` — trained classifier and TF-IDF vectorizers
- `notebooks/` — training and testing notebooks, Streamlit dashboard, and requirements
- `README.md` — project documentation

## Run the Dashboard

From the project root, activate your Python environment and install dependencies:

```powershell
pip install -r notebooks/requirements.txt
python -m streamlit run notebooks/dashboard.py
```

The dashboard expects the trained model and vectorizers in the root `models/` directory.

## Credits and Licensing

Original project: https://github.com/nancy1404/youtube-trending-analysis

Please review the original project's license and the licenses of any reused materials before redistributing them. Modifications in this repository should not be interpreted as a claim of ownership over the original work.
