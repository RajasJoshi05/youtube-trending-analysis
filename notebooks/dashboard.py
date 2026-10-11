import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix, hstack


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "youtube_data.csv"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
TITLE_TFIDF_PATH = MODELS_DIR / "tfidf_title.pkl"
DESCRIPTION_TFIDF_PATH = MODELS_DIR / "tfidf_description.pkl"
TAGS_TFIDF_PATH = MODELS_DIR / "tfidf_tags.pkl"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="YouTube Trend Analysis",
    page_icon="▶️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 YouTube Trend Analysis & Prediction")

st.markdown(
    "### Machine Learning Dashboard for YouTube Trending Prediction"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD TRAINED MODEL AND TF-IDF VECTORIZERS
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    title_tfidf = joblib.load(TITLE_TFIDF_PATH)

    description_tfidf = joblib.load(
        DESCRIPTION_TFIDF_PATH
    )

    tags_tfidf = joblib.load(
        TAGS_TFIDF_PATH
    )

    return (
        model,
        title_tfidf,
        description_tfidf,
        tags_tfidf
    )


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    df = load_data()

    (
        model,
        title_tfidf,
        description_tfidf,
        tags_tfidf
    ) = load_models()

except Exception as e:

    st.error("❌ Error loading project files")

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Dashboard")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "📈 Dataset Analysis",
        "🤖 Trend Prediction"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 Project Overview")

    st.write(
        """
        This project uses Machine Learning and Natural Language Processing
        to analyse YouTube video data and predict whether a video is likely
        to become trending.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Videos",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Dataset Columns",
            f"{df.shape[1]:,}"
        )

    with col3:

        st.metric(
            "Dataset Size",
            f"{df.memory_usage(deep=True).sum() / (1024**2):.1f} MB"
        )

    with col4:

        st.metric(
            "ML Model",
            type(model).__name__
        )

    st.divider()

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("📌 Dataset Columns")

    st.write(
        list(df.columns)
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "📈 Dataset Analysis":

    st.header("📈 YouTube Dataset Analysis")

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    st.subheader("Dataset Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{df.isnull().sum().sum():,}"
        )

    st.divider()

    # --------------------------------------------------------
    # NUMERIC FEATURE ANALYSIS
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        st.subheader("📊 Numeric Feature Distribution")

        selected_column = st.selectbox(
            "Select a numeric feature",
            numeric_columns
        )

        st.bar_chart(
            df[selected_column].value_counts().head(20)
        )

    st.divider()

    # --------------------------------------------------------
    # FULL DATASET
    # --------------------------------------------------------

    st.subheader("📋 Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


# ============================================================
# TREND PREDICTION
# ============================================================

elif page == "🤖 Trend Prediction":

    st.header("🤖 YouTube Trend Prediction")

    st.write(
        """
        Enter information about a YouTube video and the trained
        Machine Learning model will predict whether the video
        is likely to become trending.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    title = st.text_input(
        "🎬 Video Title",
        placeholder="Enter the YouTube video title"
    )

    description = st.text_area(
        "📝 Video Description",
        placeholder="Enter the video description"
    )

    tags = st.text_input(
        "🏷️ Video Tags",
        placeholder="Enter video tags separated by spaces"
    )

    st.caption(
        "Example tags: youtube technology ai machine-learning"
    )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Trend",
        use_container_width=True
    ):

        if not title.strip():

            st.warning(
                "Please enter a video title."
            )

            st.stop()

        try:

            # ==================================================
            # STEP 1: TF-IDF FEATURES
            # ==================================================

            title_features = title_tfidf.transform(
                [title]
            )

            description_features = description_tfidf.transform(
                [description]
            )

            tags_features = tags_tfidf.transform(
                [tags]
            )


            # ==================================================
            # STEP 2: NUMERICAL FEATURES
            # ==================================================

            title_length = len(title)

            description_length = len(description)

            description_word_count = len(
                description.split()
            )

            tag_count = len(
                tags.split()
            )

            has_description = int(
                description.strip() != ""
            )

            has_tags = int(
                tags.strip() != ""
            )

            title_has_question = int(
                "?" in title
            )


            # ==================================================
            # STEP 3: CREATE NUMERICAL FEATURE MATRIX
            # ==================================================

            numeric_features = [
                [
                    title_length,
                    description_length,
                    description_word_count,
                    tag_count,
                    has_description,
                    has_tags,
                    title_has_question
                ]
            ]

            numeric_features = csr_matrix(
                numeric_features
            )


            # ==================================================
            # STEP 4: COMBINE ALL FEATURES
            # ==================================================

            combined_features = hstack(
                [
                    title_features,
                    description_features,
                    tags_features,
                    numeric_features
                ]
            ).tocsr()


            # ==================================================
            # STEP 5: VERIFY FEATURE COUNT
            # ==================================================

            expected_features = getattr(
                model,
                "n_features_in_",
                None
            )

            actual_features = combined_features.shape[1]

            if (
                expected_features is not None
                and actual_features != expected_features
            ):

                st.error(
                    f"Feature mismatch: model expects "
                    f"{expected_features} features, but dashboard "
                    f"created {actual_features} features."
                )

                st.stop()


            # ==================================================
            # STEP 6: MAKE PREDICTION
            # ==================================================

            
# Get probability of each class
            probabilities = model.predict_proba(
            combined_features
            )[0]

            not_trending_probability = probabilities[0]
            trending_probability = probabilities[1]

            # Default classification threshold
            threshold = 0.50

            prediction = int(
             trending_probability >= threshold
)

            # ==================================================
            # PREDICTION RESULT
            # ==================================================

            st.divider()

            st.subheader(
                "🎯 Prediction Result"
            )

            if prediction == 1:

                st.success(
                    "🔥 The model predicts that this video "
                    "is likely to TREND."
                )

            else:

                st.info(
                    "📉 The model predicts that this video "
                    "is NOT likely to trend."
                )

            st.metric(
         "🔥 Probability of Trending",
         f"{trending_probability * 100:.2f}%"
)

            st.metric(
         "📉 Probability of Not Trending",
             f"{not_trending_probability * 100:.2f}%"
            )

            # ------------------------------------------------
            # FEATURE INFORMATION
            # ------------------------------------------------

            with st.expander(
                "🔍 View calculated features"
            ):

                st.write(
                    {
                        "Title length": title_length,
                        "Description length": description_length,
                        "Description word count": description_word_count,
                        "Tag count": tag_count,
                        "Has description": has_description,
                        "Has tags": has_tags,
                        "Title has question": title_has_question,
                        "TF-IDF features": (
                            title_features.shape[1]
                            + description_features.shape[1]
                            + tags_features.shape[1]
                        ),
                        "Total features sent to model": actual_features
                    }
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)