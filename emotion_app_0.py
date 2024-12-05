import streamlit as st

import constants
import streamlit_pages.home_page
import streamlit_pages.eda_page
from pyspark.sql import SparkSession
from streamlit_option_menu import option_menu
from auxilliary_methods import *

# Streamlit app layout
def main():

    with st.sidebar:
        selected = option_menu( # TODO: Change labels and icons
            menu_title='Navigation',
            options=[constants.HOME_PAGE_NAME, constants.EDA_PAGE_NAME, constants.MODELS_PAGE_NAME],
            icons=['house', 'graph-up-arrow', 'diagram-3'],
            menu_icon='map',
            default_index=0
        )

    if selected == constants.HOME_PAGE_NAME:
        streamlit_pages.home_page.display_home_page()

    if selected == constants.EDA_PAGE_NAME:
        streamlit_pages.eda_page.display_eda_page()

    if selected == constants.MODELS_PAGE_NAME:
        # TODO Move all below to appropriate page
        # TODO STYLING

        st.write("Select a model and start the training or testing process.")
    
        # Select the model type
        model_type = st.selectbox(
            "Choose a model to train:",
            ["lr", "rf", "dt", "nb"]
         )

        model_descriptions = {
            "lr": "Logistic Regression (Binary and Multiclass classification)",
            "rf": "Random Forest (Ensemble model for classification)",
            "dt": "Decision Tree (Classification using tree-based models)",
            "nb": "Naive Bayes (Probabilistic classification)"
        }

        st.write(f"**Selected Model:** {model_descriptions[model_type]}")

        # Initialize Spark session
        spark = SparkSession.builder \
                         .appName("EmotionAnalysis") \
                         .config("spark.memory.fraction", 0.8) \
                         .config("spark.executor.memory", "4g") \
                         .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
                         .getOrCreate()

        # Load Dataset from hdfs
        train_path = "hdfs://localhost:9000/bgd/datasets/train.csv"
        test_path = "hdfs://localhost:9000/bgd/datasets/test.csv"

        train_df = spark.read.csv(train_path, header=True, inferSchema=True)
        test_df = spark.read.csv(test_path, header=True, inferSchema=True)

        # Split the training data into training and validation sets (80% train, 20% validation)
        train_data, val_data = train_df.randomSplit([0.8, 0.2], seed=42)

     # Session state to track model training status
        if "trained_model" not in st.session_state:
         st.session_state.trained_model = None

     # Train the model when the button is clicked
        if st.button("Train Model"):
         with st.spinner("Training the model..."):
              st.session_state.trained_model = train_model(model_type, train_data, val_data, test_df)
              st.success(f"Model trained successfully! You can now evaluate or test it.")

     # Option to evaluate the model
        if st.session_state.trained_model:
          # Evaluate on validation set
         if st.button("Evaluate (Validation)"):
               with st.spinner("Evaluating model on Validation set..."):
                   val_predictions = st.session_state.trained_model.transform(val_data)
                   evaluate_model(val_predictions, model_type, "Validation")

            # Evaluate on test set
        if st.button("Evaluate on Test set"):
                    with st.spinner("Evaluating model on Test set..."):
                        test_predictions = st.session_state.trained_model.transform(test_df)
                        evaluate_model(test_predictions, model_type, "Test")

if __name__ == "__main__":
    main()
