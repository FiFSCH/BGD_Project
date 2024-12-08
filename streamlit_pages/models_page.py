from pyspark.sql import SparkSession
import constants
from auxilliary_methods import *

def display_models_page():
    # TODO Move all below to appropriate page
    # TODO STYLING
    st.title("TODO work in progress!")
    st.title("Models")
    st.markdown(constants.MODELS_INTRO_TEXT)

    with st.expander("Evaluation Metrics"):
        st.markdown(constants.EVALUATION_METRICS)
    st.markdown(constants.LINE_SEPARATOR, unsafe_allow_html=True)

    # Select the model type
    model_type = st.selectbox(
        "Choose a model to train:",
        [constants.MODEL_LR_DESC, constants.MODEL_RF_DESC, constants.MODEL_DT_DESC, constants.MODEL_NB_DESC]
    )

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
    st.write('DONT USE THIS IF YOU DIDNT TRAIN THE MODEL FIRST. IT WILL THROW AN EXCEPTION.\n TODO FIX ')
    if st.button("Evaluate on Test set"):
        with st.spinner("Evaluating model on Test set..."):
            test_predictions = st.session_state.trained_model.transform(test_df)
            evaluate_model(test_predictions, model_type, "Test")