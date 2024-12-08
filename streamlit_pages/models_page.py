from pyspark.sql import SparkSession
from auxilliary_methods import *
import constants


def display_models_page():
    st.title("Models")
    st.markdown(constants.MODELS_INTRO_TEXT)

    with st.expander("Evaluation Metrics"):
        st.markdown(constants.EVALUATION_METRICS)
    st.markdown(constants.LINE_SEPARATOR, unsafe_allow_html=True)

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Select the model type
    model_type = st.selectbox(
        "Choose a model to train:",
        [constants.MODEL_LR_DESC, constants.MODEL_RF_DESC, constants.MODEL_DT_DESC, constants.MODEL_NB_DESC],
        key="model_select"
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
    if st.session_state.selected_model != model_type:
        st.session_state.trained_model = None
        st.session_state.selected_model = model_type

    # Train the model when the button is clicked
    col1, col2 = st.columns([1, 2])
    with col1:
        train_button = st.button("Train Model")
    with col2:
        save_model_flag = st.toggle("Save Model")

    if train_button:
        with st.spinner("Training the model..."):
            trained_model = train_model(model_type, train_data, save_model_flag)
        st.success(f"Model trained successfully! You can now evaluate or test it.")
        st.session_state.trained_model = trained_model

    # Option to evaluate the model
    if st.session_state.trained_model:

        col3, col4 = st.columns([1, 2])
        with col3:
            validation_button = st.button("Evaluate on Validation Set")
        with col4:
            test_button = st.button("Evaluate on Test Set")

        # Evaluate on validation set
        if validation_button:
            with st.spinner("Evaluating model on Validation set..."):
                val_predictions = st.session_state.trained_model.transform(val_data)
                evaluate_model(val_predictions, model_type, "Validation")

        # Evaluate on test set
        if test_button:
            with st.spinner("Evaluating model on Test set..."):
                test_predictions = st.session_state.trained_model.transform(test_df)
                evaluate_model(test_predictions, model_type, "Test")
