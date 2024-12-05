import streamlit as st
import subprocess
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# Function to save model
def save_model(model, model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"hdfs://localhost:9000/bgd/models/{model_type}_pipeline_{timestamp}"
    model.save(model_save_path)
    st.write(f"Model saved to: {model_save_path}")

# Function to evaluate model
def evaluate_model(predictions, model_type, dataset_name="Validation"):
    # Calculate evaluation metrics (accuracy, f1, weightedPrecision, weightedRecall)
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)

    # Use weighted precision instead of precision
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="weightedPrecision")
    precision = evaluator_precision.evaluate(predictions)

    # Use weighted recall instead of recall
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="weightedRecall")
    recall = evaluator_recall.evaluate(predictions)

    # Prepare metrics in a dictionary to convert to DataFrame
    metrics = {
        "Metric": ["Accuracy", "F1 Score", "Weighted Precision", "Weighted Recall"],
        "Value": [accuracy, f1_score, precision, recall]
    }

    # Convert the dictionary to a Pandas DataFrame
    df_metrics = pd.DataFrame(metrics)

    # Format the values to 4 decimal places
    df_metrics["Value"] = df_metrics["Value"].apply(lambda x: f"{x:.4f}")

    # Display the metrics in a table format in Streamlit
    st.write(f"**{dataset_name} Evaluation Metrics**")
    st.table(df_metrics)

    # Plot confusion matrix
    y_true = predictions.select("label_index").toPandas()
    y_pred = predictions.select("prediction").toPandas()
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f"Confusion Matrix for {model_type} ({dataset_name} set)")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

# Function to train the model
def train_model(model_type, train_data, val_data, test_df):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=20000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index")

    # Select model based on argument
    if model_type == "lr":
        model = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)
    elif model_type == "rf":
        model = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=20)
    elif model_type == "dt":
        model = DecisionTreeClassifier(featuresCol="features", labelCol="label_index")
    elif model_type == "nb":
        model = NaiveBayes(featuresCol="features", labelCol="label_index", modelType="multinomial")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Build pipeline
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_indexer, model])
    
    # Train the model
    trained_model = pipeline.fit(train_data)
    
    # Save the model with timestamped name
    save_model(trained_model, model_type)

    return trained_model

# Streamlit app layout
def main():
    st.title("Emotion Analysis with PySpark")
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
