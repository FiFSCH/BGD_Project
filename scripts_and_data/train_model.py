import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, NaiveBayes
from pyspark.ml import Pipeline
from datetime import datetime

# Function to save the model
def save_model(model, model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"hdfs://localhost:9000/bgd/models/{model_type}_pipeline_{timestamp}"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

# Function to train the model
def train_model(model_type, train_data):
    # Preprocessing steps
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
    
    # Save the model
    save_model(trained_model, model_type)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and save a model on the emotion dataset.")
    parser.add_argument("model_type", choices=["lr", "rf", "dt", "nb"], help="Type of model to train (lr=Logistic Regression, rf=Random Forest, dt=Decision Tree, nb=Naive Bayes)")
    args = parser.parse_args()
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EmotionAnalysis") \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.executor.memory", "4g") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .getOrCreate()

    # Load Dataset from HDFS
    train_path = "hdfs://localhost:9000/bgd/datasets/train.csv"
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)

    # Split the training data into training and validation sets (80% train, 20% validation)
    train_data, val_data = train_df.randomSplit([0.8, 0.2], seed=42)

    # Train and save the model
    train_model(args.model_type, train_data)

if __name__ == "__main__":
    main()
