# This file contains most of the long strings.
# It is meant to declutter the main streamlit file

HOME_PAGE_NAME = 'Home'
EDA_PAGE_NAME = 'EDA'
MODELS_PAGE_NAME = 'Models'

MODEL_LR_DESC = "Logistic Regression (Binary and Multiclass classification)"
MODEL_RF_DESC = "Random Forest (Ensemble model for classification)"
MODEL_DT_DESC = "Decision Tree (Classification using tree-based models)"
MODEL_NB_DESC = "Naive Bayes (Probabilistic classification)"

MODEL_LR = "lr"
MODEL_RF = "rf"
MODEL_DT = "dt"
MODEL_NB = "nb"

IMAGES_PATH = './images'
LINE_SEPARATOR = '''<hr style="height:4px;border:none;color:#7C7C7C;background-color:#7C7C7C;" />'''

PROJECT_GOAL = '''The goal of the project is to explore the Emotion dataset, a collection of English-language Twitter messages categorized into six basic emotions: 
* Anger 
* Fear
* Joy
* Love
* Sadness
* Surprise

The undertaken steps include tasks such as: __emotion classification, sentiment analysis, and text-based emotion visualization.__'''

AUTHORS_LIST = '''
* Hien Anh Nguyen, s22192  
* Filip Schulz, s22455
'''

LIBRARIES_USED = '''
  * Python
  * Hadoop
  * PySpark
  * Seaborn
  * MatPlotLib
  * WordCloud
  * SciKit-Learn
  * Pandas
  * Numpy
  * Streamlit + streamlit_option_menu
'''

DATA_SET_DESCTIPTION = '''The dataset contains 18 000 rows divided into 2 columns: __text__ and __label__. 
* __Text:__ Twitter message
* __Label:__ Emotion

The data did not require any preprocessing.'''

MODELS_INTRO_TEXT = '''
This page shows the Spark machine learning modeling and evaluation process. \\
You can select the desired model, train it and then evaluate using validation and testing sets.
'''

EVALUATION_METRICS = '''
* Accuracy
* F1-Score
* Weighted Precision
* Weighted Recall
'''