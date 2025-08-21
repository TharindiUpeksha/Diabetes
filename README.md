# Diabetes Prediction Web Application

This project is a machine learning-based web application that predicts the likelihood of diabetes based on various health measurements. The application uses the PIMA Indians Diabetes dataset and provides an interactive interface for users to input their health metrics and receive predictions.

## Dataset Description

The PIMA Indians Diabetes dataset contains the following features:
- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age
- Outcome (0: No Diabetes, 1: Diabetes)

The dataset contains approximately 768 rows and 8 features, with a binary target variable indicating the presence of diabetes.

## Project Structure

```
Diabetes/
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
├── model.pkl          # Trained machine learning model
├── data/
│   └── diabetes.csv   # Dataset
├── notebooks/
│   └── model_training.ipynb  # Model development notebook
└── README.md          # Project documentation
```

## Features

1. **Data Exploration**
   - View dataset preview
   - Display basic statistics
   - Filter data by age range

2. **Data Visualization**
   - Glucose level distribution
   - Feature correlation heatmap
   - Outcome distribution

3. **Model Prediction**
   - Interactive input form for health metrics
   - Real-time predictions
   - Prediction probability display

4. **Model Performance**
   - View model metrics
   - Performance visualization
   - Model information

## Installation and Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset:
   - Go to Kaggle and search for "Diabetes Dataset"
   - Download diabetes.csv and place it in the data/ folder

3. Run the Jupyter notebook to train the model:
   - Open notebooks/model_training.ipynb
   - Run all cells to train and save the model

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Navigate to the different sections using the sidebar menu
2. Explore the dataset and visualizations
3. Enter patient information in the Model Prediction section
4. Click "Predict" to get the diabetes prediction result

## Model Training

The model was developed using:
- Data preprocessing (handling missing values and scaling)
- Feature engineering
- Multiple algorithms (Logistic Regression, Random Forest)
- Cross-validation for model evaluation
- Metrics comparison (Accuracy, Precision, Recall, F1-score)

## License

This project is licensed under the MIT License.
