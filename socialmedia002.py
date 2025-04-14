import pandas as pd
import numpy as np
from faker import Faker
import random
import string
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import logging
import requests
import json
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
plt.ion()  # Turn on interactive mode
# Initialize Faker
fake = Faker()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to generate random hashtags and mentions
def generate_hashtags_mentions(num):
    hashtags = ['#' + ''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(num)]
    mentions = ['@' + ''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(num)]
    return hashtags, mentions

# Generate synthetic dataset
def generate_synthetic_data(num_rows=50000):
    data = []
    platforms = ['Instagram', 'Twitter']
    
    for _ in range(num_rows):
        platform = random.choice(platforms)
        post_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        username = fake.user_name()
        post_content = fake.text(max_nb_chars=280)
        timestamp = fake.date_time_this_year()
        likes = random.randint(0, 10000)
        comments = random.randint(0, 1000)
        shares_retweets = random.randint(0, 5000)
        hashtags, mentions = generate_hashtags_mentions(random.randint(1, 5))
        location = fake.city() if random.random() > 0.5 else None
        
        data.append({
            'Post ID': post_id,
            'Platform': platform,
            'Username': username,
            'Post Content': post_content,
            'Timestamp': timestamp,
            'Likes': likes,
            'Comments': comments,
            'Shares/Retweets': shares_retweets,
            'Hashtags': ' '.join(hashtags),
            'Mentions': ' '.join(mentions),
            'Location': location
        })
    
    return pd.DataFrame(data)

# Predictive analytics model
def train_engagement_forecast_model(data):
    logger.info("Training engagement forecast model")
    try:
        X = data[['Likes', 'Comments', 'Shares/Retweets']].values
        y = data['Likes'] + data['Comments'] * 2 + data['Shares/Retweets'] * 3
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        logger.info("Model training completed successfully")
        return model
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

# Selenium WebDriver setup for automated testing
# Selenium WebDriver setup for automated testing
def setup_webdriver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_argument('--log-level=3')  # Only show fatal errors
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # Suppress all console output from ChromeDriver
    import os
    os.environ['WDM_LOG_LEVEL'] = '0'
    
    driver = webdriver.Chrome(options=options)
    return driver

# ... rest of the code remains the same ...

# Automated testing of data extraction
def test_data_extraction(driver, url):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        logger.info("Data extraction test passed")
    except Exception as e:
        logger.error(f"Data extraction test failed: {str(e)}")

# Automated testing of page load
def test_page_load(driver, url):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        print("Page loaded successfully")
    except Exception as e:
        print(f"Failed to load page: {e}")

# Generate the dataset
synthetic_data = generate_synthetic_data()

# Export the dataset to CSV
synthetic_data.to_csv('synthetic_social_media_data.csv', index=False)
# Export the dataset to CSV
synthetic_data.to_csv('d:\\social_media\\synthetic_social_media_data.csv', index=False)

# Train predictive analytics model
engagement_model = train_engagement_forecast_model(synthetic_data)

# Run automated tests
driver = setup_webdriver()
test_data_extraction(driver, 'https://www.uitestingplayground.com/')  # Replace with actual testing page URL
test_page_load(driver, 'https://www.uitestingplayground.com/')
driver.quit()

print("Synthetic dataset generated, predictive analytics model trained, and automated tests executed successfully!")

# Power BI REST API URL for streaming dataset
#power_bi_url = "https://api.powerbi.com/beta/YOUR_WORKSPACE_ID/datasets/YOUR_DATASET_ID/rows?key=YOUR_API_KEY"

def generate_data():
   return {
       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
       "value": random.randint(0, 100)
   }

# Limit the number of data sends to 30
#for _ in range(30):
#    data = [generate_data()]
#   response = requests.post(power_bi_url, data=json.dumps(data))
#   print(f"Data sent: {data}, Response: {response.status_code}")
#   time.sleep(5)  # Send data every 5 seconds

# Generate synthetic time series data
np.random.seed(42)
date_range = pd.date_range(start='1/1/2020', periods=100, freq='D')
data = pd.Series(np.random.randn(100).cumsum(), index=date_range)

# Fit the model
model = ExponentialSmoothing(data, trend='add', seasonal=None)
fit = model.fit()

# Forecast
forecast = fit.forecast(steps=10)
# After calculating the metrics and before the other plots


# Plot
plt.figure(figsize=(10, 6))
plt.plot(data, label='Observed')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.legend()
plt.title('Basic Trend Forecasting')
plt.savefig('plot6.png')

# Visualize the synthetic data
plt.figure(figsize=(10, 6))
plt.hist(synthetic_data['Likes'], bins=50, alpha=0.7, label='Likes')
plt.hist(synthetic_data['Comments'], bins=50, alpha=0.7, label='Comments')
plt.hist(synthetic_data['Shares/Retweets'], bins=50, alpha=0.7, label='Shares/Retweets')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Distribution of Engagement Metrics')
plt.legend()
plt.savefig('plot5.png')

# Prepare the data for prediction
X = synthetic_data[['Likes', 'Comments', 'Shares/Retweets']].values
y_true = synthetic_data['Likes'] + synthetic_data['Comments'] * 2 + synthetic_data['Shares/Retweets'] * 3

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Make predictions
y_pred = engagement_model.predict(X_test)

# Remove the metrics calculation from line 173 and move it after line 219 (after y_pred is defined)

# After this line:
y_pred = engagement_model.predict(X_test)

# Add the metrics calculation here:
# Import necessary metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# For regression metrics (if your target is continuous)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

# For classification metrics (we need to convert to binary classification)
# Let's define high engagement as above the median
y_test_binary = (y_test > y_test.median()).astype(int)
y_pred_binary = (y_pred > y_test.median()).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)

print("\nClassification Metrics (for binary engagement prediction):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test_binary, y_pred_binary)
print("\nConfusion Matrix:")
print(cm)

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary))

# Create a bar chart for classification metrics
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.ylim(0, 1.0)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Classification Metrics')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.savefig('classification_metrics.png')

# Create a confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
# Visualize the predictions

# ... existing code ...

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Engagement')
plt.ylabel('Predicted Engagement')
plt.title('Actual vs Predicted Engagement')
plt.savefig('plot7.png') 


# Additional visualizations for predictions
# 1. Residuals plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('plot3.png') 


# 2. Distribution of prediction errors
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.savefig('plot2.png') 


# 3. Feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Likes', 'Comments', 'Shares/Retweets'],
    'Importance': engagement_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.savefig('plot1.png') 

# 4. Actual vs Predicted values (line plot for a sample)
sample_size = 100
sample_indices = random.sample(range(len(y_test)), sample_size)
sample_actual = y_test[sample_indices]
sample_pred = y_pred[sample_indices]

plt.figure(figsize=(12, 6))
plt.plot(range(sample_size), sorted(sample_actual), 'b-', label='Actual')
plt.plot(range(sample_size), sorted(sample_pred), 'r-', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Engagement Value')
plt.title('Sorted Actual vs Predicted Values (Sample)')
plt.legend()
plt.savefig('plot4.png') 

  # This will block execution until all plots are closed










