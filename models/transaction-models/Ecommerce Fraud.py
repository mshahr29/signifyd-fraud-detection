# Base Model: Ecommerce Transactions Fraud Detection

# Logistic Regression (Simple)

## EDA
"""

#Importing libraries and the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the datasets
user_info = pd.read_csv("Fraud_Data.csv")         # Users information
ip_country_mapping = pd.read_csv("IpAddress_to_Country.csv")  # Country from IP information

ip_country_mapping.head()

ip_country_mapping.info()

user_info.head()

user_info.info()

"""**Mapping/Merging fraud data with country via IP address**"""

ip_country_mapping.upper_bound_ip_address.astype("float")
ip_country_mapping.lower_bound_ip_address.astype("float")
user_info.ip_address.astype("float")

def IP_to_country(ip) :
    try :
        return ip_country_mapping.country[(ip_country_mapping.lower_bound_ip_address < ip)
                                &
                                (ip_country_mapping.upper_bound_ip_address > ip)].iloc[0]
    except IndexError :
        return "Unknown"

import os

# Define the directory path within Colab's file storage
directory = "/content/datasets_fraud"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# country to each IP
user_info["IP_country"] = user_info.ip_address.apply(IP_to_country)

# saving
user_info.to_csv("/content/datasets_fraud/Fraud_data_with_country.csv", index=False)

# loading
user_info= pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

user_info.head()

# Print summary statistics
print(user_info[["purchase_value", "age"]].describe())
print('*'*50)
# Print unique values and their frequencies
for column in ["source", "browser", "sex"]:
    print(user_info[column].value_counts())
    print('*'*50)

# Check for duplicates in the "user_id" column in user_info DataFrame
print("The user_id column includes {} duplicates".format(user_info.duplicated(subset="user_id", keep=False).sum()))

# Calculate duplicate rate based on unique device_id
dup_table = pd.DataFrame(user_info.duplicated(subset="device_id"))
dup_rate = dup_table.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate * 1000) / 10))

# Calculate duplicate rate based on device_id with keep=False
dup_table2 = pd.DataFrame(user_info.duplicated(subset="device_id", keep=False))
dup_rate2 = dup_table2.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate2 * 1000) / 10))

"""The code calculates two duplicate rates based on the device_id column.
The first rate considers only the subsequent occurrences of a duplicate device_id,
while the second considers all occurrences (first and subsequent).
This provides two different perspectives on the extent of device reuse in the dataset.
"""

device_duplicates = pd.DataFrame(user_info.groupby(by="device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace=True)
dupli = device_duplicates[device_duplicates.freq_device >1]
dupli

# Reading the Dataset
user_info = pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

device_duplicates = pd.DataFrame(user_info.groupby(by = "device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace= True)

dupli = device_duplicates[device_duplicates.freq_device >1]
print("On average, when a device is used more than once it is used {mean} times, and the most used machine was used {maxi} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

dupli = device_duplicates[device_duplicates.freq_device >2]
print("On average, when a device is used more than twice it is used {mean} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

# Merge the device_duplicates with user_info
user_info = user_info.merge(device_duplicates, on="device_id")

# Calculate the proportion of fraud in the dataset
fraud_proportion = user_info["class"].mean() * 100
print("Proportion of fraud in the dataset: {:.1f}%".format(fraud_proportion))

user_info.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot device frequency distribution for values less than 4
g1 = sns.distplot(user_info.freq_device[user_info.freq_device < 4], ax=ax[0])
g1.set(xticks=[1, 2, 3])

# Plot device frequency distribution for values greater than 2
g2 = sns.distplot(user_info.freq_device[user_info.freq_device > 2], ax=ax[1])
g2.set(xticks=range(0, 21, 2))

# Display the plots
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(18, 6))

# Create subplots for bar plots
plt.subplot(1, 3, 1)
sns.barplot(x='source', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Source')

plt.subplot(1, 3, 2)
sns.barplot(x='browser', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Browser')

plt.subplot(1, 3, 3)
sns.barplot(x='sex', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Sex')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
f2, ax2 = plt.subplots(3, 1, figsize=(24, 18))

# Plot purchase_value vs. class
sns.pointplot(x="purchase_value", y="class", data=user_info, ci=None, ax=ax2[0])
ax2[0].set_title("Purchase Value vs. Fraud Probability")

# Plot age vs. class
sns.pointplot(x="age", y="class", data=user_info, ci=None, ax=ax2[1])
ax2[1].set_title("Age vs. Fraud Probability")

# Plot freq_device vs. class
sns.pointplot(x="freq_device", y="class", data=user_info, ci=None, ax=ax2[2])
ax2[2].set_title("Frequency of Device Usage vs. Fraud Probability")

# Show the plots
plt.tight_layout()
plt.show()

user_info.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and axis
f3, ax3 = plt.subplots(1, 1, figsize=(24, 18))

# Plot a stacked bar plot for IP_country vs. class
sns.barplot(x="IP_country", y="class", data=user_info[:10], estimator=sum, ci=None, ax=ax3)

# Show the plot
plt.show()

# Filter IP_country value counts where count is greater than 1000
filtered_counts = user_info.IP_country.value_counts()[user_info.IP_country.value_counts() > 1000]

# Plot the filtered counts as a bar plot
filtered_counts.plot(kind="bar")
plt.xlabel("IP Country")
plt.ylabel("Frequency")
plt.title("IP Country Frequency (Counts > 1000)")
plt.show()

user_info.signup_time

"""## Feature Engineering"""

# --- 1 ---
# Categorisation column freq_device
# We see a clear correlation between freq_device and fraudulent activities. We are going to split freq_device into 7 categories
user_info.freq_device = user_info.freq_device.apply(lambda x:
                                                    str(x) if x < 5 else
                                                    "5-10" if x >= 5 and x <= 10 else
                                                    "11-15" if x > 10 and x <= 15 else
                                                    "> 15")

# Convert signup_time and purchase_time to datetime
user_info.signup_time = pd.to_datetime(user_info.signup_time, format='%Y-%m-%d %H:%M:%S')
user_info.purchase_time = pd.to_datetime(user_info.purchase_time, format='%Y-%m-%d %H:%M:%S')

# --- 2 ---
# Column month
user_info["month_purchase"] = user_info.purchase_time.apply(lambda x: calendar.month_name[x.month])

# --- 3 ---
# Column week
user_info["weekday_purchase"] = user_info.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])

# --- 4 ---
# Column hour_of_the_day
user_info["hour_of_the_day"] = user_info.purchase_time.apply(lambda x: x.hour)

# --- 5 ---
# Column seconds_since_signup
user_info["seconds_since_signup"] = (user_info.purchase_time - user_info.signup_time).apply(lambda x: x.total_seconds())

# --- 6 ---
# Column countries_from_device (ie. number of different countries per device_id)
# We flag devices that committed purchases from different countries
country_count = user_info.groupby(by=["device_id", "IP_country"]).count().reset_index()
country_count = pd.DataFrame(country_count.groupby(by="device_id").count().IP_country)
user_info = user_info.merge(country_count, left_on="device_id", right_index=True)
user_info.rename(columns={"IP_country_x": "IP_country", "IP_country_y": "countries_from_device"}, inplace=True)

# Column "quick_purchase" : categorise time between sign_up and purchase
user_info["quick_purchase"] = user_info.seconds_since_signup.apply(lambda x: 1 if x < 30 else 0)

# age categorisation
user_info["age_category"] = user_info.age.apply(lambda x:
                                                "< 40" if x < 40 else
                                                "40 - 49" if x < 50 else
                                                "50 -59" if x < 60 else
                                                "60 - 69" if x < 70 else
                                                " > 70")

# Hour of the day categorisation
user_info["period_of_the_day"] = user_info.hour_of_the_day.apply(lambda x:
                                                                 "late night" if x < 4 else
                                                                 "early morning" if x < 8 else
                                                                 "morning" if x < 12 else
                                                                 "afternoon" if x < 16 else
                                                                 "evening" if x < 20 else
                                                                 "early night"
                                                                 )

user_info.head()

user_info.info()

"""## Logistic Regression"""

# Count missing values for each column
missing_counts = user_info.isnull().sum()

# Print the missing value counts
missing_counts

# Drop rows with any missing values.
user_info.dropna(inplace=True)

user_info.info()

# Specify the columns to drop.
columns_to_drop = ["user_id", "signup_time", "purchase_time", "device_id", "ip_address", "hour_of_the_day", "seconds_since_signup", "age"]

# Drop specified columns.
features = user_info.drop(columns=columns_to_drop)

# Display the updated dataframe.
print(features.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

target = features["class"]
features = features.drop(columns=["class"])

features.info()

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=42,
                                                    stratify=target,
                                                    test_size=0.25)

import pandas as pd
# Identify categorical and numerical features
categorical_cols = features.select_dtypes(include=['object', 'category']).columns
numerical_cols = features.select_dtypes(include=['number']).columns

# One-hot encode categorical features
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat = ohe.transform(X_test[categorical_cols])

# Create dataframes from the encoded arrays
X_train_cat_df = pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(categorical_cols))
X_test_cat_df = pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(categorical_cols))

# Standard scale numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

# Create dataframes from the scaled arrays
X_train_num_df = pd.DataFrame(X_train_num, columns=numerical_cols)
X_test_num_df = pd.DataFrame(X_test_num, columns=numerical_cols)

# Concatenate the encoded categorical and scaled numerical features
X_train_processed = pd.concat([X_train_cat_df, X_train_num_df], axis=1)
X_test_processed = pd.concat([X_test_cat_df, X_test_num_df], axis=1)

# Fitting a logistic regression model
logistic_regression = LogisticRegression(solver='liblinear', random_state=42)
logistic_regression.fit(X_train_processed, y_train)

# Printing scores
train_score = logistic_regression.score(X_train_processed, y_train)
test_score = logistic_regression.score(X_test_processed, y_test)
print("Train Score:", round(train_score * 100, 2), "%")
print("Test Score:", round(test_score * 100, 2), "%")

# Predict probabilities on the test set
y_pred_prob = logistic_regression.predict_proba(X_test_processed)[:, 1]

# Set a custom threshold
custom_threshold = 0.5  # Example threshold, adjust as needed

# Convert probabilities to binary predictions based on the threshold
y_pred = (y_pred_prob >= custom_threshold).astype(int)

# Classification Report
print(classification_report(y_test, y_pred))

# Other metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# You can visualize the confusion matrix using seaborn's heatmap
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Logistic Regression (SMOTE)

## EDA
"""

#Importing libraries and the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the datasets
user_info = pd.read_csv("Fraud_Data.csv")         # Users information
ip_country_mapping = pd.read_csv("IpAddress_to_Country.csv")  # Country from IP information

ip_country_mapping.head()

ip_country_mapping.info()

user_info.head()

user_info.info()

"""**Mapping/Merging fraud data with country via IP address**"""

ip_country_mapping.upper_bound_ip_address.astype("float")
ip_country_mapping.lower_bound_ip_address.astype("float")
user_info.ip_address.astype("float")

def IP_to_country(ip) :
    try :
        return ip_country_mapping.country[(ip_country_mapping.lower_bound_ip_address < ip)
                                &
                                (ip_country_mapping.upper_bound_ip_address > ip)].iloc[0]
    except IndexError :
        return "Unknown"

import os

# Define the directory path within Colab's file storage
directory = "/content/datasets_fraud"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# country to each IP
user_info["IP_country"] = user_info.ip_address.apply(IP_to_country)

# saving
user_info.to_csv("/content/datasets_fraud/Fraud_data_with_country.csv", index=False)

# loading
user_info= pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

user_info.head()

# Print summary statistics
print(user_info[["purchase_value", "age"]].describe())
print('*'*50)
# Print unique values and their frequencies
for column in ["source", "browser", "sex"]:
    print(user_info[column].value_counts())
    print('*'*50)

# Check for duplicates in the "user_id" column in user_info DataFrame
print("The user_id column includes {} duplicates".format(user_info.duplicated(subset="user_id", keep=False).sum()))

# Calculate duplicate rate based on unique device_id
dup_table = pd.DataFrame(user_info.duplicated(subset="device_id"))
dup_rate = dup_table.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate * 1000) / 10))

# Calculate duplicate rate based on device_id with keep=False
dup_table2 = pd.DataFrame(user_info.duplicated(subset="device_id", keep=False))
dup_rate2 = dup_table2.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate2 * 1000) / 10))

"""The code calculates two duplicate rates based on the device_id column.
The first rate considers only the subsequent occurrences of a duplicate device_id,
while the second considers all occurrences (first and subsequent).
This provides two different perspectives on the extent of device reuse in the dataset.
"""

device_duplicates = pd.DataFrame(user_info.groupby(by="device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace=True)
dupli = device_duplicates[device_duplicates.freq_device >1]
dupli

# Reading the Dataset
user_info = pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

device_duplicates = pd.DataFrame(user_info.groupby(by = "device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace= True)

dupli = device_duplicates[device_duplicates.freq_device >1]
print("On average, when a device is used more than once it is used {mean} times, and the most used machine was used {maxi} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

dupli = device_duplicates[device_duplicates.freq_device >2]
print("On average, when a device is used more than twice it is used {mean} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

# Merge the device_duplicates with user_info
user_info = user_info.merge(device_duplicates, on="device_id")

# Calculate the proportion of fraud in the dataset
fraud_proportion = user_info["class"].mean() * 100
print("Proportion of fraud in the dataset: {:.1f}%".format(fraud_proportion))

user_info.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot device frequency distribution for values less than 4
g1 = sns.distplot(user_info.freq_device[user_info.freq_device < 4], ax=ax[0])
g1.set(xticks=[1, 2, 3])

# Plot device frequency distribution for values greater than 2
g2 = sns.distplot(user_info.freq_device[user_info.freq_device > 2], ax=ax[1])
g2.set(xticks=range(0, 21, 2))

# Display the plots
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(18, 6))

# Create subplots for bar plots
plt.subplot(1, 3, 1)
sns.barplot(x='source', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Source')

plt.subplot(1, 3, 2)
sns.barplot(x='browser', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Browser')

plt.subplot(1, 3, 3)
sns.barplot(x='sex', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Sex')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
f2, ax2 = plt.subplots(3, 1, figsize=(24, 18))

# Plot purchase_value vs. class
sns.pointplot(x="purchase_value", y="class", data=user_info, ci=None, ax=ax2[0])
ax2[0].set_title("Purchase Value vs. Fraud Probability")

# Plot age vs. class
sns.pointplot(x="age", y="class", data=user_info, ci=None, ax=ax2[1])
ax2[1].set_title("Age vs. Fraud Probability")

# Plot freq_device vs. class
sns.pointplot(x="freq_device", y="class", data=user_info, ci=None, ax=ax2[2])
ax2[2].set_title("Frequency of Device Usage vs. Fraud Probability")

# Show the plots
plt.tight_layout()
plt.show()

user_info.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and axis
f3, ax3 = plt.subplots(1, 1, figsize=(24, 18))

# Plot a stacked bar plot for IP_country vs. class
sns.barplot(x="IP_country", y="class", data=user_info[:10], estimator=sum, ci=None, ax=ax3)

# Show the plot
plt.show()

# Filter IP_country value counts where count is greater than 1000
filtered_counts = user_info.IP_country.value_counts()[user_info.IP_country.value_counts() > 1000]

# Plot the filtered counts as a bar plot
filtered_counts.plot(kind="bar")
plt.xlabel("IP Country")
plt.ylabel("Frequency")
plt.title("IP Country Frequency (Counts > 1000)")
plt.show()

user_info.signup_time

"""## Feature Engineering"""

# --- 1 ---
# Categorisation column freq_device
# We see a clear correlation between freq_device and fraudulent activities. We are going to split freq_device into 7 categories
user_info.freq_device = user_info.freq_device.apply(lambda x:
                                                    str(x) if x < 5 else
                                                    "5-10" if x >= 5 and x <= 10 else
                                                    "11-15" if x > 10 and x <= 15 else
                                                    "> 15")

# Convert signup_time and purchase_time to datetime
user_info.signup_time = pd.to_datetime(user_info.signup_time, format='%Y-%m-%d %H:%M:%S')
user_info.purchase_time = pd.to_datetime(user_info.purchase_time, format='%Y-%m-%d %H:%M:%S')

# --- 2 ---
# Column month
user_info["month_purchase"] = user_info.purchase_time.apply(lambda x: calendar.month_name[x.month])

# --- 3 ---
# Column week
user_info["weekday_purchase"] = user_info.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])

# --- 4 ---
# Column hour_of_the_day
user_info["hour_of_the_day"] = user_info.purchase_time.apply(lambda x: x.hour)

# --- 5 ---
# Column seconds_since_signup
user_info["seconds_since_signup"] = (user_info.purchase_time - user_info.signup_time).apply(lambda x: x.total_seconds())

# --- 6 ---
# Column countries_from_device (ie. number of different countries per device_id)
# We flag devices that committed purchases from different countries
country_count = user_info.groupby(by=["device_id", "IP_country"]).count().reset_index()
country_count = pd.DataFrame(country_count.groupby(by="device_id").count().IP_country)
user_info = user_info.merge(country_count, left_on="device_id", right_index=True)
user_info.rename(columns={"IP_country_x": "IP_country", "IP_country_y": "countries_from_device"}, inplace=True)

# Column "quick_purchase" : categorise time between sign_up and purchase
user_info["quick_purchase"] = user_info.seconds_since_signup.apply(lambda x: 1 if x < 30 else 0)

# age categorisation
user_info["age_category"] = user_info.age.apply(lambda x:
                                                "< 40" if x < 40 else
                                                "40 - 49" if x < 50 else
                                                "50 -59" if x < 60 else
                                                "60 - 69" if x < 70 else
                                                " > 70")

# Hour of the day categorisation
user_info["period_of_the_day"] = user_info.hour_of_the_day.apply(lambda x:
                                                                 "late night" if x < 4 else
                                                                 "early morning" if x < 8 else
                                                                 "morning" if x < 12 else
                                                                 "afternoon" if x < 16 else
                                                                 "evening" if x < 20 else
                                                                 "early night"
                                                                 )

user_info.head()

user_info.info()

"""## Logistic Regression"""

# Count missing values for each column
missing_counts = user_info.isnull().sum()

# Print the missing value counts
missing_counts

# Drop rows with any missing values.
user_info.dropna(inplace=True)

user_info.info()

# Specify the columns to drop.
columns_to_drop = ["user_id", "signup_time", "purchase_time", "device_id", "ip_address", "hour_of_the_day", "seconds_since_signup", "age"]

# Drop specified columns.
features = user_info.drop(columns=columns_to_drop)

# Display the updated dataframe.
print(features.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

target = features["class"]
features = features.drop(columns=["class"])

features.info()

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=42,
                                                    stratify=target,
                                                    test_size=0.25)

import pandas as pd
# Identify categorical and numerical features
categorical_cols = features.select_dtypes(include=['object', 'category']).columns
numerical_cols = features.select_dtypes(include=['number']).columns

# One-hot encode categorical features
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat = ohe.transform(X_test[categorical_cols])

# Create dataframes from the encoded arrays
X_train_cat_df = pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(categorical_cols))
X_test_cat_df = pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(categorical_cols))

# Standard scale numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

# Create dataframes from the scaled arrays
X_train_num_df = pd.DataFrame(X_train_num, columns=numerical_cols)
X_test_num_df = pd.DataFrame(X_test_num, columns=numerical_cols)

# Concatenate the encoded categorical and scaled numerical features
X_train_processed = pd.concat([X_train_cat_df, X_train_num_df], axis=1)
X_test_processed = pd.concat([X_test_cat_df, X_test_num_df], axis=1)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Fitting a logistic regression model
logistic_regression = LogisticRegression(solver='liblinear', random_state=42)
logistic_regression.fit(X_train_resampled, y_train_resampled)

# Printing scores
train_score = logistic_regression.score(X_train_resampled, y_train_resampled)
test_score = logistic_regression.score(X_test_processed, y_test)
print("Train Score:", round(train_score * 100, 2), "%")
print("Test Score:", round(test_score * 100, 2), "%")

# Predict probabilities on the test set
y_pred_prob = logistic_regression.predict_proba(X_test_processed)[:, 1]

# Set a custom threshold
custom_threshold = 0.5  # Example threshold, adjust as needed

# Convert probabilities to binary predictions based on the threshold
y_pred = (y_pred_prob >= custom_threshold).astype(int)

# Classification Report
print(classification_report(y_test, y_pred))

# Other metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
AUC_ROC= roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC_ROC: {AUC_ROC}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# You can visualize the confusion matrix using seaborn's heatmap
sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# XGBoost

## EDA
"""

#Importing libraries and the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the datasets
user_info = pd.read_csv("Fraud_Data.csv")         # Users information
ip_country_mapping = pd.read_csv("IpAddress_to_Country.csv")  # Country from IP information

ip_country_mapping.head()

ip_country_mapping.info()

user_info.head()

user_info.info()

"""**Mapping/Merging fraud data with country via IP address**"""

ip_country_mapping.upper_bound_ip_address.astype("float")
ip_country_mapping.lower_bound_ip_address.astype("float")
user_info.ip_address.astype("float")

def IP_to_country(ip) :
    try :
        return ip_country_mapping.country[(ip_country_mapping.lower_bound_ip_address < ip)
                                &
                                (ip_country_mapping.upper_bound_ip_address > ip)].iloc[0]
    except IndexError :
        return "Unknown"

import os

# Define the directory path within Colab's file storage
directory = "/content/datasets_fraud"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# country to each IP
user_info["IP_country"] = user_info.ip_address.apply(IP_to_country)

# saving
user_info.to_csv("/content/datasets_fraud/Fraud_data_with_country.csv", index=False)

# loading
user_info= pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

user_info.head()

# Print summary statistics
print(user_info[["purchase_value", "age"]].describe())
print('*'*50)
# Print unique values and their frequencies
for column in ["source", "browser", "sex"]:
    print(user_info[column].value_counts())
    print('*'*50)

# Check for duplicates in the "user_id" column in user_info DataFrame
print("The user_id column includes {} duplicates".format(user_info.duplicated(subset="user_id", keep=False).sum()))

# Calculate duplicate rate based on unique device_id
dup_table = pd.DataFrame(user_info.duplicated(subset="device_id"))
dup_rate = dup_table.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate * 1000) / 10))

# Calculate duplicate rate based on device_id with keep=False
dup_table2 = pd.DataFrame(user_info.duplicated(subset="device_id", keep=False))
dup_rate2 = dup_table2.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate2 * 1000) / 10))

"""The code calculates two duplicate rates based on the device_id column.
The first rate considers only the subsequent occurrences of a duplicate device_id,
while the second considers all occurrences (first and subsequent).
This provides two different perspectives on the extent of device reuse in the dataset.
"""

device_duplicates = pd.DataFrame(user_info.groupby(by="device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace=True)
dupli = device_duplicates[device_duplicates.freq_device >1]
dupli

# Reading the Dataset
user_info = pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

device_duplicates = pd.DataFrame(user_info.groupby(by = "device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace= True)

dupli = device_duplicates[device_duplicates.freq_device >1]
print("On average, when a device is used more than once it is used {mean} times, and the most used machine was used {maxi} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

dupli = device_duplicates[device_duplicates.freq_device >2]
print("On average, when a device is used more than twice it is used {mean} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

# Merge the device_duplicates with user_info
user_info = user_info.merge(device_duplicates, on="device_id")

# Calculate the proportion of fraud in the dataset
fraud_proportion = user_info["class"].mean() * 100
print("Proportion of fraud in the dataset: {:.1f}%".format(fraud_proportion))

user_info.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot device frequency distribution for values less than 4
g1 = sns.distplot(user_info.freq_device[user_info.freq_device < 4], ax=ax[0])
g1.set(xticks=[1, 2, 3])

# Plot device frequency distribution for values greater than 2
g2 = sns.distplot(user_info.freq_device[user_info.freq_device > 2], ax=ax[1])
g2.set(xticks=range(0, 21, 2))

# Display the plots
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(18, 6))

# Create subplots for bar plots
plt.subplot(1, 3, 1)
sns.barplot(x='source', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Source')

plt.subplot(1, 3, 2)
sns.barplot(x='browser', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Browser')

plt.subplot(1, 3, 3)
sns.barplot(x='sex', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Sex')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
f2, ax2 = plt.subplots(3, 1, figsize=(24, 18))

# Plot purchase_value vs. class
sns.pointplot(x="purchase_value", y="class", data=user_info, ci=None, ax=ax2[0])
ax2[0].set_title("Purchase Value vs. Fraud Probability")

# Plot age vs. class
sns.pointplot(x="age", y="class", data=user_info, ci=None, ax=ax2[1])
ax2[1].set_title("Age vs. Fraud Probability")

# Plot freq_device vs. class
sns.pointplot(x="freq_device", y="class", data=user_info, ci=None, ax=ax2[2])
ax2[2].set_title("Frequency of Device Usage vs. Fraud Probability")

# Show the plots
plt.tight_layout()
plt.show()

user_info.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and axis
f3, ax3 = plt.subplots(1, 1, figsize=(24, 18))

# Plot a stacked bar plot for IP_country vs. class
sns.barplot(x="IP_country", y="class", data=user_info[:10], estimator=sum, ci=None, ax=ax3)

# Show the plot
plt.show()

# Filter IP_country value counts where count is greater than 1000
filtered_counts = user_info.IP_country.value_counts()[user_info.IP_country.value_counts() > 1000]

# Plot the filtered counts as a bar plot
filtered_counts.plot(kind="bar")
plt.xlabel("IP Country")
plt.ylabel("Frequency")
plt.title("IP Country Frequency (Counts > 1000)")
plt.show()

user_info.signup_time

"""## Feature Engineering"""

# --- 1 ---
# Categorisation column freq_device
# We see a clear correlation between freq_device and fraudulent activities. We are going to split freq_device into 7 categories
user_info.freq_device = user_info.freq_device.apply(lambda x:
                                                    str(x) if x < 5 else
                                                    "5-10" if x >= 5 and x <= 10 else
                                                    "11-15" if x > 10 and x <= 15 else
                                                    "> 15")

# Convert signup_time and purchase_time to datetime
user_info.signup_time = pd.to_datetime(user_info.signup_time, format='%Y-%m-%d %H:%M:%S')
user_info.purchase_time = pd.to_datetime(user_info.purchase_time, format='%Y-%m-%d %H:%M:%S')

# --- 2 ---
# Column month
user_info["month_purchase"] = user_info.purchase_time.apply(lambda x: calendar.month_name[x.month])

# --- 3 ---
# Column week
user_info["weekday_purchase"] = user_info.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])

# --- 4 ---
# Column hour_of_the_day
user_info["hour_of_the_day"] = user_info.purchase_time.apply(lambda x: x.hour)

# --- 5 ---
# Column seconds_since_signup
user_info["seconds_since_signup"] = (user_info.purchase_time - user_info.signup_time).apply(lambda x: x.total_seconds())

# --- 6 ---
# Column countries_from_device (ie. number of different countries per device_id)
# We flag devices that committed purchases from different countries
country_count = user_info.groupby(by=["device_id", "IP_country"]).count().reset_index()
country_count = pd.DataFrame(country_count.groupby(by="device_id").count().IP_country)
user_info = user_info.merge(country_count, left_on="device_id", right_index=True)
user_info.rename(columns={"IP_country_x": "IP_country", "IP_country_y": "countries_from_device"}, inplace=True)

user_info.head(30)

# Column "quick_purchase" : categorise time between sign_up and purchase
user_info["quick_purchase"] = user_info.seconds_since_signup.apply(lambda x: 1 if x < 30 else 0)

# age categorisation
user_info["age_category"] = user_info.age.apply(lambda x:
                                                "< 40" if x < 40 else
                                                "40 - 49" if x < 50 else
                                                "50 -59" if x < 60 else
                                                "60 - 69" if x < 70 else
                                                " > 70")

# Hour of the day categorisation
user_info["period_of_the_day"] = user_info.hour_of_the_day.apply(lambda x:
                                                                 "late night" if x < 4 else
                                                                 "early morning" if x < 8 else
                                                                 "morning" if x < 12 else
                                                                 "afternoon" if x < 16 else
                                                                 "evening" if x < 20 else
                                                                 "early night"
                                                                 )

user_info.head()

user_info.info()

"""## Logistic Regression"""

# Count missing values for each column
missing_counts = user_info.isnull().sum()

# Print the missing value counts
missing_counts

# Drop rows with any missing values.
user_info.dropna(inplace=True)

user_info.info()

# Specify the columns to drop.
columns_to_drop = ["user_id", "signup_time", "purchase_time", "device_id", "ip_address", "hour_of_the_day", "seconds_since_signup", "age"]

# Drop specified columns.
features = user_info.drop(columns=columns_to_drop)

# Display the updated dataframe.
print(features.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

target = features["class"]
features = features.drop(columns=["class"])

features.info()

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=42,
                                                    stratify=target,
                                                    test_size=0.25)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# 5. Identify Column Types
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# 6. Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# 7. Fit Transformer and Transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 8. Train XGBoost Model with class_weight equivalent
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42
)
xgb_model.fit(X_train_processed, y_train)

# 9. Predict Probabilities
y_proba = xgb_model.predict_proba(X_test_processed)[:, 1]

# 11. Apply Optimal Threshold
y_pred = (y_proba >= 0.3).astype(int)

# 12. Evaluate
print("# Classification Report")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 13. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# CATBoost

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

user_info = pd.read_csv("/content/Fraud_Data.csv")
ip_country_mapping = pd.read_csv("/content/IpAddress_to_Country.csv")

# Map IP to Country
ip_country_mapping['lower_bound_ip_address'] = ip_country_mapping['lower_bound_ip_address'].astype(float)
ip_country_mapping['upper_bound_ip_address'] = ip_country_mapping['upper_bound_ip_address'].astype(float)
user_info['ip_address'] = user_info['ip_address'].astype(float)

def IP_to_country(ip):
    try:
        return ip_country_mapping.country[
            (ip_country_mapping.lower_bound_ip_address <= ip) &
            (ip_country_mapping.upper_bound_ip_address >= ip)
        ].iloc[0]
    except IndexError:
        return "Unknown"

user_info["IP_country"] = user_info["ip_address"].apply(IP_to_country)

# Feature Engineering

# Device frequency category
device_freq = user_info.groupby("device_id")["device_id"].transform('count')
user_info["freq_device"] = device_freq.apply(lambda x: str(x) if x < 5 else
                                             "5-10" if 5 <= x <= 10 else
                                             "11-15" if 10 < x <= 15 else "> 15")

# Time difference and quick purchase flag
user_info['signup_time'] = pd.to_datetime(user_info['signup_time'])
user_info['purchase_time'] = pd.to_datetime(user_info['purchase_time'])
user_info['time_diff_hours'] = (user_info['purchase_time'] - user_info['signup_time']).dt.total_seconds() / 3600
user_info['quick_purchase'] = (user_info['time_diff_hours'] < 1).astype(int)

# Time-based features
user_info['month_purchase'] = user_info['purchase_time'].dt.month.astype(str)
user_info['weekday_purchase'] = user_info['purchase_time'].dt.weekday.astype(str)
user_info['period_of_the_day'] = user_info['purchase_time'].dt.hour.apply(
    lambda x: 'Night' if x < 6 else 'Morning' if x < 12 else 'Afternoon' if x < 18 else 'Evening'
)

# Age category
user_info['age_category'] = pd.cut(user_info['age'], bins=[17, 25, 35, 45, 60, 100],
                                   labels=['18-25', '26-35', '36-45', '46-60', '60+'])

# Unique countries per device
device_country_counts = user_info.groupby("device_id")["IP_country"].nunique()
user_info["countries_from_device"] = user_info["device_id"].map(device_country_counts)

# -----------------------------
# Feature Set
features = ['purchase_value', 'source', 'browser', 'sex', 'IP_country', 'freq_device',
            'month_purchase', 'weekday_purchase', 'countries_from_device',
            'quick_purchase', 'age_category', 'period_of_the_day']
target = 'class'

X = user_info[features]
y = user_info[target]

# -----------------------------
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train CatBoost
categorical_features = [
    'source', 'browser', 'sex', 'IP_country', 'freq_device',
    'month_purchase', 'weekday_purchase', 'age_category', 'period_of_the_day'
]

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    auto_class_weights='Balanced',
    verbose=100,
    random_state=42
)

cat_model.fit(X_train, y_train)

# Predict and Evaluate
y_proba = cat_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("# Classification Report (CatBoost - Threshold 0.5)")
print(classification_report(y_test, y_pred))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")

# -----------------------------
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('CatBoost Confusion Matrix (Threshold = 0.5)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Step 1: Define thresholds to test
thresholds = np.arange(0.1, 1.0, 0.05)

# Step 2: Loop through each threshold and calculate metrics
metrics = []

for t in thresholds:
    y_pred_thresh = (y_proba >= t).astype(int)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    accuracy = accuracy_score(y_test, y_pred_thresh)
    metrics.append([t, precision, recall, f1, accuracy])

# Step 3: Store results in a DataFrame
threshold_results = pd.DataFrame(metrics, columns=['Threshold', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])

# Step 4: Show all results
print(threshold_results)

# Step 5 (Optional): Visualize
threshold_results.plot(x='Threshold', y=['Precision', 'Recall', 'F1 Score'], marker='o')
plt.title("Threshold vs Precision, Recall, F1")
plt.grid(True)
plt.show()

# Random Forest & XGB (New Features)

## EDA
"""

#Importing libraries and the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the datasets
user_info = pd.read_csv("Fraud_Data.csv")         # Users information
ip_country_mapping = pd.read_csv("IpAddress_to_Country.csv")  # Country from IP information

ip_country_mapping.head()

ip_country_mapping.info()

user_info.head()

user_info.info()

print("Dataset Shape:", user_info.shape)

class_counts = user_info['class'].value_counts()
print(class_counts)

"""**Mapping/Merging fraud data with country via IP address**"""

ip_country_mapping.upper_bound_ip_address.astype("float")
ip_country_mapping.lower_bound_ip_address.astype("float")
user_info.ip_address.astype("float")

def IP_to_country(ip) :
    try :
        return ip_country_mapping.country[(ip_country_mapping.lower_bound_ip_address < ip)
                                &
                                (ip_country_mapping.upper_bound_ip_address > ip)].iloc[0]
    except IndexError :
        return "Unknown"

import os

# Define the directory path within Colab's file storage
directory = "/content/datasets_fraud"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# country to each IP
user_info["IP_country"] = user_info.ip_address.apply(IP_to_country)

# saving
user_info.to_csv("/content/datasets_fraud/Fraud_data_with_country.csv", index=False)

# loading
user_info= pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

user_info.head()

# Print summary statistics
print(user_info[["purchase_value", "age"]].describe())
print('*'*50)
# Print unique values and their frequencies
for column in ["source", "browser", "sex"]:
    print(user_info[column].value_counts())
    print('*'*50)

# Check for duplicates in the "user_id" column in user_info DataFrame
print("The user_id column includes {} duplicates".format(user_info.duplicated(subset="user_id", keep=False).sum()))

# Calculate duplicate rate based on unique device_id
dup_table = pd.DataFrame(user_info.duplicated(subset="device_id"))
dup_rate = dup_table.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate * 1000) / 10))

# Calculate duplicate rate based on device_id with keep=False
dup_table2 = pd.DataFrame(user_info.duplicated(subset="device_id", keep=False))
dup_rate2 = dup_table2.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate2 * 1000) / 10))

"""The code calculates two duplicate rates based on the device_id column.
The first rate considers only the subsequent occurrences of a duplicate device_id,
while the second considers all occurrences (first and subsequent).
This provides two different perspectives on the extent of device reuse in the dataset.
"""

device_duplicates = pd.DataFrame(user_info.groupby(by="device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace=True)
dupli = device_duplicates[device_duplicates.freq_device >1]
dupli

# Reading the Dataset
user_info = pd.read_csv("/content/datasets_fraud/Fraud_data_with_country.csv")

device_duplicates = pd.DataFrame(user_info.groupby(by = "device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace= True)

dupli = device_duplicates[device_duplicates.freq_device >1]
print("On average, when a device is used more than once it is used {mean} times, and the most used machine was used {maxi} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

dupli = device_duplicates[device_duplicates.freq_device >2]
print("On average, when a device is used more than twice it is used {mean} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

# Merge the device_duplicates with user_info
user_info = user_info.merge(device_duplicates, on="device_id")

# Calculate the proportion of fraud in the dataset
fraud_proportion = user_info["class"].mean() * 100
print("Proportion of fraud in the dataset: {:.1f}%".format(fraud_proportion))

user_info.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot device frequency distribution for values less than 4
g1 = sns.distplot(user_info.freq_device[user_info.freq_device < 4], ax=ax[0])
g1.set(xticks=[1, 2, 3])

# Plot device frequency distribution for values greater than 2
g2 = sns.distplot(user_info.freq_device[user_info.freq_device > 2], ax=ax[1])
g2.set(xticks=range(0, 21, 2))

# Display the plots
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(18, 6))

# Create subplots for bar plots
plt.subplot(1, 3, 1)
sns.barplot(x='source', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Source')

plt.subplot(1, 3, 2)
sns.barplot(x='browser', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Browser')

plt.subplot(1, 3, 3)
sns.barplot(x='sex', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Sex')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
f2, ax2 = plt.subplots(3, 1, figsize=(24, 18))

# Plot purchase_value vs. class
sns.pointplot(x="purchase_value", y="class", data=user_info, ci=None, ax=ax2[0])
ax2[0].set_title("Purchase Value vs. Fraud Probability")

# Plot age vs. class
sns.pointplot(x="age", y="class", data=user_info, ci=None, ax=ax2[1])
ax2[1].set_title("Age vs. Fraud Probability")

# Plot freq_device vs. class
sns.pointplot(x="freq_device", y="class", data=user_info, ci=None, ax=ax2[2])
ax2[2].set_title("Frequency of Device Usage vs. Fraud Probability")

# Show the plots
plt.tight_layout()
plt.show()

user_info.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and axis
f3, ax3 = plt.subplots(1, 1, figsize=(24, 18))

# Plot a stacked bar plot for IP_country vs. class
sns.barplot(x="IP_country", y="class", data=user_info[:10], estimator=sum, ci=None, ax=ax3)

# Show the plot
plt.show()

# Filter IP_country value counts where count is greater than 1000
filtered_counts = user_info.IP_country.value_counts()[user_info.IP_country.value_counts() > 1000]

# Plot the filtered counts as a bar plot
filtered_counts.plot(kind="bar")
plt.xlabel("IP Country")
plt.ylabel("Frequency")
plt.title("IP Country Frequency (Counts > 1000)")
plt.show()

user_info.signup_time

"""## Feature Engineering"""

# --- 1 ---
# Categorisation column freq_device
# We see a clear correlation between freq_device and fraudulent activities. We are going to split freq_device into 7 categories
user_info.freq_device = user_info.freq_device.apply(lambda x:
                                                    str(x) if x < 5 else
                                                    "5-10" if x >= 5 and x <= 10 else
                                                    "11-15" if x > 10 and x <= 15 else
                                                    "> 15")

# Convert signup_time and purchase_time to datetime
user_info.signup_time = pd.to_datetime(user_info.signup_time, format='%Y-%m-%d %H:%M:%S')
user_info.purchase_time = pd.to_datetime(user_info.purchase_time, format='%Y-%m-%d %H:%M:%S')

# --- 2 ---
# Column month
user_info["month_purchase"] = user_info.purchase_time.apply(lambda x: calendar.month_name[x.month])

# --- 3 ---
# Column week
user_info["weekday_purchase"] = user_info.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])

# --- 4 ---
# Column hour_of_the_day
user_info["hour_of_the_day"] = user_info.purchase_time.apply(lambda x: x.hour)

# --- 5 ---
# Column seconds_since_signup
user_info["seconds_since_signup"] = (user_info.purchase_time - user_info.signup_time).apply(lambda x: x.total_seconds())

# --- 6 ---
# Column countries_from_device (ie. number of different countries per device_id)
# We flag devices that committed purchases from different countries
country_count = user_info.groupby(by=["device_id", "IP_country"]).count().reset_index()
country_count = pd.DataFrame(country_count.groupby(by="device_id").count().IP_country)
user_info = user_info.merge(country_count, left_on="device_id", right_index=True)
user_info.rename(columns={"IP_country_x": "IP_country", "IP_country_y": "countries_from_device"}, inplace=True)

user_info.head(30)

# Column "quick_purchase" : categorise time between sign_up and purchase
user_info["quick_purchase"] = user_info.seconds_since_signup.apply(lambda x: 1 if x < 30 else 0)

# age categorisation
user_info["age_category"] = user_info.age.apply(lambda x:
                                                "< 40" if x < 40 else
                                                "40 - 49" if x < 50 else
                                                "50 -59" if x < 60 else
                                                "60 - 69" if x < 70 else
                                                " > 70")

# Hour of the day categorisation
user_info["period_of_the_day"] = user_info.hour_of_the_day.apply(lambda x:
                                                                 "late night" if x < 4 else
                                                                 "early morning" if x < 8 else
                                                                 "morning" if x < 12 else
                                                                 "afternoon" if x < 16 else
                                                                 "evening" if x < 20 else
                                                                 "early night"
                                                                 )

user_info.head()

user_info.info()

"""## Logistic Regression"""

# Count missing values for each column
missing_counts = user_info.isnull().sum()

# Print the missing value counts
missing_counts

# Drop rows with any missing values.
user_info.dropna(inplace=True)

user_info.info()

from tqdm import tqdm

# Ensure datetime column is parsed
user_info['purchase_time'] = pd.to_datetime(user_info['purchase_time'])

# Initialize the column
user_info['transactions_last_24h_per_device'] = 0

# Loop over each device
for device_id, group in tqdm(user_info.groupby('device_id'), desc='Calculating transactions_last_24h'):
    group = group.sort_values(by='purchase_time')
    purchase_times = group['purchase_time'].tolist()
    result = []

    for i in range(len(purchase_times)):
        current_time = purchase_times[i]
        past_24h = [t for t in purchase_times[:i] if (current_time - t).total_seconds() <= 86400]
        result.append(len(past_24h))

    user_info.loc[group.index, 'transactions_last_24h_per_device'] = result




# ----------------------------------------
# 2. avg_purchase_value_per_device
user_info['avg_purchase_value_per_device'] = user_info.groupby('device_id')['purchase_value'].transform('mean')

# # ----------------------------------------
# # 3. fraud_rate_per_device
# device_fraud_rate = user_info.groupby('device_id')['class'].mean()
# user_info['fraud_rate_per_device'] = user_info['device_id'].map(device_fraud_rate)

# ----------------------------------------
# 4. num_accounts_per_ip
user_info['ip_address'] = user_info['ip_address'].astype(str)
ip_user_counts = user_info.groupby('ip_address')['user_id'].nunique()
user_info['num_accounts_per_ip'] = user_info['ip_address'].map(ip_user_counts)

# ----------------------------------------
# 5. signup_day_gap
user_info['signup_day_gap'] = (user_info['purchase_time'] - user_info['signup_time']).dt.days

# ----------------------------------------
# Preview new features
user_info[[
    'transactions_last_24h_per_device',
    'avg_purchase_value_per_device',

    'num_accounts_per_ip',
    'signup_day_gap'
]].describe()

# Specify the columns to drop.
columns_to_drop = ["user_id", "signup_time", "purchase_time", "device_id", "ip_address", "hour_of_the_day", "seconds_since_signup", "age"]

# Drop specified columns.
features = user_info.drop(columns=columns_to_drop)

# Display the updated dataframe.
print(features.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

target = features["class"]
features = features.drop(columns=["class"])

features.info()

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=42,
                                                    stratify=target,
                                                    test_size=0.25)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# 5. Identify Column Types
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# 6. Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# 7. Fit Transformer and Transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 8. Train XGBoost Model with class_weight equivalent
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42
)
xgb_model.fit(X_train_processed, y_train)

# 9. Predict Probabilities
y_proba = xgb_model.predict_proba(X_test_processed)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score

# Try thresholds from 0.01 to 0.99
thresholds = np.arange(0.01, 1.0, 0.01)
results = []

for threshold in thresholds:
    y_pred_temp = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_temp)
    recall = recall_score(y_test, y_pred_temp)
    f1 = f1_score(y_test, y_pred_temp)
    results.append((threshold, precision, recall, f1))

# Convert to DataFrame
threshold_df = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1"])

# Filter for recall >= 0.85
filtered = threshold_df[threshold_df.recall >= 0.85]

# Sort by F1 score (or choose by highest precision if needed)
best = filtered.sort_values(by="f1", ascending=False).iloc[0]
best_threshold = best['threshold']
print(f"Optimal Threshold (recall ≥ 0.85): {best_threshold:.2f}")
print(best)

# 11. Apply Optimal Threshold
y_pred = (y_proba >= 0.30).astype(int)

# 12. Evaluate
print("# Classification Report")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 13. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 11. Apply Optimal Threshold
y_pred = (y_proba >= 0.50).astype(int)

# 12. Evaluate
print("# Classification Report")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 13. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# 1. Prepare Data: Add new features
X = user_info[
    ['purchase_value', 'source', 'browser', 'sex', 'IP_country', 'freq_device',
     'month_purchase', 'weekday_purchase', 'countries_from_device', 'quick_purchase',
     'age_category', 'period_of_the_day',

     # new engineered features
     'transactions_last_24h_per_device',
     'avg_purchase_value_per_device',

     'num_accounts_per_ip',
     'signup_day_gap'
    ]
]
y = user_info['class']

# -----------------------------------
# 2. Feature Types
numeric_features = [
    'purchase_value', 'countries_from_device', 'quick_purchase',
    'transactions_last_24h_per_device', 'avg_purchase_value_per_device',
    'num_accounts_per_ip', 'signup_day_gap'
]

categorical_features = [
    'source', 'browser', 'sex', 'IP_country', 'freq_device',
    'month_purchase', 'weekday_purchase', 'age_category', 'period_of_the_day'
]

# 3. Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42)

# 5. Train Random Forest with custom hyperparameters
rf_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1  # use all cores for faster training
)

rf_model.fit(X_train, y_train)

print(f"Number of trees trained: {len(rf_model.estimators_)}")

# 5. Predict with default threshold = 0.5
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]  # still useful for ROC AUC

# 6. Evaluation
print("# Classification Report Random Forest (Threshold = 0.5)")
print(classification_report(y_test, y_pred))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")

# -------------------------------
# 7. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix (Default Threshold = 0.5)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Assuming you already have:
# - y_test (true labels)
# - y_proba (predicted probabilities from model.predict_proba()[:, 1])

# ----------------------------------
# Set thresholds you want to test
thresholds = np.arange(0.1, 1.0, 0.05)

# Initialize list to store results
metrics = []

# ----------------------------------
# Loop through each threshold
for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    metrics.append([t, precision, recall, f1, accuracy])

# ----------------------------------
# Create a DataFrame of results
threshold_results = pd.DataFrame(metrics, columns=['Threshold', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])

# Display all results
print(threshold_results)

# ----------------------------------
# Optional: Sort by F1 Score or filter
# print(threshold_results.sort_values(by='F1 Score', ascending=False))
# print(threshold_results[threshold_results['Recall'] >= 0.85])