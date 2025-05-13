#!/usr/bin/env python
# coding: utf-8

# In[5]:


# 📊 Titanic Exploratory Data Analysis (EDA) - Final Notebook
# Author: [Your Name] | Date: May 13, 2025

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 🚀 Load the Dataset
file_path = r"C:\Users\prudh\Downloads\Titanic-Dataset.csv"
titanic = pd.read_csv(file_path)

# 🧹 Data Cleaning and Preparation
# Fill missing values
median_age = titanic['Age'].median()
titanic.loc[:, 'Age'] = titanic['Age'].fillna(median_age)

top_embarked = titanic['Embarked'].mode()[0]
titanic.loc[:, 'Embarked'] = titanic['Embarked'].fillna(top_embarked)

# Encode 'Sex'
titanic.loc[:, 'Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

# Drop irrelevant columns
titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

# 📋 Data Overview
print("\n✅ Data Info:\n")
print(titanic.info())
print("\n✅ Data Description:\n")
print(titanic.describe())

# 🔥 Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(titanic[['Survived', 'Age', 'Fare', 'Pclass', 'Sex', 'SibSp', 'Parch']].corr(), 
            annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix - Key Features")
plt.show()

# 📈 Pairplot for Key Features
sns.pairplot(titanic[['Survived', 'Age', 'Fare', 'Pclass', 'Sex']], 
             hue='Survived', diag_kind='kde', palette='Set1', markers=["o", "s"])
plt.show()

# 📊 Distribution Plot - Age
plt.figure(figsize=(8, 6))
sns.histplot(titanic['Age'], kde=True, color='steelblue', bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 💰 Boxplot - Fare Distribution
plt.figure(figsize=(8, 6))
sns.boxplot(x=titanic['Fare'], color='lightblue')
plt.title('Fare Boxplot')
plt.xlabel('Fare')
plt.show()

# 👥 Countplot - Gender Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='Sex', data=titanic, palette='Set2')
plt.title('Count of Passengers by Gender')
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()

# 🚀 Interactive Sunburst Plot (Plotly)
fig = px.sunburst(titanic, path=['Pclass', 'Sex', 'Survived'], values='Fare',
                 title='Survival Rates by Class and Gender',
                 color='Survived', color_continuous_scale='Blues')
fig.show()

# ✨ Key Insights and Final Note
print("\n✅ EDA Complete! Ready for Feature Engineering and Modeling.")

# Key Insights
print("\n🔍 Key Insights:")
print("- Passengers in 1st class had a significantly higher survival rate.")
print("- Females had a much higher survival rate than males.")
print("- Younger passengers (especially children) had better survival chances.")
print("- Passengers with higher fares tended to survive more, likely linked to class.")
print("- Most passengers embarked from Southampton.")


# In[ ]:




