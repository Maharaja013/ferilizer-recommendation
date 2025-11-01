import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
df = pd.read_csv('/kaggle/input/fertilizer-recommendation/fertilizer_recommendation_dataset.csv')
df.head()
df.shape
df.isnull().sum()
plt.figure(figsize=(10,5))
sns.countplot(x='Fertilizer', data=df, order=df['Fertilizer'].value_counts().index,hue='Fertilizer', palette='viridis')
plt.xticks(rotation=90)
plt.title('Fertilizer Type Distribution')
plt.show()
df['Fertilizer'].value_counts()
df['Crop'].value_counts()
df['Soil'].value_counts()
df = pd.get_dummies(df, columns=['Soil'], dtype=np.float64)
df.head()
data = df.drop('Remark',axis=1)
data.head()
soil_mapping = {
    'Soil_Acidic Soil': 'Acidic_Soil',
    'Soil_Alkaline Soil': 'Alkaline_Soil',
    'Soil_Loamy Soil': 'Loamy_Soil',
    'Soil_Neutral Soil': 'Neutral_Soil',
    'Soil_Peaty Soil': 'Peaty_Soil'
}
data.rename(columns=soil_mapping,inplace=True)
data.head()
label_encoder = LabelEncoder()
data['Crops'] = label_encoder.fit_transform(data['Crop'])
data.head()
data = data.drop('Crop',axis=1)
data.head()
# Boxplots to detect outliers
plt.figure(figsize=(15,8))
for i, col in enumerate(['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon']):
    plt.subplot(2,4,i+1)
    sns.boxplot(y=data[col],color='lime')
    plt.title(f'Boxplot of {col}')
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show() 
# Handle Outliers
# IQR Method to identify and cap Outliers
def cap_outliers(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)


for col in ['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon']:
    cap_outliers(data, col)
    # Boxplot
plt.figure(figsize=(15,8))
for i, col in enumerate(['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon']):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=data[col],color='lime')
    plt.title(f'Boxplot of {col}')
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()
# Histograms for the numerical cols
data[['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon']].hist(bins=15, figsize=(15, 10))
plt.show()
# KDE plots for numerical features
for col in['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon']:
    sns.kdeplot(data[col], fill=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    # Select only numeric cols
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()
# Pairplot
sns.pairplot(
    data, 
    vars=['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon'],
    hue='Fertilizer', 
    palette='husl'
)
plt.suptitle("Pairplot of Features by Fertilizer Type", y=1.02)
plt.show()
# Boxplots of features grouped by crop
for col in ['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Fertilizer', y=col, data=data)
    plt.xticks(rotation=90)
    plt.title(f'{col} Distribution by Fertilizer Type')
    plt.show()
    fertilizer_mean = data.groupby('Fertilizer').mean()
print(fertilizer_mean)

plt.figure(figsize=(24, 12))
sns.heatmap(fertilizer_mean, annot=True, cmap='YlGnBu',fmt='f')
plt.title('Average Feature Values by Fertilizer Type')
plt.show()
label_encoder_fertilizer = LabelEncoder()
data['fertilizer'] = label_encoder_fertilizer.fit_transform(data['Fertilizer'])
data.head()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon','Acidic_Soil','Alkaline_Soil','Loamy_Soil','Neutral_Soil','Peaty_Soil','Crops']])
X = pd.DataFrame(scaled_features, columns=['Temperature','Moisture','Rainfall','PH','Nitrogen', 'Phosphorous', 'Potassium','Carbon','Acidic_Soil','Alkaline_Soil','Loamy_Soil','Neutral_Soil','Peaty_Soil','Crops'])
y = data['fertilizer']
feature_names = X.columns.tolist()
data.shape
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42,n_estimators=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
model.score(X_test,y_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Heatmap for Fertilizer Types")
plt.show()
print(classification_report(y_test,y_pred))
dump(model, "fertilizer_recommendation.joblib")
dump(scaler, "fertilizer_scaler.joblib")
dump(label_encoder_fertilizer, "fertilizer_encoder.joblib")
dump(label_encoder, "crop_encoder.joblib")
dump(feature_names, "fertilizer_feature_names.joblib")