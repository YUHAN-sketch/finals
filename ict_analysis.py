import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Create output directory for plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Read the dataset
print("Loading dataset...")
df = pd.read_excel('ICT.xlsx')

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

# Check for null values
print("\nNull values in each column:")
null_counts = df.isnull().sum()
print(null_counts)

# Check total number of rows
print(f"\nTotal number of rows before cleaning: {len(df)}")

# Remove null values
df_clean = df.dropna()

print(f"\nNumber of rows after cleaning: {len(df_clean)}")

# Verify we have at least 100 data points
if len(df_clean) < 100:
    raise ValueError("Dataset must contain at least 100 data points after cleaning")
else:
    print(f"✓ Dataset contains {len(df_clean)} data points (meets requirement of 100+)")

# Save the cleaned dataset
df_clean.to_csv('ict_clean.csv', index=False)
print("✓ Cleaned dataset saved to 'ict_clean.csv'")

# Select only numeric columns for analysis
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
numeric_data = df_clean[numeric_columns]

print("\nNumeric columns for analysis:")
print(numeric_columns.tolist())

# NumPy Operations (5 operations)
print("\nPerforming NumPy operations...")

# 1. Calculate mean of numeric columns
numeric_means = np.mean(numeric_data, axis=0)
print("\n1. Mean values of numeric columns:")
print(numeric_means)

# 2. Calculate standard deviation
numeric_std = np.std(numeric_data, axis=0)
print("\n2. Standard deviation of numeric columns:")
print(numeric_std)

# 3. Normalize the data
normalized_data = (numeric_data - numeric_means) / numeric_std
print("\n3. Data normalization completed")

# 4. Calculate correlation matrix using NumPy
correlation_matrix = np.corrcoef(numeric_data.T)
print("\n4. Correlation matrix (using NumPy):")
print(correlation_matrix)

# 5. Calculate percentile values
percentiles = np.percentile(numeric_data, [25, 50, 75], axis=0)
print("\n5. 25th, 50th, and 75th percentiles:")
print(percentiles)

# SciPy Operations
print("\nPerforming SciPy operations...")
# Perform t-test between first two numeric columns
col1, col2 = numeric_columns[:2]
t_stat, p_val = stats.ttest_ind(numeric_data[col1], numeric_data[col2])
print(f"\nT-test between {col1} and {col2}:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3e}")

# Statsmodels Operations
print("\nPerforming Statsmodels operations...")
# Perform linear regression
X = sm.add_constant(numeric_data[col1])
y = numeric_data[col2]
model = sm.OLS(y, X).fit()
print("\nLinear Regression Results:")
print(model.summary())

# Correlation Analysis
print("\nGenerating visualizations...")
correlation_matrix = numeric_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()
print("✓ Correlation heatmap saved")

# Distribution plots for key features
key_features = numeric_columns[:4]  # First 4 numeric columns
plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=numeric_data, x=feature, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('plots/feature_distributions.png')
plt.close()
print("✓ Distribution plots saved")

# Box plots for key features
plt.figure(figsize=(15, 6))
numeric_data[key_features].boxplot()
plt.title('Box Plots of Key Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/feature_boxplots.png')
plt.close()
print("✓ Box plots saved")

# Scatter plot matrix for key features
plt.figure(figsize=(12, 8))
sns.pairplot(numeric_data[key_features])
plt.savefig('plots/feature_scatter_matrix.png')
plt.close()
print("✓ Scatter plot matrix saved")

# Correlation plot with regression line (I.B.5)
plt.figure(figsize=(10, 6))
sns.regplot(data=numeric_data, x=col1, y=col2, scatter_kws={'alpha':0.5})
plt.title(f'Correlation between {col1} and {col2} with Regression Line')
plt.xlabel(col1)
plt.ylabel(col2)
plt.tight_layout()
plt.savefig('plots/correlation_regression.png')
plt.close()
print("✓ Correlation plot with regression line saved")

# Feature Importance Analysis using PCA
print("\nPerforming PCA analysis...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), 
         marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig('plots/pca_variance.png')
plt.close()
print("✓ PCA variance plot saved")

# Print feature importance
feature_importance = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=numeric_columns
)
print("\nFeature Importance (Principal Components):")
print(feature_importance.round(3))

# Save all results to a text file
print("\nSaving analysis results...")
with open('analysis_results.txt', 'w') as f:
    f.write("ICT Dataset Analysis Results\n")
    f.write("===========================\n\n")
    f.write(f"Total number of records: {len(df_clean)}\n")
    f.write(f"Number of numeric features: {len(numeric_columns)}\n\n")
    f.write("Data Cleaning Summary:\n")
    f.write(f"Initial rows: {len(df)}\n")
    f.write(f"Rows after cleaning: {len(df_clean)}\n")
    f.write(f"Rows removed: {len(df) - len(df_clean)}\n\n")
    f.write(f"Correlation between {col1} and {col2}:\n")
    f.write(f"Correlation coefficient: {correlation_matrix.loc[col1, col2]:.3f}\n")
    f.write(f"P-value: {p_val:.3e}\n\n")
    f.write("Feature Importance (Principal Components):\n")
    f.write(feature_importance.round(3).to_string())
    f.write("\n\nT-test Results:\n")
    f.write(f"t-statistic: {t_stat:.3f}\n")
    f.write(f"p-value: {p_val:.3e}\n")
    f.write("\nLinear Regression Summary:\n")
    f.write(model.summary().as_text())
print("✓ Analysis results saved to 'analysis_results.txt'") 