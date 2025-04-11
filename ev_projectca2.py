import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
file_path = "C:\\Users\\ky143\\Electric_Vehicle_Population_Data.csv"
df = pd.read_csv(file_path)


print("Shape of the dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample records:")
print(df.describe())
print(df.head(10))
print(df.tail(10))

print("Actual column names:")
print(df.columns.tolist())

print("Missing values:\n", df.isnull().sum())

cleaned_df = df.dropna(subset=['Make', 'Model Year', 'Electric Range', 'Electric Vehicle Type', 'City', 'County'], inplace=True)
print(cleaned_df)


#Check for missing values
print("Missing values:\n", df.isnull().sum())


#Convert relevant columns to appropriate datatypes
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')

#Drop rows with missing essential values
df.dropna(subset=['Make', 'Model Year', 'Electric Range', 'Electric Vehicle Type', 'City', 'County'], inplace=True)

#Calculate Correlation
correlation = df[['Model Year', 'Electric Range']].corr()
print("Correlation Matrix:\n", correlation)


# Calculate Covariance 
covariance = df[['Model Year', 'Electric Range']].cov()
print("\nCovariance Matrix:\n", covariance)


#Filter out unrealistic data
df = df[df['Electric Range'] < 500]
df = df[df['Model Year'] >= 2000]

#Preview cleaned data
print("\nCleaned DataFrame:")
print(df.head())


#1.Electric Vehicle Type Distribution(using Seaborn Count Plot)
if 'Electric Vehicle Type' in df.columns:
    plt.figure(figsize=(6, 6))
    sns.countplot(data=df, x='Electric Vehicle Type', hue='Electric Vehicle Type',
        palette='pastel', order=df['Electric Vehicle Type'].value_counts().index)
    plt.title('Distribution of Electric Vehicle Type')
    plt.xlabel('EV Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

#2.Top 10 Cities with Most EVs(Bar Plot)
top_cities = df['City'].value_counts().head(10)
colors = sns.color_palette("coolwarm", n_colors=10)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=top_cities.values, y=top_cities.index, palette=colors, hue=top_cities.index)
plt.title("Top 10 Cities with Most EVs")
plt.xlabel("Number of Vehicles")
plt.ylabel("City")
plt.legend(title='City', labels=top_cities.index, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#3.EVs Over Model Years Growth(Line Plot)
yearly_counts = df['Model Year'].value_counts().sort_index()
plt.figure(facecolor='green', figsize=(12, 6))
ax = sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='green', label='EV Count')
ax.set_facecolor('red')
plt.title("EV Population Growth by Model Year", color='white')
plt.xlabel("Model Year", color='white')
plt.ylabel("Number of EVs", color='white')
ax.tick_params(colors='blue')
plt.grid(True, color='white', linestyle='--', alpha=0.3)
plt.legend(title='Legend', facecolor='red', edgecolor='white', labelcolor='white')
plt.tight_layout()
plt.show()


#4.Electric Range Distribution(HistogramPlot + KDE)
plt.figure(figsize=(12, 6))
sns.histplot(df['Electric Range'], bins=30, kde=True, color='purple', label='Electric Range Distribution')
plt.title("Distribution of Electric Vehicle Ranges")
plt.xlabel("Electric Range (miles)")
plt.ylabel("Count")
plt.legend(title='Legend')
plt.show()

#5.Average Electric Range by Vehicle Make Top-10(Boxplot)
# Filter top 10 makes
top_makes = df['Make'].value_counts().head(10).index
filtered_df = df[df['Make'].isin(top_makes)]
plt.figure(figsize=(12, 6))
colors = sns.color_palette('coolwarm', n_colors=len(top_makes))
sns.boxplot(data=filtered_df, x='Make', y='Electric Range', hue='Make', palette=colors, dodge=False)
plt.title("Electric Range Distribution by Top 10 Vehicle Makes")
plt.xlabel("Make")
plt.ylabel("Electric Range (miles)")
plt.xticks(rotation=45)
plt.legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#6.Correlation Heatmap (Numerical Fields)
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 6))
heatmap = sns.heatmap( numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Between Numerical Features")
plt.xlabel("Numerical Features (Columns)")
plt.ylabel("Numerical Features (Rows)")
plt.tight_layout()
plt.show()  

#7.Line Plot with hue='Electric Vehicle Type'
plt.figure(figsize=(12, 6))
grouped = df.groupby(['Model Year', 'Electric Vehicle Type']).size().reset_index(name='Count')
ax = sns.lineplot( data=grouped, x='Model Year', y='Count', hue='Electric Vehicle Type', marker='o', palette='deep')
fig = plt.gcf()
fig.patch.set_facecolor('mistyrose')      
ax.set_facecolor('lavenderblush')        
plt.title('EV Population Growth by Type Over Model Years', color='darkred')
plt.xlabel('Model Year', color='darkred')
plt.ylabel('Number of EVs', color='darkred')
ax.tick_params(colors='darkred')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title='EV Type', facecolor='#ffe6f0', edgecolor='white', labelcolor='darkred', title_fontsize='medium')
plt.tight_layout()
plt.show()

#8.Combo Chart: Bar + Line(Model Year vs Count & Avg Range)
plt.style.use('dark_background')
sns.set_palette('pastel')
model_year_data = df.groupby('Model Year').agg({'VIN (1-10)': 'count', 'Electric Range': 'mean'}).rename(columns={'VIN (1-10)': 'EV Count', 'Electric Range': 'Avg Range'}).reset_index()
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=model_year_data, x='Model Year', y='EV Count', ax=ax1, color='cyan')
ax1.set_ylabel('Number of EVs', color='cyan')
ax1.tick_params(axis='y', labelcolor='cyan')
ax1.set_xlabel('Model Year')
plt.xticks(rotation=45)

ax2 = ax1.twinx()
sns.lineplot(data=model_year_data, x='Model Year', y='Avg Range', ax=ax2, color='orange', marker='o')
ax2.set_ylabel('Average Electric Range (miles)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.title("EV Count and Average Electric Range by Model Year", color='white')
ax1.legend(['EV Count'], loc='upper left', facecolor='pink', labelcolor='cyan')
ax2.legend(['Avg Range'], loc='upper right', facecolor='black', labelcolor='orange')
plt.tight_layout()
plt.show()

#9.Average Electric Range Over Years by Vehicle Type(Seaborn Lineplot)
plt.figure(figsize=(12, 6))
avg_range_by_type_year = df.groupby(['Model Year', 'Electric Vehicle Type'])['Electric Range'].mean().reset_index()
sns.lineplot(data=avg_range_by_type_year, x='Model Year', y='Electric Range', hue='Electric Vehicle Type', marker='o', palette='pastel')
plt.title("Average Electric Range Over Model Years by Vehicle Type")
plt.xlabel("Model Year")
plt.ylabel("Average Electric Range (miles)")
plt.legend(title='Vehicle Type')
plt.grid(True)
plt.tight_layout()
plt.show()

#10.EV Count by ZIP Code (Barplot)
sns.set_style("whitegrid") 
top_zips = df['Postal Code'].value_counts().head(10).reset_index()
top_zips.columns = ['Postal Code', 'EV Count']
plt.figure(figsize=(11, 6))
sns.barplot(data=top_zips, x='EV Count', y='Postal Code', hue='Postal Code', dodge=False, palette='cubehelix')
plt.title("Top 10 ZIP Codes with Most EVs")
plt.xlabel("Number of EVs")
plt.ylabel("ZIP Code")
plt.legend(title='Postal Code', loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()



