import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("personal_carbon_footprint_behavior.csv")

# Create folder to save graphs
os.makedirs("graphs", exist_ok=True)

# ---------------- BASIC DATASET ANALYSIS ----------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").nunique())

print("\nCategory Counts:")
for col in ["day_type", "transport_mode", "food_type", "carbon_impact_level"]:
    print(f"\n{col}:")
    print(df[col].value_counts())


# ---------------- CORRELATION ANALYSIS ----------------

print("\nCorrelation with Carbon Footprint:")

correlation = df.select_dtypes(include="number").corr()["carbon_footprint_kg"]
correlation = correlation.drop(["carbon_footprint_kg", "user_id"])

print(correlation.sort_values(ascending=False))

plt.figure(figsize=(10, 6))
correlation.sort_values().plot(kind="barh")

plt.title("Correlation with Carbon Footprint")
plt.xlabel("Correlation")
plt.ylabel("Factors")
plt.tight_layout()

plt.savefig("graphs/correlation_carbon_footprint.png", dpi=300)
plt.show()
plt.close()


# ---------------- CARBON FOOTPRINT DISTRIBUTION ----------------

plt.figure(figsize=(8, 5))

plt.hist(df["carbon_footprint_kg"], bins=20)

plt.title("Distribution of Carbon Footprint")
plt.xlabel("Carbon Footprint (kg)")
plt.ylabel("Number of Users")

plt.tight_layout()

plt.savefig("graphs/carbon_footprint_distribution.png", dpi=300)
plt.show()
plt.close()


# ---------------- TRANSPORT MODE ----------------

transport_carbon = (
    df.groupby("transport_mode")["carbon_footprint_kg"]
    .mean()
    .sort_values()
)

print("\nAverage Carbon Footprint by Transport Mode:")
print(transport_carbon)

transport_carbon.plot(kind="bar", figsize=(9, 5))

plt.title("Average Carbon Footprint by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Average Carbon Footprint (kg)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("graphs/transport_carbon_footprint.png", dpi=300)
plt.show()
plt.close()


# ---------------- FOOD TYPE ----------------

food_carbon = (
    df.groupby("food_type")["carbon_footprint_kg"]
    .mean()
    .sort_values()
)

print("\nAverage Carbon Footprint by Food Type:")
print(food_carbon)

food_carbon.plot(kind="bar", figsize=(9, 5))

plt.title("Average Carbon Footprint by Food Type")
plt.xlabel("Food Type")
plt.ylabel("Average Carbon Footprint (kg)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("graphs/food_carbon_footprint.png", dpi=300)
plt.show()
plt.close()


# ---------------- DAY TYPE ----------------

day_carbon = (
    df.groupby("day_type")["carbon_footprint_kg"]
    .mean()
    .sort_values()
)

print("\nAverage Carbon Footprint by Day Type:")
print(day_carbon)

day_carbon.plot(kind="bar", figsize=(8, 5))

plt.title("Average Carbon Footprint by Day Type")
plt.xlabel("Day Type")
plt.ylabel("Average Carbon Footprint (kg)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("graphs/day_type_carbon_footprint.png", dpi=300)
plt.show()
plt.close()


# ---------------- CARBON IMPACT LEVEL ----------------

impact_carbon = (
    df.groupby("carbon_impact_level")["carbon_footprint_kg"]
    .mean()
    .sort_values()
)

print("\nAverage Carbon Footprint by Impact Level:")
print(impact_carbon)

impact_carbon.plot(kind="bar", figsize=(8, 5))

plt.title("Average Carbon Footprint by Carbon Impact Level")
plt.xlabel("Carbon Impact Level")
plt.ylabel("Average Carbon Footprint (kg)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("graphs/carbon_impact_level.png", dpi=300)
plt.show()
plt.close()


print("\n===================================")
print("ALL ANALYSIS COMPLETED SUCCESSFULLY")
print("All graphs have been saved in the 'graphs' folder.")
print("===================================")