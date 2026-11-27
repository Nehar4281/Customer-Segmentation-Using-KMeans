
import pandas as pd

# Creating customer dataset
data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [20, 22, 25, 30, 35, 40, 45, 50, 55, 60],
    'Annual_Income': [15, 18, 20, 25, 30, 60, 65, 70, 75, 80],
    'Spending_Score': [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Display dataset
print("CUSTOMER DATASET")
print(df)

# Display basic information
print("\nDataset Information:")
print(df.describe())

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Visualize customer data
plt.figure(figsize=(8, 5))

plt.scatter(
    df['Annual_Income'],
    df['Spending_Score']
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Income vs Spending Score')

plt.savefig('customer_graph.png')
print("Graph saved successfully!")


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
X = df[['Annual_Income', 'Spending_Score']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# Train the model
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display customer groups
print("\nCUSTOMER SEGMENTATION RESULTS")
print(df[['Customer_ID', 'Annual_Income',
          'Spending_Score', 'Cluster']])


# Plot customer clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster'],
    cmap='viridis',
    s=100
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation Using K-Means')

plt.savefig('customer_clusters.png')

print("\nCluster graph saved successfully!")


# ELBOW METHOD

wcss = []

for i in range(1, 8):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8, 5))

plt.plot(range(1, 8), wcss, marker='o')

plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.savefig('elbow_graph.png')

print("\nElbow graph saved successfully!")


# CUSTOMER GROUP NAMES

df['Customer_Group'] = df['Cluster'].map({
    0: 'Budget Customers',
    1: 'Premium Customers'
})

print("\nFINAL CUSTOMER SEGMENTATION")
print(df[['Customer_ID', 'Annual_Income',
          'Spending_Score', 'Customer_Group']])


# SAVE FINAL RESULTS

df.to_csv('customer_segments.csv', index=False)

print("\nCustomer segmentation results saved successfully!")


# PREDICT NEW CUSTOMER GROUP

new_customer = [[22, 28]]

# Scale the new customer's data
new_customer_scaled = scaler.transform(new_customer)

# Predict the customer cluster
predicted_cluster = kmeans.predict(new_customer_scaled)[0]

# Display customer group
if predicted_cluster == 0:
    print("\nNew Customer Group: Budget Customers")
else:
    print("\nNew Customer Group: Premium Customers")


# INTERACTIVE CUSTOMER PREDICTION

print("\n--- NEW CUSTOMER PREDICTION ---")

income = float(input("Enter annual income: "))
spending = float(input("Enter spending score: "))

new_customer = pd.DataFrame(
    [[income, spending]],
    columns=['Annual_Income', 'Spending_Score']
)

new_customer_scaled = scaler.transform(new_customer)

predicted_cluster = kmeans.predict(new_customer_scaled)[0]

if predicted_cluster == 0:
    print("Customer Group: Budget Customers")
else:
    print("Customer Group: Premium Customers")
