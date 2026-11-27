
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Customer dataset
data = {
    'Annual_Income': [15, 18, 20, 25, 30, 60, 65, 70, 75, 80],
    'Spending_Score': [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
}

df = pd.DataFrame(data)

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Train K-Means model
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# Prediction function
def predict_customer(income, spending):

    new_customer = pd.DataFrame(
        [[income, spending]],
        columns=['Annual_Income', 'Spending_Score']
    )

    scaled_customer = scaler.transform(new_customer)

    cluster = kmeans.predict(scaled_customer)[0]

    if cluster == 0:
        return "Budget Customers"
    else:
        return "Premium Customers"


def get_customer_data():

    df['Cluster'] = kmeans.labels_

    df['Customer_Group'] = df['Cluster'].map({
        0: 'Budget Customers',
        1: 'Premium Customers'
    })

    return df


