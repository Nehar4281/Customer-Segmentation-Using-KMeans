import streamlit as st
from model import predict_customer, get_customer_data
from model import predict_customer

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️"
)

# Title
st.title("🛍️ Customer Segmentation")

# Sidebar
st.sidebar.title("Customer Segmentation")
st.sidebar.write("AI/ML Mini Project")

st.sidebar.markdown("---")

st.sidebar.subheader("About Project")

st.sidebar.write(
    "This application uses the K-Means "
    "clustering algorithm to group customers "
    "based on their annual income and spending score."
)

st.sidebar.markdown("---")

st.sidebar.write("Algorithm: K-Means")
st.sidebar.write("Language: Python")
st.sidebar.write("Framework: Streamlit")

st.write("AI-powered customer group prediction using K-Means")

st.divider()

# Dashboard statistics

customer_data = get_customer_data()

total_customers = len(customer_data)

budget_customers = len(
    customer_data[customer_data['Customer_Group'] == 'Budget Customers']
)

premium_customers = len(
    customer_data[customer_data['Customer_Group'] == 'Premium Customers']
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Budget Customers", budget_customers)
col3.metric("Premium Customers", premium_customers)


# Customer inputs
st.subheader("Enter Customer Details")

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=22.0
)

spending = st.number_input(
    "Spending Score",
    min_value=0.0,
    max_value=100.0,
    value=28.0
)

# Prediction
if st.button("Predict Customer Group"):

    if income < 15 or income > 80:
        st.warning("Income is outside the training data range (15–80).")

    elif spending < 20 or spending > 80:
        st.warning("Spending score is outside the training data range (20–80).")

    else:
        result = predict_customer(income, spending)

        st.success(f"Predicted Customer Group: {result}")

# Customer visualization

st.divider()

st.subheader("Customer Segmentation Graph")

customer_data = get_customer_data()

st.scatter_chart(
    customer_data,
    x="Annual_Income",
    y="Spending_Score",
    color="Customer_Group"
)


# Customer details table

st.divider()

st.subheader("Customer Details")

st.dataframe(
    customer_data,
    use_container_width=True
)
