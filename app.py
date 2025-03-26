# import pandas as pd
# import joblib


# final_model = joblib.load('final_model.pkl')
# joblib.dump(final_model, 'final_model.pkl')

# test_df = pd.read_csv("test.csv")

# features = ['month', 'day', 'weekday', 'is_weekend', 'store_nbr', 'onpromotion']

# X_test = test_df[features]


# # Predict sales
# test_df['predicted_sales'] = final_model.predict(X_test)


import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

# Function to load training data
@st.cache_data
def load_train_data():
    return pd.read_csv("train.csv")

train = load_train_data()

features = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'year', 'month', 'day', 'weekday', 'is_weekend']
target = 'sales'

label_encoders = {}
for col in ['family', 'city', 'state', 'type']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le


X_train = train[features]
y_train = train[target]


xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, "xgb_model.pkl")


def main():
    st.title("🛒 Store Sales Prediction App")
    st.write("Upload a **test.csv** file to predict store sales.")

    uploaded_file = st.file_uploader("Choose a test.csv file", type="csv")

    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        
        for col in ['family', 'city', 'state', 'type']:
            if col in test_data.columns and col in label_encoders:
                test_data[col] = label_encoders[col].transform(test_data[col])

        X_test = test_data[features]


        xgb_model = joblib.load("xgb_model.pkl")
        predictions = xgb_model.predict(X_test)

        test_data['Predicted_Sales'] = predictions

        st.write("### Predicted Sales")
        st.write(test_data[['store_nbr', 'family', 'Predicted_Sales']].head(10))

        csv_output = test_data.to_csv(index=False)
        st.download_button(label="Download Predictions", data=csv_output, file_name="predicted_sales.csv", mime="text/csv")

st.write("📂 Upload a test dataset to start predictions!")

if __name__ == '__main__':
    main()
