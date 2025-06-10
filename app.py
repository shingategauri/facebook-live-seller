import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Live_20210128.csv")
    df = df.drop(columns=['Column1', 'Column2', 'Column3', 'Column4'], errors='ignore')
    return df

df = load_data()
st.title("📊 LivePulse Analytics - Facebook Live Data Dashboard")

# Preview data
st.header("🔍 Data Preview")
st.dataframe(df.head())

# User input for filtering
st.sidebar.header("🔧 Filter Options")
status_type = st.sidebar.multiselect("Select Status Type", options=df['status_type'].unique(), default=df['status_type'].unique())
filtered_df = df[df['status_type'].isin(status_type)]

# Summary Report
st.header("📋 Summary Report")
st.write(filtered_df.describe())

# Visualizations
st.header("📈 Visual Analysis")

# Bar chart
fig, ax = plt.subplots()
filtered_df['status_type'].value_counts().plot(kind='bar', ax=ax)
ax.set_title("Count of Posts by Status Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# Correlation heatmap
st.subheader("📊 Correlation Heatmap")
corr = filtered_df.select_dtypes(include='number').corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Scatter plot
st.subheader("💬 Comments vs Reactions")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x='num_comments', y='num_reactions', hue='status_type', ax=ax)
st.pyplot(fig)

# Model: Predict Reactions
st.header("🤖 Predict Number of Reactions")

# Input features and target
features = ['num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']
X = filtered_df[features]
y = filtered_df['num_reactions']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display performance
st.write(f"**Model R² Score:** {r2_score(y_test, y_pred):.2f}")

# User prediction input
st.subheader("📥 Enter Features to Predict Reactions")
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0, value=0)

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"🔮 Predicted Reactions: {int(prediction)}")

