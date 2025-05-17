import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# ------------------------------
# Sample Data Generator
# ------------------------------
def generate_sample_data():
    data = {
        'venue': ['Mumbai', 'Delhi', 'Chennai', 'Mumbai', 'Delhi'],
        'last_5_scores': [22345, 15234, 10000, 23452, 13451],
        'opponent_rank': [2, 3, 1, 4, 2],
        'dream11_score': [75, 45, 30, 80, 60]
    }
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print("✅ Sample dataset generated: data/raw_data.csv")

# ------------------------------
# Load and preprocess the data
# ------------------------------
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df['venue_encoded'] = df['venue'].astype('category').cat.codes
    df['form'] = df['last_5_scores'].apply(lambda x: sum(map(int, str(x))) / 5 if isinstance(x, int) else 0)
    return df

# ------------------------------
# Train ML Model
# ------------------------------
def train_model(df):
    X = df[['venue_encoded', 'form', 'opponent_rank']]
    y = df['dream11_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"🎯 Model trained. MSE: {mse:.2f}")
    
    return model

# ------------------------------
# Make Predictions
# ------------------------------
def predict_score(model, new_data):
    prediction = model.predict(new_data)
    return prediction

# ------------------------------
# Main function
# ------------------------------
def main():
    print("🏏 Sports Betting Forecast System Starting...")

    if not os.path.exists('data/raw_data.csv'):
        generate_sample_data()
    
    df = load_and_clean_data('data/raw_data.csv')
    model = train_model(df)
    joblib.dump(model, 'model.pkl')
    print("✅ Model saved as model.pkl")

    # Prediction example
    example_input = pd.DataFrame([{
        'venue_encoded': 1,
        'form': 45,
        'opponent_rank': 2
    }])
    prediction = predict_score(model, example_input)
    print(f"📈 Predicted Dream11 Score: {prediction[0]:.2f}")

# Entry point
if __name__ == "__main__":
    main()
