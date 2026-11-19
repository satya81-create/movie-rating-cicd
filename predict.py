import joblib
import pandas as pd


def predict_rating(genre, duration, budget, popularity):
    model = joblib.load("movie_model.pkl")
    data = pd.DataFrame([{
        "Genre": genre,
        "Duration_Min": duration,
        "Budget_Crore_INR": budget,
        "Popularity_Index": popularity,
    }])
    return round(float(model.predict(data)[0]), 2)


if __name__ == "__main__":
    rating = predict_rating("Thriller", 130, 55.0, 80.0)
    print("Predicted Rating (updated):", rating)