import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

# Constants
TRAFFIC_LEVELS = ["low", "medium", "high"]
ROAD_TYPES = ["highway", "urban", "suburban", "rural"]
VEHICLE_TYPES = ["electric", "hybrid", "petrol", "diesel", "truck"]
MODEL_DT = "decision_tree"
MODEL_LR = "linear_regression"


class EcoRouteEngine:
    """
    An engine to train models and recommend eco-friendly routes based on vehicle type,
    road conditions, and traffic.
    """

    def __init__(self):
        self.traffic_enc = LabelEncoder().fit(TRAFFIC_LEVELS)
        self.road_enc = LabelEncoder().fit(ROAD_TYPES)
        self.vehicle_enc = LabelEncoder().fit(VEHICLE_TYPES)
        self.models = {}
        self.metrics = {}

    def generate_training_data(self, n: int = 2000, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)

        vehicle_emission_base = {"electric": 0, "hybrid": 60, "petrol": 130, "diesel": 120, "truck": 280}
        traffic_mult = {"low": 1.0, "medium": 1.35, "high": 1.80}
        road_mult = {"highway": 0.85, "urban": 1.30, "suburban": 1.0, "rural": 1.05}

        data = []
        for i in range(n):
            distance = np.random.uniform(2, 120)
            traffic = np.random.choice(TRAFFIC_LEVELS)
            road = np.random.choice(ROAD_TYPES)
            vehicle = np.random.choice(VEHICLE_TYPES)
            traffic_lights = np.random.randint(0, 15)
            obstacles = np.random.randint(0, 4)

            emission = (
                distance
                * vehicle_emission_base[vehicle]
                * traffic_mult[traffic]
                * road_mult[road]
                + (traffic_lights * 5.5)
                + (obstacles * 15.0)
                + np.random.normal(0, 5)
            )

            data.append({
                "name": f"Route {i+1}",
                "distance_km": distance,
                "traffic": traffic,
                "road_type": road,
                "vehicle": vehicle,
                "traffic_lights": traffic_lights,
                "obstacles": obstacles,
                "emission_g": max(0, emission),
            })
        
        df = pd.DataFrame(data)
        df["traffic"] = self.traffic_enc.transform(df["traffic"])
        df["road_type"] = self.road_enc.transform(df["road_type"])
        df["vehicle"] = self.vehicle_enc.transform(df["vehicle"])
        df["emission_g"] = df["emission_g"].round(2)

        return df

    def train(self, data: pd.DataFrame):
        features = ["distance_km", "traffic", "road_type", "vehicle", "traffic_lights", "obstacles"]
        target = "emission_g"
        
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        # Linear Regression
        lr = LinearRegression().fit(X_train, y_train)
        self.models[MODEL_LR] = lr
        self.metrics[MODEL_LR] = {
            "name": "Linear Regression",
            "rmse": round(np.sqrt(mean_squared_error(y_test, lr.predict(X_test))), 2),
            "r2": round(r2_score(y_test, lr.predict(X_test)), 4)
        }

        # Decision Tree
        dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=7).fit(X_train, y_train)
        self.models[MODEL_DT] = dt
        self.metrics[MODEL_DT] = {
            "name": "Decision Tree",
            "rmse": round(np.sqrt(mean_squared_error(y_test, dt.predict(X_test))), 2),
            "r2": round(r2_score(y_test, dt.predict(X_test)), 4)
        }

    def _preprocess_route(self, route: Dict[str, Any], vehicle_type: str) -> List[float]:
        distance = float(route.get("distance_km", route.get("dist", 0)))
        t = self.traffic_enc.transform([route["traffic"]])[0]
        r = self.road_enc.transform([route["road_type"]])[0]
        v = self.vehicle_enc.transform([vehicle_type])[0]
        lights = int(route.get("traffic_lights", 0))
        obs = int(route.get("obstacles", 0))
        return [distance, float(t), float(r), float(v), float(lights), float(obs)]

    def recommend_eco_route(self, routes: List[Dict[str, Any]], vehicle_type: str, model_name: str = MODEL_DT, 
                           start_location: str = "Start", destination: str = "Destination") -> Dict[str, Any]:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model_label = self.metrics[model_name]["name"]
        
        results = []
        for route in routes:
            features = self._preprocess_route(route, vehicle_type)
            pred = model.predict([features])[0]

            processed_route = {**route}
            processed_route["distance_km"] = features[0]
            processed_route["predicted_emission_g"] = round(max(0, float(pred)), 1)
            results.append(processed_route)

        results.sort(key=lambda x: (x["predicted_emission_g"], x["distance_km"]))

        best = results[0]
        shortest = min(results, key=lambda x: x["distance_km"])

        return {
            "start_location": start_location,
            "destination": destination,
            "vehicle": vehicle_type,
            "model_used": model_label,
            "recommended": best,
            "shortest_route": shortest,
            "eco_win": best["name"] != shortest["name"],
            "all_routes": results
        }