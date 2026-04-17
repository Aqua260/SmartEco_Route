from flask import Flask, render_template, request, jsonify
from eco_route_engine import EcoRouteEngine, VEHICLE_TYPES, TRAFFIC_LEVELS, ROAD_TYPES

app = Flask(__name__)

# Initialize and train the AI engine once when the server starts
engine = EcoRouteEngine()
print("🤖 Training AI Engine with synthetic data...")
train_data = engine.generate_training_data(n=3000)
engine.train(train_data)
print("✅ Engine ready. Regression metrics:", engine.metrics)

@app.route('/')
def index():
    """Renders the main dashboard."""
    return render_template('index.html', 
                           vehicle_types=VEHICLE_TYPES, 
                           traffic_levels=TRAFFIC_LEVELS, 
                           road_types=ROAD_TYPES)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles route analysis and returns the eco-friendly recommendation."""
    try:
        data = request.get_json() or {}
        vehicle = data.get('vehicle')
        model_name = data.get('model', 'decision_tree')
        routes = data.get('routes', [])
        start_location = data.get('start_location', 'Current Location')
        destination = data.get('destination', 'Destination')

        # Basic validation
        if not routes:
            return jsonify({"error": "No routes provided"}), 400

        # Run the AI recommendation engine
        result = engine.recommend_eco_route(routes, vehicle, model_name, start_location, destination)
        
        # Calculate actual CO2 savings relative to the shortest route
        rec = result['recommended']
        sho = result['shortest_route']
        # Ensure we don't subtract from different types if multiple routes exist
        co2_saved = round(max(0.0, float(sho['predicted_emission_g']) - float(rec['predicted_emission_g'])), 1)

        return jsonify({
            "start_location": result['start_location'],
            "destination": result['destination'],
            "recommended": rec,
            "all_routes": result['all_routes'],
            "co2_saved_g": max(0, co2_saved),
            "eco_win": result['eco_win'],
            "model_used": result['model_used']
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "Could not calculate emissions. Check input values."}), 500

if __name__ == '__main__':
    app.run(debug=True)