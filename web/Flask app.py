from flask import Flask, render_template, jsonify
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['road_monitoring']
collection = db['vehicle_data']

@app.route('/no_helmet')
def no_helmet():
    vehicles = list(collection.find())
    return render_template('no_helmet.html', vehicles=vehicles)

@app.route('/api/vehicles')
def get_vehicles():
    vehicles = list(collection.find({}, {'_id': 0, 'registration_number': 1, 'image_url': 1}))
    return jsonify(vehicles)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/litter')
def litter():
    vehicles = list(collection.find())
    return render_template('litter.html', vehicles=vehicles)

@app.route('/road_accident')
def road_accident():
    accidents = list(collection.find())
    return render_template('road_accident.html', accidents=accidents)
