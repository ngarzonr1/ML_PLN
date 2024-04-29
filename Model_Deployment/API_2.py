#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from m02_model_deployment import predict_price
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Price Prediction API',
    description='Price Prediction API')

ns = api.namespace('predict', 
     description='Prices Classifier')
   
parser = api.parser()

parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='Mileage of the car', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price(args['Mileage'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
