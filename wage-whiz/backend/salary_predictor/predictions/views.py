from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
import pickle

# Create your views here.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GBR_MODEL_PATH = os.path.join(BASE_DIR, 'predictions', 'model', 'best_gbr_model.pkl')
RF_MODEL_PATH = os.path.join(BASE_DIR, 'predictions', 'model', 'rf_model.pkl')

with open(GBR_MODEL_PATH, 'rb') as file:
    gbr_model = pickle.load(file)

with open(RF_MODEL_PATH, 'rb') as file:
    rf_model = pickle.load(file)

@api_view(['POST'])
def predict_salary(request):
    try:
        print(request.data)
        model_choice = request.data.get('model_choice', 'gbr')  # Default to 'gbr'
        features = request.data.get('features', None)
        
        if features and isinstance(features, list):
            if model_choice == 'rf':
                prediction = rf_model.predict([features])
            else:
                prediction = gbr_model.predict([features])
            
            print(model_choice)
            return Response({'predicted_salary': prediction[0]}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Invalid input format or number of features'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
