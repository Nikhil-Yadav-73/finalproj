from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import SentimentData
import os

def home(request):
    return render(request, "home.html")


def upload_csv(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Process CSV to extract sentiments and tickers
        # For now, we just save the dataframe as is for later processing
        for index, row in df.iterrows():
            SentimentData.objects.create(
                text=row['Text'], 
                sentiment=row['Sentiment'],
                file=uploaded_file,  # Save the file in the model
            )
        
        return JsonResponse({"message": "File uploaded successfully"})
    
    return render(request, 'upload_csv.html')
