from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import pandas as pd
from .models import SentimentData
import os
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.utils import timezone
import spacy
from django.conf import settings

nlp = spacy.load("en_core_web_sm")

TICKER_CSV_PATH = '/home/nikhil/Desktop/finalproj/Yahoo-Finance-Ticker-Symbols.csv'
ticker_df = pd.read_csv(TICKER_CSV_PATH)
ticker_df['Ticker'] = ticker_df['Ticker'].astype(str).str.strip().str.upper()
valid_tickers = set(ticker_df['Ticker'])

def extract_tickers_spacy(text):
    doc = nlp(str(text))
    ents = [ent.text.upper() for ent in doc.ents if ent.label_ == "ORG"]
    return [t for t in ents if t in valid_tickers]

def home(request):
    return render(request, "home.html")

@login_required
def profile(request):
    uploaded_files = SentimentData.objects.filter(user=request.user)

    return render(request, 'profile.html', {
        'uploaded_files': uploaded_files
    })

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = password

        if password == confirm_password:
            try:
                user = User.objects.create_user(username=username, password=password)
                login(request, user)
                messages.success(request, "Account created successfully!")
                return redirect('home')
            except Exception as e:
                messages.error(request, f"Error creating account: {e}")
        else:
            messages.error(request, "Passwords do not match. Please try again.")

    return render(request, 'signup.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password. Please try again.")

    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('home')

@login_required
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            f = request.FILES['file']
            username = request.user.username
            base, ext = os.path.splitext(f.name)
            ts = timezone.now().strftime("%Y%m%d%H%M%S")
            new_name = f"{username}_{base}_{ts}{ext}"

            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'sentiments'),
                                   base_url=os.path.join(settings.MEDIA_URL, 'sentiments'))
            filename = fs.save(new_name, f)
            full_path = fs.path(filename)

            df = pd.read_csv(full_path)
            df['Tickers'] = df['Text'].apply(extract_tickers_spacy)
            df = df[df['Tickers'].map(len) > 0]
            df.to_csv(full_path, index=False)

            # Create a SentimentData instance
            sd = SentimentData.objects.create(
                user=request.user,
                file=os.path.join('sentiments', filename)
            )

            # Add success message to be shown on the redirected page
            messages.success(request, "File uploaded and processed successfully!")
            
            return JsonResponse({
                'success': True,
                'sentiment_data_id': sd.id,
                'message': "File uploaded and processed successfully."
            })

        except Exception as e:
            print(f"Error occurred while uploading: {str(e)}")
            # Add error message and redirect back to the upload page
            messages.error(request, f"An error occurred while uploading: {str(e)}")
            return redirect('upload_csv')

    # Render the upload page if not POST or if file is not provided
    return render(request, 'upload_csv.html')

@login_required
def analyzed_results(request, sentiment_data_id):
    sd = get_object_or_404(SentimentData, id=sentiment_data_id, user=request.user)
    sentiment_df = pd.read_csv(sd.file.path)

    ticker_df = pd.read_csv(TICKER_CSV_PATH)
    ticker_df.rename(columns={'Category Name': 'Category_Name'}, inplace=True)

    exploded = sentiment_df.explode('Tickers').rename(columns={'Tickers': 'Ticker'})
    
    merged = exploded.merge(ticker_df, on='Ticker', how='left')

    agg = merged.groupby('Ticker').agg(
        total_sentiment=('Sentiment', 'sum'),
        avg_sentiment=('Sentiment', 'mean'),
        mention_count=('Sentiment', 'count')
    ).reset_index()

    top5 = agg.nlargest(5, 'total_sentiment').merge(ticker_df, on='Ticker')

    low5 = agg.nsmallest(5, 'avg_sentiment').merge(ticker_df, on='Ticker')

    most_mentioned = (
        agg.nlargest(5, 'mention_count')
           [['Ticker','mention_count']]
           .to_dict('records')
    )

    if 'Category_Name' in merged.columns:
        top_by_category = (
            merged.groupby('Category_Name')['Sentiment']
                  .mean()
                  .nlargest(5)
                  .to_dict()
        )
    else:
        top_by_category = {}

    if 'Date' in merged.columns:
        sentiment_trends = merged.groupby('Date')['Sentiment'].mean().to_dict()
    else:
        sentiment_trends = {}

    return render(request, 'analyzed_results.html', {
        'top_profitable_companies': enumerate(top5.to_dict('records'), start=1),
        'lowest_sentiment_companies': enumerate(low5.to_dict('records'), start=1),
        'most_mentioned_tickers': most_mentioned,
        'top_sentiment_by_category': top_by_category,
        'sentiment_trends_over_time': sentiment_trends,
    })