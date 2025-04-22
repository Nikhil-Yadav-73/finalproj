from django.db import models
from django.contrib.auth.models import User

class SentimentData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='sentiments/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

    def __str__(self):
        return f"Sentiment Data {self.id} by {self.user.username}"