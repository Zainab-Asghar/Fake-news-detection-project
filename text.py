import pickle

with open('/content/drive/MyDrive/Colab Notebooks/fake_news_detector/vectorizer', 'rb') as f:
    vectorizer = pickle.load(f)

with open('/content/drive/MyDrive/Colab Notebooks/fake_news_detector/vectorizer', 'rb') as f:
    model = pickle.load(f)

print("Model and Vectorizer loaded successfully!")
