from azure.storage.blob import BlobClient
import pandas as pd
import pickle
import azure.functions as func
from surprise import SVD
import json
import io

# URL des fichiers dans Azure Blob Storage
df_blob_url = "https://contentrecomander81f7.blob.core.windows.net/model/df_merged_compressed.pkl"
model_blob_url = "https://contentrecomander81f7.blob.core.windows.net/model/best_svd_model.pkl"

# Fonction pour obtenir des recommandations
def get_recommendations(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.params.get('user_id')
    n = int(req.params.get('n', default=5))

    try:
        # Téléchargez le fichier DataFrame depuis Azure Blob Storage
        df_blob_client = BlobClient.from_blob_url(df_blob_url)
        df_bytes = df_blob_client.download_blob().readall()
        df_merged = pd.read_pickle(io.BytesIO(df_bytes), compression="gzip")

        # Téléchargez le modèle SVD depuis Azure Blob Storage
        model_blob_client = BlobClient.from_blob_url(model_blob_url)
        model_bytes = model_blob_client.download_blob().readall()
        loaded_model = pickle.loads(model_bytes)

        # Logique de recommandation
        if user_id and user_id in df_merged['user_id'].unique():
            articles = set(df_merged['article_id'].unique())
            predictions = [loaded_model.predict(user_id, article).est for article in articles]
            top_articles = sorted(zip(articles, predictions), key=lambda x: x[1], reverse=True)[:n]
            recommended_articles = [int(article) for article, _ in top_articles]
        else:
            recommended_articles = get_popular_articles(df_merged, n)

        response_data = {
            'user_id': user_id,
            'recommendations': recommended_articles
        }

        return func.HttpResponse(
            body=json.dumps(response_data),
            mimetype='application/json'
        )
    except Exception as e:
        return func.HttpResponse(
            body=str(e),
            status_code=500
        )

# Fonction pour obtenir les articles populaires
def get_popular_articles(df, n=5):
    popular_articles = df['article_id'].value_counts().head(n).index.tolist()
    return popular_articles