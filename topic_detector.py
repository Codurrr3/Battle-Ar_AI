from sklearn.feature_extraction.text import TfidfVectorizer

def extract_topics(text, top_k=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_k]]
