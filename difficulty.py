def estimate_difficulty(text):
    complex_words = sum(1 for w in text.split() if len(w) > 10)
    if complex_words > 200:
        return "Advanced"
    elif complex_words > 100:
        return "Intermediate"
    else:
        return "Beginner"
