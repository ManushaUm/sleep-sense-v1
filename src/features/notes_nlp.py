import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec

# Simulated raw notes CSV path
NOTES_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "notes.csv"

def generate_simulated_notes_if_missing():
    """
    If the notes file doesn't exist, we generate simulated diary notes for all users
    based on their activity, stress, and phone usage metrics to ground the text in real behaviors.
    """
    if NOTES_CSV_PATH.exists():
        return
        
    print("Generating simulated user notes for NLP training...")
    # We will read some raw EMAs and activity to synthesize notes realistically
    from src.data.loader import get_all_users, load_ema_sleep, load_ema_stress
    from src.features.activity_features import extract_activity_features
    from src.features.phonelock_features import extract_phonelock_features
    
    users = get_all_users()
    notes_data = []
    
    for uid in users:
        try:
            df_sleep = load_ema_sleep(uid)
            if df_sleep.empty:
                continue
            for _, row in df_sleep.iterrows():
                date_str = row['date']
                
                # Extract some behaviors to dynamically build note text
                act = extract_activity_features(uid, date_str)
                lock = extract_phonelock_features(uid, date_str)
                
                # Synthesize text note phrases
                phrases = []
                
                # 1. Caffeine trigger
                # We'll tie caffeine search keyword simulation to late night phone usage or stress
                if lock.get('unlock_count_late_night', 0) > 3:
                    phrases.append("drank two cups of strong coffee in the evening to stay alert")
                    phrases.append("searched online about caffeine side effects late at night")
                else:
                    phrases.append("drank hot chamomile tea in the afternoon")
                    
                # 2. Activity / Exercise
                if act.get('walking_minutes', 0) > 30:
                    phrases.append("went for a long walk around campus and felt active")
                elif act.get('stationary_ratio', 0) > 0.85:
                    phrases.append("sat in the library all day studying and felt stationary")
                    
                # 3. Screens
                if lock.get('last_unlock_hour', 0) >= 23.0:
                    phrases.append("scrolled social media apps and stared at my phone screen in bed")
                else:
                    phrases.append("turned off all screens early to wind down")
                    
                # 4. Stress
                if act.get('exercise_detected', 0):
                    phrases.append("did vigorous running exercise to relieve exam stress")
                else:
                    phrases.append("felt stressed about the upcoming midterm exams and homework")
                    
                note_text = ". ".join(phrases) + "."
                notes_data.append({
                    'user_id': uid,
                    'date': date_str,
                    'note_text': note_text
                })
        except Exception:
            continue
            
    # Save simulated notes to raw data directory
    NOTES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(notes_data).to_csv(NOTES_CSV_PATH, index=False)
    print(f"Simulated notes written to {NOTES_CSV_PATH}")

def preprocess_text(text: str) -> list[str]:
    """Preprocess text: lowercase, remove punctuation, and filter out common stopwords."""
    text = text.lower()
    words = re.findall(r'[a-z]+', text)
    
    # Common English stopwords
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
        'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'a', 'an', 'the', 
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
        'will', 'just', 'don', 'should', 'now', 'felt', 'went', 'about', 'did', 'about'
    }
    return [w for w in words if w not in stopwords]

class NotesNLPExtractor:
    """
    Fits a Word2Vec embedding model on user notes and extracts daily NLP features
    by computing similarity scores between note text and key concept words.
    """
    def __init__(self, vector_size: int = 16, window: int = 3):
        self.vector_size = vector_size
        self.window = window
        self.model = None
        self.notes_df = None
        
    def fit_word2vec(self):
        """Train a Word2Vec model on the corpus of all user notes."""
        try:
            generate_simulated_notes_if_missing()
            if not NOTES_CSV_PATH.exists():
                print("Warning: notes.csv does not exist. Skipping Word2Vec training.")
                return
                
            try:
                self.notes_df = pd.read_csv(NOTES_CSV_PATH)
            except pd.errors.EmptyDataError:
                print("Warning: notes.csv is empty. Skipping Word2Vec training.")
                return
                
            if self.notes_df is None or self.notes_df.empty or 'note_text' not in self.notes_df.columns:
                print("Warning: No note_text column or data found in notes.csv. Skipping Word2Vec training.")
                return
                
            sentences = [preprocess_text(str(note)) for note in self.notes_df['note_text'].values]
            if not sentences or all(len(s) == 0 for s in sentences):
                print("Warning: No words found for training Word2Vec. Skipping.")
                return
                
            # Train Word2Vec
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=1,
                workers=1,
                seed=42
            )
            print("Trained Word2Vec embedding model successfully.")
        except Exception as e:
            print(f"Warning: Failed to train Word2Vec model during startup: {e}")

    def get_document_embedding(self, words: list[str]) -> np.ndarray:
        """Calculate average word embedding for a list of words."""
        if not self.model or not words:
            return np.zeros(self.vector_size)
            
        wv = self.model.wv
        valid_vectors = [wv[w] for w in words if w in wv]
        
        if not valid_vectors:
            return np.zeros(self.vector_size)
            
        return np.mean(valid_vectors, axis=0)

    def extract_nlp_features(self, user_id: str, date: str) -> dict:
        """
        Extract daily NLP similarity features for a user and date.
        """
        if self.notes_df is None:
            if NOTES_CSV_PATH.exists():
                self.notes_df = pd.read_csv(NOTES_CSV_PATH)
            else:
                generate_simulated_notes_if_missing()
                if NOTES_CSV_PATH.exists():
                    self.notes_df = pd.read_csv(NOTES_CSV_PATH)
                else:
                    return {
                        'nlp_caffeine_similarity': 0.0,
                        'nlp_screen_similarity': 0.0,
                        'nlp_stress_similarity': 0.0
                    }
                    
        # Find note for user/date
        day_note = self.notes_df[(self.notes_df['user_id'] == user_id) & (self.notes_df['date'] == date)]
        if day_note.empty:
            return {
                'nlp_caffeine_similarity': 0.0,
                'nlp_screen_similarity': 0.0,
                'nlp_stress_similarity': 0.0
            }
            
        note_text = str(day_note.iloc[0]['note_text'])
        words = preprocess_text(note_text)
        doc_emb = self.get_document_embedding(words)
        
        if np.all(doc_emb == 0.0) or self.model is None:
            return {
                'nlp_caffeine_similarity': 0.0,
                'nlp_screen_similarity': 0.0,
                'nlp_stress_similarity': 0.0
            }
            
        # Helper to compute cosine similarity
        def cosine_similarity(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
            
        wv = self.model.wv
        
        # Sleep-disruptive concept vectors (anchored on words from Word2Vec vocab)
        caffeine_vec = wv['coffee'] if 'coffee' in wv else np.zeros(self.vector_size)
        screen_vec = wv['phone'] if 'phone' in wv else np.zeros(self.vector_size)
        stress_vec = wv['stress'] if 'stress' in wv else np.zeros(self.vector_size)
        
        return {
            'nlp_caffeine_similarity': cosine_similarity(doc_emb, caffeine_vec),
            'nlp_screen_similarity': cosine_similarity(doc_emb, screen_vec),
            'nlp_stress_similarity': cosine_similarity(doc_emb, stress_vec)
        }
