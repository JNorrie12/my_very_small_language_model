"""Frequency-based language model implementation."""

from typing import List, Dict, Tuple, Optional, Counter
import re
import random
from collections import defaultdict, Counter
import json
import pickle


class FrequencyModel:
    """A frequency-based language model that predicts the next word based on X previous words."""
    
    def __init__(self, context_size: int = 2):
        """
        Initialize the frequency model.
        
        Args:
            context_size: Number of previous words to use for prediction (X)
        """
        self.context_size = context_size
        self.ngram_counts: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocabulary: set = set()
        self.total_ngrams = 0
        
    def train(self, texts: List[str]) -> None:
        """
        Train the model on a list of texts.
        
        Args:
            texts: List of text strings to train on
        """
        print(f"Training frequency model with context size {self.context_size}...")
        
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                print(f"Processing text {i}/{len(texts)}")
            
            # Tokenize text
            words = self._tokenize(text)
            
            # Build n-grams
            for j in range(len(words) - self.context_size):
                context = tuple(words[j:j + self.context_size])
                next_word = words[j + self.context_size]
                
                self.ngram_counts[context][next_word] += 1
                self.vocabulary.add(next_word)
                self.total_ngrams += 1
        
        print(f"Training completed. Total n-grams: {self.total_ngrams}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Number of contexts: {len(self.ngram_counts)}")
    
    def predict_next_word(self, context: List[str], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Predict the next word given a context.
        
        Args:
            context: List of previous words (length should be context_size)
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples, sorted by probability
        """
        if len(context) != self.context_size:
            raise ValueError(f"Context must have exactly {self.context_size} words")
        
        context_tuple = tuple(context)
        
        if context_tuple not in self.ngram_counts:
            # Return uniform distribution over vocabulary if context not seen
            total_words = len(self.vocabulary)
            if total_words == 0:
                return []
            
            prob = 1.0 / total_words
            words = list(self.vocabulary)
            return [(word, prob) for word in words[:top_k]]
        
        # Get counts for this context
        word_counts = self.ngram_counts[context_tuple]
        total_count = sum(word_counts.values())
        
        if total_count == 0:
            return []
        
        # Calculate probabilities
        predictions = []
        for word, count in word_counts.items():
            prob = count / total_count
            predictions.append((word, prob))
        
        # Sort by probability (descending) and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def generate_text(self, seed: List[str], length: int = 50) -> List[str]:
        """
        Generate text starting from a seed context.
        
        Args:
            seed: Initial context words (length should be context_size)
            length: Number of words to generate
            
        Returns:
            List of generated words
        """
        if len(seed) != self.context_size:
            raise ValueError(f"Seed must have exactly {self.context_size} words")
        
        generated = list(seed)
        current_context = seed
        
        for _ in range(length):
            # Predict next word
            predictions = self.predict_next_word(current_context, top_k=1)
            
            if not predictions:
                # If no prediction available, sample from vocabulary
                if self.vocabulary:
                    next_word = random.choice(list(self.vocabulary))
                else:
                    break
            else:
                next_word = predictions[0][0]
            
            generated.append(next_word)
            
            # Update context (sliding window)
            current_context = current_context[1:] + [next_word]
        
        return generated
    
    def get_context_stats(self) -> Dict[str, int]:
        """Get statistics about the trained model."""
        return {
            "context_size": self.context_size,
            "vocabulary_size": len(self.vocabulary),
            "total_ngrams": self.total_ngrams,
            "unique_contexts": len(self.ngram_counts),
            "avg_words_per_context": self.total_ngrams / len(self.ngram_counts) if self.ngram_counts else 0
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file."""
        model_data = {
            "context_size": self.context_size,
            "ngram_counts": dict(self.ngram_counts),
            "vocabulary": list(self.vocabulary),
            "total_ngrams": self.total_ngrams
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.context_size = model_data["context_size"]
        self.ngram_counts = defaultdict(Counter, model_data["ngram_counts"])
        self.vocabulary = set(model_data["vocabulary"])
        self.total_ngrams = model_data["total_ngrams"]
        
        print(f"Model loaded from {filepath}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens (words)
        """
        # Convert to lowercase and split on whitespace
        text = text.lower()
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter out empty strings and very short words
        words = [word for word in words if len(word) > 1]
        
        return words
    
    def get_most_common_contexts(self, n: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        """Get the most common contexts in the training data."""
        context_counts = [(context, sum(counts.values())) for context, counts in self.ngram_counts.items()]
        context_counts.sort(key=lambda x: x[1], reverse=True)
        return context_counts[:n]
    
    def get_most_common_words(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the most common words in the training data."""
        word_counts = Counter()
        for counts in self.ngram_counts.values():
            word_counts.update(counts)
        
        return word_counts.most_common(n)
