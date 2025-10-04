"""Simple tokenizer implementation for the small language model."""

from typing import List, Dict, Optional, Union
import re
from collections import Counter


class SimpleTokenizer:
    """A simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.eos_token = "<EOS>"
        self.bos_token = "<BOS>"
        self._special_tokens = [self.pad_token, self.unk_token, self.eos_token, self.bos_token]
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on a list of texts.
        
        Args:
            texts: List of text strings to fit on
        """
        # Collect all characters
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)
        
        # Create vocabulary (most frequent chars + special tokens)
        chars = [char for char, _ in char_counter.most_common(self.vocab_size - len(self._special_tokens))]
        
        # Build mappings
        idx = 0
        for token in self._special_tokens:
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
            idx += 1
        
        for char in chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1
        
        self._is_fitted = True
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
            
        Raises:
            ValueError: If tokenizer hasn't been fitted
        """
        if not self._is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        # Add BOS and EOS tokens
        text = self.bos_token + text + self.eos_token
        
        token_ids = []
        for char in text:
            if char in self.char_to_idx:
                token_ids.append(self.char_to_idx[char])
            else:
                token_ids.append(self.char_to_idx[self.unk_token])
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            Decoded text string
            
        Raises:
            ValueError: If tokenizer hasn't been fitted
        """
        if not self._is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding")
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        text = ""
        for token_id in token_ids:
            if token_id in self.idx_to_char:
                char = self.idx_to_char[token_id]
                # Skip special tokens in output
                if char not in self._special_tokens:
                    text += char
            else:
                text += self.unk_token
        
        return text
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.char_to_idx)
    
    def get_pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self.char_to_idx[self.pad_token]
    
    def get_unk_token_id(self) -> int:
        """Get the unknown token ID."""
        return self.char_to_idx[self.unk_token]
    
    def get_eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.char_to_idx[self.eos_token]
    
    def get_bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.char_to_idx[self.bos_token]
