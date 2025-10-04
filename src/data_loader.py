"""Data loading utilities for the Enron email corpus."""

import pandas as pd
import re
from typing import List, Optional, Tuple
import os


class EmailDataLoader:
    """Utility class for loading and preprocessing Enron email data."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the emails.csv file
        """
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
    
    def load_emails(self, max_emails: Optional[int] = None) -> List[str]:
        """
        Load emails from the CSV file.
        
        Args:
            max_emails: Maximum number of emails to load (None for all)
            
        Returns:
            List of email message texts
        """
        print(f"Loading emails from {self.csv_path}...")
        
        # Read CSV in chunks to handle large files
        chunk_size = 1000
        emails = []
        
        try:
            for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
                # Extract message content from the 'message' column
                messages = chunk['message'].dropna().tolist()
                emails.extend(messages)
                
                if max_emails and len(emails) >= max_emails:
                    emails = emails[:max_emails]
                    break
                    
                print(f"Loaded {len(emails)} emails so far...")
        
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: read the entire file (for smaller files)
            try:
                self.df = pd.read_csv(self.csv_path)
                messages = self.df['message'].dropna().tolist()
                emails = messages[:max_emails] if max_emails else messages
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                return []
        
        print(f"Successfully loaded {len(emails)} emails")
        return emails
    
    def preprocess_email(self, email_text: str) -> str:
        """
        Preprocess a single email text.
        
        Args:
            email_text: Raw email text
            
        Returns:
            Preprocessed email text
        """
        # Remove email headers
        email_text = self._remove_headers(email_text)
        
        # Remove HTML tags
        email_text = re.sub(r'<[^>]+>', ' ', email_text)
        
        # Remove URLs
        email_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', email_text)
        
        # Remove email addresses
        email_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', email_text)
        
        # Remove extra whitespace
        email_text = re.sub(r'\s+', ' ', email_text)
        
        return email_text.strip()
    
    def _remove_headers(self, email_text: str) -> str:
        """Remove email headers from the text."""
        # Common email header patterns
        header_patterns = [
            r'Message-ID:.*?\n',
            r'Date:.*?\n',
            r'From:.*?\n',
            r'To:.*?\n',
            r'Subject:.*?\n',
            r'X-.*?\n',
            r'Content-Type:.*?\n',
            r'Content-Transfer-Encoding:.*?\n',
            r'MIME-Version:.*?\n',
        ]
        
        for pattern in header_patterns:
            email_text = re.sub(pattern, '', email_text, flags=re.IGNORECASE)
        
        # Remove everything before the first blank line (typical header separation)
        lines = email_text.split('\n')
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '':
                body_start = i + 1
                break
        
        return '\n'.join(lines[body_start:])
    
    def load_and_preprocess(
        self, 
        max_emails: Optional[int] = None,
        min_words: int = 5
    ) -> List[str]:
        """
        Load emails and preprocess them.
        
        Args:
            max_emails: Maximum number of emails to load
            min_words: Minimum number of words required in an email
            
        Returns:
            List of preprocessed email texts
        """
        raw_emails = self.load_emails(max_emails)
        
        print("Preprocessing emails...")
        processed_emails = []
        
        for i, email in enumerate(raw_emails):
            if i % 1000 == 0:
                print(f"Preprocessed {i}/{len(raw_emails)} emails...")
            
            processed = self.preprocess_email(email)
            
            # Filter out very short emails
            word_count = len(processed.split())
            if word_count >= min_words:
                processed_emails.append(processed)
        
        print(f"Preprocessing completed. {len(processed_emails)} emails kept after filtering.")
        return processed_emails
    
    def get_email_stats(self, emails: List[str]) -> dict:
        """Get statistics about the email corpus."""
        if not emails:
            return {}
        
        total_emails = len(emails)
        total_words = sum(len(email.split()) for email in emails)
        avg_words_per_email = total_words / total_emails if total_emails > 0 else 0
        
        word_counts = [len(email.split()) for email in emails]
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        return {
            "total_emails": total_emails,
            "total_words": total_words,
            "avg_words_per_email": avg_words_per_email,
            "min_words_per_email": min_words,
            "max_words_per_email": max_words
        }
    
    def save_processed_emails(self, emails: List[str], output_path: str) -> None:
        """Save processed emails to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for email in emails:
                f.write(email + '\n\n')
        
        print(f"Processed emails saved to {output_path}")
    
    def load_processed_emails(self, input_path: str) -> List[str]:
        """Load processed emails from a text file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split on double newlines (email separator)
        emails = [email.strip() for email in content.split('\n\n') if email.strip()]
        
        print(f"Loaded {len(emails)} processed emails from {input_path}")
        return emails
