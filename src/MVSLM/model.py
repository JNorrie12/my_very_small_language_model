"""Core model implementation for the small language model."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class SmallLanguageModel(nn.Module):
    """A very small language model for experimentation."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_length: int = 256,
    ):
        """
        Initialize the small language model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Logits [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Position indices
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb
        
        # Create causal mask if no attention mask provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_length, device)
        
        # Transformer forward pass
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def _create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).expand(seq_length, -1, -1)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting sequence [batch_size, seq_length]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated sequence [batch_size, seq_length + max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature  # Get last token logits
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, _ = torch.topk(logits, top_k)
                    logits[logits < top_k_logits[:, -1:]] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
