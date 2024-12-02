import json
from typing import Dict, List
from pathlib import Path

class CharTokenizer:
    def __init__(self):
        pass
    
    def fit(self, data: str) -> None:
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.character_to_index = {char: idx for idx, char in enumerate(self.chars)}
        self.index_to_character = {idx: char for idx, char in enumerate(self.chars)}    
        self.tokenizer = {
            "chars": self.chars,
            "vocab_size": self.vocab_size,
            "character_to_index": self.character_to_index,
            "index_to_character": self.index_to_character
        }
        
    def save(self, file: Path) -> None:
        with open(file, "w") as f:
            json.dump(self.tokenizer, f)
    
    def load(self, file: Path) -> None:
        with open(file, "r") as f:
            json.loads(self.tokenizer, f)
    
    def encode(self, x: List[str]) -> List[int]:
        return [self.character_to_index[char] for char in x]
    
    def decode(self, x: List[int]) -> List[str]:
        return [self.index_to_character[char] for char in x]