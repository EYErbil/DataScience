"""
Text normalization utilities for product names.
Handles multilingual text processing, cleaning, and standardization.
"""
import re
import unicodedata
import logging
from typing import List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Comprehensive text normalization for product names.
    Handles case normalization, accent removal, stop words, and basic cleaning.
    """
    
    def __init__(self, language: str = 'en', custom_stop_words: Optional[Set[str]] = None):
        """
        Initialize normalizer with language-specific settings.
        
        Args:
            language: ISO language code (en, tr, es, fr, etc.)
            custom_stop_words: Additional stop words to remove
        """
        self.language = language
        self.stop_words = self._get_stop_words(language)
        
        if custom_stop_words:
            self.stop_words.update(custom_stop_words)
    
    def _get_stop_words(self, language: str) -> Set[str]:
        """Get stop words for specified language."""
        # Basic stop words for common languages
        stop_words_dict = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
            'tr': {'ve', 'ile', 'için', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'mi', 'mu', 'mü'},
            'es': {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'en', 'con', 'para', 'de', 'del'},
            'fr': {'le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'dans', 'sur', 'avec', 'pour', 'de', 'du'},
            'de': {'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'in', 'auf', 'mit', 'für', 'von'},
            'it': {'il', 'la', 'lo', 'gli', 'le', 'un', 'una', 'e', 'o', 'ma', 'in', 'su', 'con', 'per', 'di'}
        }
        
        return stop_words_dict.get(language, set())
    
    def remove_accents(self, text: str) -> str:
        """
        Remove accents and diacritical marks from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with accents removed
        """
        # Normalize to NFD (decomposed form)
        normalized = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        without_accents = ''.join(char for char in normalized 
                                if unicodedata.category(char) != 'Mn')
        return without_accents
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: remove special characters, normalize whitespace.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s\-\.]', '', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[\-\.]{2,}', '', text)
        
        return text.strip()
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stop words removed
        """
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def normalize_case(self, text: str) -> str:
        """
        Normalize text case (lowercase by default).
        
        Args:
            text: Input text
            
        Returns:
            Case-normalized text
        """
        return text.lower()
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common abbreviations in product names.
        
        Args:
            text: Input text
            
        Returns:
            Text with abbreviations expanded
        """
        # Common abbreviations mapping
        abbreviations = {
            'tbl': 'table',
            'dsk': 'desk',
            'chr': 'chair',
            'bk': 'book',
            'comp': 'computer',
            'pc': 'piece',
            'set': 'set',
            'lg': 'large',
            'sm': 'small',
            'med': 'medium',
            'std': 'standard',
            'delx': 'deluxe',
            'prof': 'professional',
            'off': 'office',
            'hm': 'home',
            'w/': 'with',
            '&': 'and'
        }
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check exact match
            if word.lower() in abbreviations:
                expanded_words.append(abbreviations[word.lower()])
            # Check without punctuation
            elif word.lower().rstrip('.,!?') in abbreviations:
                expanded_words.append(abbreviations[word.lower().rstrip('.,!?')])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def normalize(self, text: str, 
                 remove_accents: bool = True,
                 clean: bool = True,
                 remove_stops: bool = True,
                 expand_abbrev: bool = True,
                 normalize_case: bool = True) -> str:
        """
        Apply full normalization pipeline to text.
        
        Args:
            text: Input text
            remove_accents: Whether to remove accents
            clean: Whether to clean special characters
            remove_stops: Whether to remove stop words
            expand_abbrev: Whether to expand abbreviations
            normalize_case: Whether to normalize case
            
        Returns:
            Fully normalized text
        """
        if not isinstance(text, str):
            text = str(text)
        
        result = text
        
        # Apply normalization steps in order
        if normalize_case:
            result = self.normalize_case(result)
        
        if remove_accents:
            result = self.remove_accents(result)
        
        if clean:
            result = self.clean_text(result)
        
        if expand_abbrev:
            result = self.expand_abbreviations(result)
        
        if remove_stops:
            result = self.remove_stop_words(result)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result.strip())
        
        return result
    
    def normalize_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Normalize a batch of texts.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments passed to normalize()
            
        Returns:
            List of normalized texts
        """
        return [self.normalize(text, **kwargs) for text in texts]


class MultilingualNormalizer:
    """
    Handles normalization across multiple languages with language detection.
    """
    
    def __init__(self):
        """Initialize multilingual normalizer."""
        self.normalizers = {}
        
    def get_normalizer(self, language: str) -> TextNormalizer:
        """
        Get or create normalizer for specified language.
        
        Args:
            language: ISO language code
            
        Returns:
            TextNormalizer instance for the language
        """
        if language not in self.normalizers:
            self.normalizers[language] = TextNormalizer(language)
        return self.normalizers[language]
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        For production use, consider using langdetect or similar libraries.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Simple heuristics for common languages
        if any(char in text for char in 'çğıöşüÇĞIÖŞÜ'):
            return 'tr'  # Turkish
        elif any(char in text for char in 'áéíóúñüÁÉÍÓÚÑÜ'):
            return 'es'  # Spanish
        elif any(char in text for char in 'àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ'):
            return 'fr'  # French
        elif any(char in text for char in 'äöüßÄÖÜ'):
            return 'de'  # German
        elif any(char in text for char in 'àáéèíìóòúù'):
            return 'it'  # Italian
        else:
            return 'en'  # Default to English
    
    def normalize_multilingual(self, text: str, language: Optional[str] = None) -> str:
        """
        Normalize text with automatic or specified language detection.
        
        Args:
            text: Input text
            language: Language code (if None, will be auto-detected)
            
        Returns:
            Normalized text
        """
        if language is None:
            language = self.detect_language(text)
        
        normalizer = self.get_normalizer(language)
        return normalizer.normalize(text)


def create_synonym_groups() -> dict:
    """
    Create predefined synonym groups for common product categories.
    
    Returns:
        Dictionary mapping canonical terms to their synonyms
    """
    return {
        'table': ['table', 'desk', 'masa', 'bureau', 'escritorio', 'tavolo', 'tisch'],
        'chair': ['chair', 'seat', 'sandalye', 'silla', 'chaise', 'sedia', 'stuhl'],
        'book': ['book', 'kitap', 'libro', 'livre', 'buch'],
        'computer': ['computer', 'pc', 'bilgisayar', 'ordenador', 'ordinateur', 'computer'],
        'phone': ['phone', 'telefon', 'teléfono', 'téléphone', 'telefono'],
        'lamp': ['lamp', 'light', 'lamba', 'lámpara', 'lampe', 'lampada'],
        'cabinet': ['cabinet', 'cupboard', 'dolap', 'armario', 'armoire', 'armadio'],
        'bed': ['bed', 'yatak', 'cama', 'lit', 'letto', 'bett'],
        'sofa': ['sofa', 'couch', 'koltuk', 'sofá', 'canapé', 'divano'],
        'mirror': ['mirror', 'ayna', 'espejo', 'miroir', 'specchio', 'spiegel']
    }


def map_to_canonical(text: str, synonym_groups: dict) -> str:
    """
    Map a normalized text to its canonical form using synonym groups.
    
    Args:
        text: Normalized text
        synonym_groups: Dictionary of canonical -> synonyms mapping
        
    Returns:
        Canonical form if found, otherwise original text
    """
    text_lower = text.lower().strip()
    
    for canonical, synonyms in synonym_groups.items():
        if text_lower in [syn.lower() for syn in synonyms]:
            return canonical
    
    return text


def demo_normalization():
    """Demo function showing text normalization capabilities."""
    # Sample multilingual product names
    sample_names = [
        "MacBook Pro 13\"",
        "MASA - Çalışma Masası",
        "Silla de Oficina Ergonómica",
        "Chaise de Bureau Noire",
        "Tisch für Büro",
        "Gaming Chr w/ RGB",
        "Schreibtisch Standard",
        "Lámpara LED para Escritorio"
    ]
    
    print("=" * 60)
    print("TEXT NORMALIZATION DEMO")
    print("=" * 60)
    
    # Single language normalizer
    normalizer_en = TextNormalizer('en')
    normalizer_tr = TextNormalizer('tr')
    
    # Multilingual normalizer
    ml_normalizer = MultilingualNormalizer()
    
    # Synonym groups
    synonyms = create_synonym_groups()
    
    print("\nOriginal → Normalized (Single Language) → Multilingual → Canonical")
    print("-" * 80)
    
    for name in sample_names:
        # Single language normalization (English)
        norm_en = normalizer_en.normalize(name)
        
        # Multilingual normalization
        norm_ml = ml_normalizer.normalize_multilingual(name)
        
        # Map to canonical
        canonical = map_to_canonical(norm_ml, synonyms)
        
        print(f"{name:<25} → {norm_en:<20} → {norm_ml:<15} → {canonical}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_normalization()
