"""
Data ingestion module for CSV files with name-barcode mapping.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CSVIngester:
    """
    Handles CSV ingestion and basic data exploration for name-barcode datasets.
    """
    
    def __init__(self):
        self.data = None
        self.name_column = None
        self.barcode_column = None
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with basic error handling.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            DataFrame with loaded data
        """
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            logger.info(f"Loaded CSV with {len(self.data)} rows and {len(self.data.columns)} columns")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load CSV from {file_path}: {e}")
            raise
    
    def detect_columns(self, name_hints: List[str] = None, barcode_hints: List[str] = None) -> Tuple[str, str]:
        """
        Auto-detect name and barcode columns based on column names and content patterns.
        
        Args:
            name_hints: List of possible column names for product names
            barcode_hints: List of possible column names for barcodes
            
        Returns:
            Tuple of (name_column, barcode_column)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        # Default hints
        name_hints = name_hints or ['name', 'product', 'item', 'title', 'description', 'product_name']
        barcode_hints = barcode_hints or ['barcode', 'code', 'sku', 'id', 'product_id', 'item_id']
        
        columns = [col.lower() for col in self.data.columns]
        
        # Find name column
        name_col = None
        for hint in name_hints:
            matches = [col for col in self.data.columns if hint.lower() in col.lower()]
            if matches:
                name_col = matches[0]
                break
        
        # Find barcode column  
        barcode_col = None
        for hint in barcode_hints:
            matches = [col for col in self.data.columns if hint.lower() in col.lower()]
            if matches:
                barcode_col = matches[0]
                break
        
        # If not found by name, use heuristics
        if not name_col:
            # Look for string columns with reasonable length
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    avg_length = self.data[col].str.len().mean()
                    if 3 <= avg_length <= 100:  # Reasonable name length
                        name_col = col
                        break
        
        if not barcode_col:
            # Look for numeric or alphanumeric codes
            for col in self.data.columns:
                if col != name_col:
                    # Check if looks like codes (numbers or short alphanumeric)
                    sample = self.data[col].dropna().astype(str)
                    if len(sample) > 0:
                        avg_length = sample.str.len().mean()
                        if 3 <= avg_length <= 20:  # Reasonable barcode length
                            barcode_col = col
                            break
        
        if not name_col or not barcode_col:
            available_cols = list(self.data.columns)
            raise ValueError(f"Could not auto-detect name and barcode columns. Available columns: {available_cols}")
        
        self.name_column = name_col
        self.barcode_column = barcode_col
        
        logger.info(f"Detected columns - Name: '{name_col}', Barcode: '{barcode_col}'")
        return name_col, barcode_col
    
    def get_basic_stats(self) -> Dict:
        """
        Generate basic statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if not self.name_column or not self.barcode_column:
            self.detect_columns()
        
        stats = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'name_column': self.name_column,
            'barcode_column': self.barcode_column,
            'unique_names': self.data[self.name_column].nunique(),
            'unique_barcodes': self.data[self.barcode_column].nunique(),
            'missing_names': self.data[self.name_column].isna().sum(),
            'missing_barcodes': self.data[self.barcode_column].isna().sum(),
            'duplicate_barcodes': len(self.data) - self.data[self.barcode_column].nunique(),
            'sample_names': self.data[self.name_column].dropna().head(10).tolist(),
            'sample_barcodes': self.data[self.barcode_column].dropna().head(10).tolist()
        }
        
        return stats
    
    def print_stats(self):
        """Print formatted statistics about the dataset."""
        stats = self.get_basic_stats()
        
        print("=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Total columns: {stats['total_columns']}")
        print(f"Name column: '{stats['name_column']}'")
        print(f"Barcode column: '{stats['barcode_column']}'")
        print()
        print(f"Unique names: {stats['unique_names']:,}")
        print(f"Unique barcodes: {stats['unique_barcodes']:,}")
        print(f"Missing names: {stats['missing_names']:,}")
        print(f"Missing barcodes: {stats['missing_barcodes']:,}")
        print(f"Duplicate barcodes: {stats['duplicate_barcodes']:,}")
        print()
        print("Sample names:")
        for name in stats['sample_names']:
            print(f"  - {name}")
        print()
        print("Sample barcodes:")
        for barcode in stats['sample_barcodes']:
            print(f"  - {barcode}")
        print("=" * 50)
    
    def get_clean_data(self) -> pd.DataFrame:
        """
        Return cleaned dataset with only name and barcode columns, removing missing values.
        
        Returns:
            DataFrame with clean name-barcode pairs
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if not self.name_column or not self.barcode_column:
            self.detect_columns()
        
        # Create clean dataset
        clean_data = self.data[[self.name_column, self.barcode_column]].copy()
        clean_data = clean_data.dropna()
        
        # Standardize column names
        clean_data.columns = ['name', 'barcode']
        
        # Convert to string and strip whitespace
        clean_data['name'] = clean_data['name'].astype(str).str.strip()
        clean_data['barcode'] = clean_data['barcode'].astype(str).str.strip()
        
        # Remove empty strings
        clean_data = clean_data[(clean_data['name'] != '') & (clean_data['barcode'] != '')]
        
        logger.info(f"Cleaned data: {len(clean_data)} rows remaining")
        return clean_data


def demo_ingestion(file_path: str):
    """
    Demo function showing how to use the CSV ingester.
    
    Args:
        file_path: Path to CSV file to analyze
    """
    ingester = CSVIngester()
    
    print(f"Loading CSV from: {file_path}")
    ingester.load_csv(file_path)
    
    print("\nDetecting name and barcode columns...")
    ingester.detect_columns()
    
    print("\nDataset statistics:")
    ingester.print_stats()
    
    print("\nCleaned data preview:")
    clean_data = ingester.get_clean_data()
    print(clean_data.head(10))
    
    return clean_data


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        demo_ingestion(file_path)
    else:
        print("Usage: python ingest.py <csv_file_path>")
