import pandas as pd
import numpy as np
import sqlite3
import faiss
from openai import OpenAI
from typing import Dict, List, Set, Tuple
import pickle
import os
from pathlib import Path

class CSVEmbeddingsProcessor:
    def __init__(self, openai_api_key: str, db_path: str = "data.db", faiss_index_path: str = "embeddings.index"):
        """Initialize the CSV processor with OpenAI API key and storage paths"""
        self.client = OpenAI(api_key = openai_api_key)
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.embedding_dimension = 1536  # OpenAI embedding dimension
        self.category_mappings = {}  # Maps category text to index in FAISS
        self.index_to_category = {}  # Maps FAISS index to category text
        
    def identify_categorical_columns(self, df: pd.DataFrame, max_unique_ratio: float = 0.1) -> List[str]:
        """Identify categorical columns in the dataframe"""
        categorical_cols = []
        
        for col in df.columns:
            # skip numeric columns and date columns that don't need embeddings
            if col in ['order_id', 'cost', 'date']:
                continue
                
            # check if column has reasonable number of unique values for categorical data
            unique_ratio = df[col].nunique() / len(df)
            
            # check data type - strings are likely categorical
            is_string_type = df[col].dtype == 'object'
            
            if unique_ratio <= max_unique_ratio or (is_string_type and df[col].nunique() <= 50):
                categorical_cols.append(col)
                
        return categorical_cols
    
    def extract_unique_categories(self, df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Set[str]]:
        """Extract all unique categories from categorical columns"""
        unique_categories = {}
        
        for col in categorical_columns:
            # Get unique values, excluding NaN
            unique_vals = df[col].dropna().unique()
            unique_categories[col] = set(str(val) for val in unique_vals)
            
        return unique_categories
    
    def generate_embeddings(self, categories: Dict[str, Set[str]]) -> Dict[str, np.ndarray]:
        """Generate OpenAI embeddings for all unique categories"""
        embeddings = {}
        all_categories = []
        
        # collect all unique category texts across all columns
        for col, cats in categories.items():
            for cat in cats:
                category_text = f"{col}: {cat}"  # Prefix with column name for context
                all_categories.append(category_text)
        
        print(f"Generating embeddings for {len(all_categories)} unique categories...")
        
        # generate embeddings in batches to handle rate limits
        batch_size = 100
        for i in range(0, len(all_categories), batch_size):
            batch = all_categories[i:i+batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input = batch,
                    model = "text-embedding-ada-002"
                )
                
                for j, category_text in enumerate(batch):
                    embeddings[category_text] = np.array(response.data[j].embedding)
                    
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                continue
        
        return embeddings
    
    def create_faiss_index(self, embeddings: Dict[str, np.ndarray]) -> faiss.IndexFlatIP:
        """Create and populate FAISS index with embeddings"""
        # create FAISS index for inner product similarity
        index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # prepare embeddings matrix and mappings
        embedding_matrix = []
        idx = 0
        
        for category_text, embedding in embeddings.items():
            # normalize embedding for cosine similarity
            normalized_embedding = embedding / np.linalg.norm(embedding)
            embedding_matrix.append(normalized_embedding)
            
            self.category_mappings[category_text] = idx
            self.index_to_category[idx] = category_text
            idx += 1
        
        # add embeddings to index
        embedding_matrix = np.array(embedding_matrix, dtype = np.float32)
        index.add(embedding_matrix)
        
        return index
    
    def create_sqlite_table(self, df: pd.DataFrame, table_name: str = "csv_data"):
        """Create SQLite table and insert all CSV data"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # create table and insert data
            df.to_sql(table_name, conn, if_exists = 'replace', index = False)
            print(f"Inserted {len(df)} rows into SQLite table '{table_name}'")
            
        except Exception as e:
            print(f"Error creating SQLite table: {e}")
        finally:
            conn.close()
    
    def save_faiss_index(self, index: faiss.IndexFlatIP):
        """Save FAISS index and category mappings to disk"""
        #save FAISS index
        faiss.write_index(index, self.faiss_index_path)
        
        # save category mappings
        mappings_path = self.faiss_index_path.replace('.index', '_mappings.pkl')
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'category_mappings': self.category_mappings,
                'index_to_category': self.index_to_category
            }, f)
    
    def load_faiss_index(self) -> faiss.IndexFlatIP:
        """Load FAISS index and category mappings from disk"""
        # load FAISS index
        index = faiss.read_index(self.faiss_index_path)
        
        # load category mappings
        mappings_path = self.faiss_index_path.replace('.index', '_mappings.pkl')
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.category_mappings = mappings['category_mappings']
            self.index_to_category = mappings['index_to_category']
        
        return index
    
    def similarity_search(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform similarity search on FAISS index"""
        try:
            # Load index if not already loaded
            if not os.path.exists(self.faiss_index_path):
                raise FileNotFoundError("FAISS index not found. Run process_csv first.")
            
            index = self.load_faiss_index()
            
            # generate embedding for query
            response = self.client.embeddings.create(
                input = [query_text],
                model="text-embedding-ada-002"
            )
            query_embedding = np.array(response.data[0].embedding)
            
            # normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # search
            similarities, indices = index.search(query_embedding, k)
            
            # return results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx in self.index_to_category:
                    category_text = self.index_to_category[idx]
                    results.append((category_text, float(similarity)))
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def process_csv(self, csv_file_path: str, table_name: str = "csv_data"):
        """Main function to process CSV file: extract categories, generate embeddings,create FAISS index, and store data in SQLite"""
        print(f"Processing CSV file: {csv_file_path}")
        
        # load CSV
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # identify categorical columns
        categorical_columns = self.identify_categorical_columns(df)
        print(f"Identified categorical columns: {categorical_columns}")
        
        # extract unique categories
        unique_categories = self.extract_unique_categories(df, categorical_columns)
        total_categories = sum(len(cats) for cats in unique_categories.values())
        print(f"Found {total_categories} unique categories across {len(categorical_columns)} columns")
        
        # create SQLite table (always do this regardless of embedding success)
        self.create_sqlite_table(df, table_name)
        
        # generate embeddings
        embeddings = self.generate_embeddings(unique_categories)
        print(f"Generated {len(embeddings)} embeddings")
        
        if len(embeddings) > 0:
            # create FAISS index
            index = self.create_faiss_index(embeddings)
            print(f"Created FAISS index with {index.ntotal} vectors")
            
            #save FAISS index
            self.save_faiss_index(index)
        else:
            print("No embeddings generated - skipping FAISS index creation")
        
        print("Processing complete!")
        return {
            'categorical_columns': categorical_columns,
            'unique_categories': unique_categories,
            'total_embeddings': len(embeddings),
            'total_rows_stored': len(df)
        }


def main():
    """Example usage of the CSVEmbeddingsProcessor"""
    # Initialize processor (replace with your OpenAI API key)
    processor = CSVEmbeddingsProcessor(
        openai_api_key = "your-openai-api-key-here",
        db_path = "sample_data.db",
        faiss_index_path = "sample_embeddings.index"
    )
    
    # process the CSV file
    results = processor.process_csv("sample.csv")
    print(f"Processing results: {results}")
    
    # example similarity search
    print("\n--- Example Similarity Searches ---")
    
    search_queries = [
        "broken item",
        "audio device", 
        "computer store",
        "refund approved"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = processor.similarity_search(query, k = 3)
        for category, score in results:
            print(f"  {category} (similarity: {score:.3f})")


if __name__ == "__main__":
    main()
