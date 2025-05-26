import pandas as pd
import os

# File paths
RAW_DATA_PATH = os.path.join('data', 'Genesis GSC monthly data.xlsx')
PROCESSED_DATA_PATH = os.path.join('data', 'keywords_processed.csv')  # Or .csv if you prefer

def load_and_clean_data(raw_path):
    # Load Excel file (by default first sheet, specify if needed)
    df = pd.read_excel(raw_path)
    
    # Preview the data
    print("Original columns:", df.columns.tolist())
    print(df.head())

    # Standardize column names: lowercase, replace spaces with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Drop completely empty rows
    df.dropna(how='all', inplace=True)
    
    # Remove duplicates (exact row match)
    df.drop_duplicates(inplace=True)
    
    # Optional: Remove leading/trailing whitespace in keyword text columns
    if 'keyword' in df.columns:
        df['keyword'] = df['keyword'].astype(str).str.strip()
    
    # Optional: Convert month/year to datetime for easier handling
    if 'month' in df.columns and 'year' in df.columns:
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
    elif 'month' in df.columns:
        # Try parsing month column as datetime if possible
        df['date'] = pd.to_datetime(df['month'], errors='coerce')

    # Check for missing data summary
    print("\nMissing data per column:")
    print(df.isnull().sum())
    
    # Optionally fill or drop rows with missing essential fields
    essential_cols = ['keyword', 'clicks', 'impressions', 'average_ranking_position']
    df.dropna(subset=[col for col in essential_cols if col in df.columns], inplace=True)
    
    return df

def save_processed_data(df, out_path):
    # Save as CSV with UTF-8 encoding
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\nCleaned data saved to {out_path}")

if __name__ == '__main__':
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    save_processed_data(df_clean, PROCESSED_DATA_PATH)
