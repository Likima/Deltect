from data.api import fetch_clinvar_deletions
from data.data_processor import process_api_data
from data.preprocessing import load_dataframe

def main():
    raw_data = fetch_clinvar_deletions()

    processed_data = process_api_data(raw_data)

    # 
    df = load_dataframe(processed_data)
    print(df.head())

if __name__ == "__main__":
    main()