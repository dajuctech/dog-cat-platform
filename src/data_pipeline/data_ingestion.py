import os
import zipfile
import gdown

def download_from_google_drive(drive_url, local_zip_path):
    """
    Download a file from Google Drive using gdown.
    """
    print(f"Downloading from Google Drive: {drive_url} to {local_zip_path}...")
    gdown.download(drive_url, local_zip_path, quiet=False)
    print("Download complete!")

def extract_zip(local_zip_path, extract_to):
    """
    Extract the downloaded zip file.
    """
    print(f"Extracting {local_zip_path} to {extract_to}...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def run_data_ingestion_google_drive():
    """
    Main ingestion function for Google Drive source.
    """
    drive_url = 'https://drive.google.com/uc?id=1OnsD-agt_U7s46bFtM-mO6Uc2QNVkS1S'
    local_zip_path = 'data/raw/dogs-vs-cats-vvsmall.zip'
    extract_to = 'data/processed/dogs-vs-cats-vvsmall'
    
    os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)
    
    download_from_google_drive(drive_url, local_zip_path)
    extract_zip(local_zip_path, extract_to)
    print("Data ingestion from Google Drive completed!")

if __name__ == "__main__":
    run_data_ingestion_google_drive()
