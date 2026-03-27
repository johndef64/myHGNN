import os

if os.getcwd().endswith('src/downloaders'):
    os.chdir('../..')

# get this dataset from:
url = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"

# download and extract to dataset/drkg/
import os
from pathlib import Path
import tarfile
import requests
DATA_DIR = Path("dataset/drkg")
DATA_DIR.mkdir(parents=True, exist_ok=True)
def download_and_extract(url: str, extract_path: Path) -> None:
    filename = url.split("/")[-1]
    file_path = extract_path / filename

    # Download the file
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

    # Extract the tar.gz file
    print(f"Extracting {filename}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted {filename}")

    # Remove the tar.gz file after extraction
    os.remove(file_path)
    print(f"Removed {filename}")

download_and_extract(url, DATA_DIR)
print("DRKG dataset is ready in 'dataset/drkg/' directory.")
