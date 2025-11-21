"""
Dataset download and preparation script
"""
import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_nsl_kdd():
    """Download NSL-KDD dataset"""
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    files = [
        "KDDTrain+.txt",
        "KDDTest+.txt",
        "KDDTrain+_20Percent.txt"
    ]
    
    os.makedirs("data/raw/nsl-kdd", exist_ok=True)
    
    for file in files:
        url = base_url + file
        dest = f"data/raw/nsl-kdd/{file}"
        print(f"Downloading {file}...")
        download_file(url, dest)
        print(f"✓ {file} downloaded")

def download_unsw_nb15():
    """Download UNSW-NB15 dataset"""
    print("\nUNSW-NB15 Dataset:")
    print("Please download manually from:")
    print("https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("Save CSV files to: data/raw/unsw-nb15/")
    
def download_cicids2017():
    """Download CICIDS2017 dataset"""
    print("\nCICIDS2017 Dataset:")
    print("Please download manually from:")
    print("https://www.unb.ca/cic/datasets/ids-2017.html")
    print("Save CSV files to: data/raw/cicids2017/")

if __name__ == "__main__":
    print("HAMAD Dataset Downloader\n" + "="*50)
    
    # Download NSL-KDD (small, can automate)
    download_nsl_kdd()
    
    # Provide manual download instructions
    download_unsw_nb15()
    download_cicids2017()
    
    print("\n" + "="*50)
    print("Dataset download instructions completed!")
