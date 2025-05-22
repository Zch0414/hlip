import os
import requests
from tqdm import tqdm


if __name__ == "__main__":

    save_dir = "/download/rad_chestct/"
    os.makedirs(save_dir, exist_ok=True)
    
    access_token = "" # your token
    record_id = "6406114"

    r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': access_token})
    download_urls = [f['links']['self'] for f in r.json()['files']]
    filenames = [f['key'] for f in r.json()['files']]
    
    for filename, url in tqdm(zip(filenames, download_urls), total=len(filenames)):
        # print("Downloading:", filename)
        r = requests.get(url, params={'access_token': access_token})
        with open(os.path.join(save_dir, filename), 'wb') as f:
            f.write(r.content)