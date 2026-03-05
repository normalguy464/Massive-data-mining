import qbittorrentapi
import os
import sys
import re  
from huggingface_hub import HfApi
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "anhchanghoangsg/reddit_pushshift_dataset_cleaned"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN NOT FOUND IN .env FILE")
    sys.exit(1)

CONN_INFO = dict(
    host="localhost",
    port=8080,
    username="admin",
    password="123456"
)

NSFW_LIST_PATH = r"D:\reddit_pushshift\list_of_nsfw_reddit.txt"
TRASH_LIST_PATH = r"D:\reddit_pushshift\trash_files.txt"  
SENSITIVE_LIST_PATH = r"D:\reddit_pushshift\sensitive_keyword.txt"
MIN_SIZE_BYTES = 1 * 1024 * 1024 

def is_truly_sensitive(filename_original, keyword):
    if keyword.capitalize() in filename_original:
        return True
        
    pattern = r"(?<![a-zA-Z])" + re.escape(keyword.lower())
    if re.search(pattern, filename_original.lower()):
        return True
        
    return False

try:
    qbt_client = qbittorrentapi.Client(**CONN_INFO)
    qbt_client.auth_log_in()
except Exception as e:
    print(f"Error connection {e}")
    sys.exit(1)

def load_keywords(path, is_sensitive=False):
    if not os.path.exists(path):
        print(f"Warning: List not found: {path}")
        return set() if not is_sensitive else []

    print(f"Reading list: {path}")
    try:
        if is_sensitive:
            keywords = []
            with open(path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    word = line.strip().lower()
                    if word: keywords.append(word)
        else:
            keywords = set()
            with open(path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    word = line.strip().lower()
                    if word: keywords.add(word)
                    
        print(f"Loaded {len(keywords)} entries/keywords")
        return keywords
    except Exception as e:
        print(f"Error reading list: {e}")
        return set() if not is_sensitive else []

def load_trash_files(path):
    trash_set = set()
    if not os.path.exists(path):
        print(f"Note: Trash file list not found (First run?): {path}")
        return trash_set

    print(f"Reading trash list: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                fname = line.strip()
                if not fname: continue
                trash_set.add(fname)
        print(f"Loaded {len(trash_set)} known trash files")
    except Exception as e:
        print(f"Error reading trash list: {e}")
    return trash_set

def get_base_name_from_hf(filename):
    fname = os.path.basename(filename)
    return fname.replace('_cleaned.parquet', '').replace('.parquet', '').lower()

def get_existing_hf_files():
    print("Scanning Hugging Face repository...")
    api = HfApi(token=HF_TOKEN)
    hf_files = set()
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        for f in files:
            if f.endswith(".parquet") and "pushshift_" in f:
                base_name = get_base_name_from_hf(f)
                hf_files.add(base_name)
        print(f"Found {len(hf_files)} existing files on Hugging Face")
        return hf_files
    except Exception as e:
        print(f"Error checking Hugging Face: {e}")
        return set()

def main_filter():
    torrents = qbt_client.torrents_info()
    if not torrents:
        print("No torrents found")
        return
    
    torrent_hash = torrents[0].hash
    torrent_name = torrents[0].name
    print(f"Processing Torrent: {torrent_name}")
    
    print("Downloading file list from qBittorrent...")
    files_info = qbt_client.torrents_files(torrent_hash)
    
    nsfw_set = load_keywords(NSFW_LIST_PATH, is_sensitive=False)
    sensitive_list = load_keywords(SENSITIVE_LIST_PATH, is_sensitive=True) 
    hf_existing_set = get_existing_hf_files()
    trash_set = load_trash_files(TRASH_LIST_PATH)
    
    skip_indices = [] 
    download_indices = []
    
    total_size = 0
    skip_size = 0
    download_size = 0
    
    count_size_filtered = 0 
    count_blacklist_filtered = 0 
    count_sensitive_filtered = 0 
    count_hf_filtered = 0
    count_trash_filtered = 0
    
    print(f"Scanning {len(files_info)} files...")

    count = 0
    for file in files_info:
        count += 1
        
        full_path = file.get('name')
        if not full_path:
            full_path = "None"

        fsize = file['size']
        total_size += fsize
        
        should_skip = False
        
        fname_only_original = os.path.basename(full_path) 
        fname_lower = fname_only_original.lower()

        if fsize < MIN_SIZE_BYTES:
            should_skip = True
            count_size_filtered += 1
        else:
            subreddit_name = fname_lower.replace('_submissions.zst', '').replace('_comments.zst', '')
            exact_filename = fname_lower.replace('.zst', '')

            if fname_only_original in trash_set:
                should_skip = True
                count_trash_filtered += 1
  
            elif subreddit_name in nsfw_set:
                should_skip = True
                count_blacklist_filtered += 1
    
            elif exact_filename in hf_existing_set:
                should_skip = True
                count_hf_filtered += 1
    
            else:
                for kw in sensitive_list:
                    if kw in fname_lower:
                        if is_truly_sensitive(fname_only_original, kw):
                            should_skip = True
                            count_sensitive_filtered += 1
                            break 
        
        if should_skip:
            skip_indices.append(file['index'])
            skip_size += fsize
        else:
            download_indices.append(file['index'])
            download_size += fsize
            
        if count % 10000 == 0:
            print(f"Scanned {count}/{len(files_info)} files")

    print("Updating priorities in qBittorrent...")
    
    if skip_indices:
        chunk_size = 50000
        for i in range(0, len(skip_indices), chunk_size):
            chunk = skip_indices[i:i + chunk_size]
            qbt_client.torrents_file_priority(torrent_hash, file_ids=chunk, priority=0)
        print(f"Blocked/Skipped {len(skip_indices)} files")
        
    if download_indices:
        chunk_size = 50000
        for i in range(0, len(download_indices), chunk_size):
            chunk = download_indices[i:i + chunk_size]
            qbt_client.torrents_file_priority(torrent_hash, file_ids=chunk, priority=1)
        print(f"Marked {len(download_indices)} files as safe/new")

    gb = 1024**3
    print(f"Total files: {len(files_info)}")
    print(f"Filtered by size (< 1MB): {count_size_filtered}")
    print(f"Filtered by Trash List: {count_trash_filtered}")
    print(f"Filtered by NSFW Blacklist: {count_blacklist_filtered}")
    print(f"Filtered by Sensitive Keyword: {count_sensitive_filtered}")
    print(f"Filtered by Hugging Face (Exists): {count_hf_filtered}")
    print(f"Valid files to download: {len(download_indices)}")
    print(f"Skipped size: {skip_size / gb:.2f} GB")
    print(f"Download size: {download_size / gb:.2f} GB")

if __name__ == "__main__":
    main_filter()