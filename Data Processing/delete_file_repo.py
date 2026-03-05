import os
import time
import re
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem, HfApi, login, CommitOperationDelete
import pyarrow.parquet as pq
import pyarrow.lib as plib
from tqdm import tqdm
import concurrent.futures

REPO_ID = "anhchanghoangsg/reddit_pushshift_dataset_cleaned"
MAX_WORKERS = 10
BATCH_SIZE = 100
SENSITIVE_FILE = r"D:\reddit_pushshift\sensitive_keyword.txt"
TRASH_FILE = r"D:\reddit_pushshift\trash_files.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Token not found")
    exit()

login(token=HF_TOKEN)

def get_keywords(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def is_truly_sensitive(filename_original, keyword):
    if keyword.capitalize() in filename_original:
        return True
    pattern = r"(?<![a-zA-Z])" + re.escape(keyword.lower())
    if re.search(pattern, filename_original.lower()):
        return True
    return False

def get_trash_sync_files(all_repo_files):
    if not os.path.exists(TRASH_FILE):
        return []
    with open(TRASH_FILE, 'r', encoding='utf-8') as f:
        trash_lines = [line.strip() for line in f if line.strip()]
    
    to_delete = set()
    for line in trash_lines:
        if "_comments.zst" in line:
            base = line.replace("_comments.zst", "")
            target = f"{base}_submissions_cleaned.parquet"
        elif "_submissions.zst" in line:
            base = line.replace("_submissions.zst", "")
            target = f"{base}_comments_cleaned.parquet"
        else:
            continue
        
        for repo_file in all_repo_files:
            if repo_file.endswith(target):
                path_in_repo = repo_file.replace(f"datasets/{REPO_ID}/", "")
                to_delete.add(path_in_repo)
    return list(to_delete)

def check_file_safe(file_info):
    file_path, fs = file_info
    path_in_repo = file_path.replace(f"datasets/{REPO_ID}/", "")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with fs.open(file_path, "rb") as f:
                pq.read_metadata(f)
            return None
        except (plib.ArrowInvalid, plib.ArrowIOError) as e:
            msg = str(e)
            if "Magic bytes not found" in msg or "Parquet" in msg:
                return (path_in_repo, "CORRUPTED")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return (path_in_repo, f"Unknown Error: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "500" in error_msg or "connection" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
            if attempt == max_retries - 1:
                return (path_in_repo, "Network Error")
    return None

def main():
    print("Connecting to Dataset Repo...")
    fs = HfFileSystem()
    api = HfApi(token=HF_TOKEN)
    
    print("Fetching file list...")
    path_to_scan = f"datasets/{REPO_ID}/**/*.parquet"
    try:
        all_files = fs.glob(path_to_scan)
    except Exception as e:
        print(f"Error scanning files: {e}")
        return

    if not all_files:
        print("No parquet files found in the repo.")
        return

    print(f"Found {len(all_files)} files. Starting check...")

    sync_delete_list = get_trash_sync_files(all_files)
    
    keywords = get_keywords(SENSITIVE_FILE)
    sensitive_delete_list = []
    if keywords:
        for file_path in all_files:
            filename_original = os.path.basename(file_path)
            filename_lower = filename_original.lower()
            path_in_repo = file_path.replace(f"datasets/{REPO_ID}/", "")
            
            if path_in_repo in sync_delete_list:
                continue
                
            for kw in keywords:
                if kw in filename_lower:
                    if is_truly_sensitive(filename_original, kw):
                        sensitive_delete_list.append(path_in_repo)
                        break

    files_to_check_corruption = []
    for f in all_files:
        path_in_repo = f.replace(f"datasets/{REPO_ID}/", "")
        if path_in_repo not in sync_delete_list and path_in_repo not in sensitive_delete_list:
            files_to_check_corruption.append(f)

    corrupted_delete_list = []
    if files_to_check_corruption:
        print(f"Starting structural check for {len(files_to_check_corruption)} files...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_file_safe, (f, fs)): f for f in files_to_check_corruption}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_check_corruption), unit="file"):
                result = future.result()
                if result:
                    path, reason = result
                    if reason == "CORRUPTED":
                        corrupted_delete_list.append(path)

    all_to_delete = list(set(sync_delete_list + sensitive_delete_list + corrupted_delete_list))

    print(f"Total files in repo: {len(all_files)}")
    print(f"Files to delete due to trash sync: {len(sync_delete_list)}")
    print(f"Files to delete due to sensitive keywords: {len(sensitive_delete_list)}")
    print(f"Files to delete due to corruption: {len(corrupted_delete_list)}")
    print(f"Total files to delete: {len(all_to_delete)}")

    if not all_to_delete:
        print("All files are valid, nothing to delete.")
        return

    confirm = input(f"Are you sure you want to permanently delete these {len(all_to_delete)} files? (y/n): ").strip().lower()
    
    if confirm == 'y':
        print(f"Deleting in batches (Batch size: {BATCH_SIZE} files)")
        total_deleted = 0
        for i in range(0, len(all_to_delete), BATCH_SIZE):
            batch = all_to_delete[i:i + BATCH_SIZE]
            operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
            try:
                api.create_commit(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Batch delete {len(batch)} files"
                )
                total_deleted += len(batch)
                print(f"Deleted batch {i//BATCH_SIZE + 1}. Total: {total_deleted}")
            except Exception as e:
                print(f"Error deleting batch {i//BATCH_SIZE + 1}: {e}")
        print("Completed")
    else:
        print("Deletion cancelled")

if __name__ == "__main__":
    main()