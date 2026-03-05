import time
import qbittorrentapi
import os
import zstandard as zstd
import orjson
import io
import threading
import tkinter as tk
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import shutil
import sys
import gc
from tqdm import tqdm

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN NOT FOUND IN .env FILE")

CONN_INFO = dict(
    host="localhost",
    port=8080,
    username="admin",
    password="123456"
)

TARGET_BATCH_SIZE_GB = 20
TARGET_BATCH_SIZE_BYTES = TARGET_BATCH_SIZE_GB * 1024**3
MAX_DISK_QUEUE_GB = 50
MAX_DISK_QUEUE_BYTES = MAX_DISK_QUEUE_GB * 1024**3
MAX_FILES_PER_HF_FOLDER = 9000
REPO_ID = "anhchanghoangsg/reddit_pushshift_dataset_cleaned"
UPLOAD_BATCH_SIZE = 50
MAX_PENDING_SIZE_GB = 10
TEMP_UPLOAD_DIR = r"D:\qBit_files_tmp"
TRASH_LIST_PATH = r"D:\reddit_pushshift\trash_files.txt" 
NUM_THREADS = 1 

class AutoCloseDialog(tk.Toplevel):
    def __init__(self, parent, title, message, timeout_seconds=30):
        super().__init__(parent)
        self.title(title)
        self.result = True
        self.timeout = timeout_seconds
        self.parent = parent
        
        try:
            x = parent.winfo_x() + (parent.winfo_width() // 2) - 200
            y = parent.winfo_y() + (parent.winfo_height() // 2) - 100
        except:
            x = 100
            y = 100
            
        self.geometry(f"400x180+{x}+{y}")
        self.resizable(False, False)
        
        lbl_msg = ttk.Label(self, text=message, wraplength=380, justify="center", font=("Arial", 10))
        lbl_msg.pack(pady=20)
        
        self.lbl_timer = ttk.Label(self, text=f"Auto-continue in: {self.timeout}s", foreground="red")
        self.lbl_timer.pack(pady=5)
        
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)
        
        btn_yes = ttk.Button(btn_frame, text="Continue Now", command=self.on_yes)
        btn_yes.pack(side="left", padx=10)
        
        btn_no = ttk.Button(btn_frame, text="Stop", command=self.on_no)
        btn_no.pack(side="left", padx=10)
        
        self.after(1000, self.countdown)
        
        self.transient(parent)
        self.grab_set()
        self.parent.wait_window(self)

    def countdown(self):
        self.timeout -= 1
        if self.timeout <= 0:
            self.result = True
            self.destroy()
        else:
            self.lbl_timer.config(text=f"Auto-continue in: {self.timeout}s")
            self.after(1000, self.countdown)

    def on_yes(self):
        self.result = True
        self.destroy()

    def on_no(self):
        self.result = False
        self.destroy()

class DataProcessor:
    def __init__(self, app_context, log_left_callback, log_right_callback, progress_callback, disk_callback, hf_progress_callback):
        self.app = app_context 
        self.log_left = log_left_callback
        self.log_right = log_right_callback
        self.update_progress = progress_callback
        self.update_disk = disk_callback
        self.update_hf_progress = hf_progress_callback
        self.is_running = False
        self.is_paused = False
        self.qbt_client = None
        self.hf_api = HfApi(token=HF_TOKEN)
        self.current_folder_index = 1
        self.current_folder_file_count = 0
        
        self.current_torrent_hash = None
        self.save_path = ""
        self.pending_uploads = [] 
        self.current_pending_size = 0
        self.processed_files_indices = set()
        self.safe_file_indices = set()
        
        self.lock = threading.Lock() 
        self.deletion_queue = set() 
        
        trash_dir = os.path.dirname(TRASH_LIST_PATH)
        if not os.path.exists(trash_dir):
            try: os.makedirs(trash_dir)
            except: pass
        
        self.init_hf_state()

    def connect(self):
        try:
            self.qbt_client = qbittorrentapi.Client(**CONN_INFO)
            self.qbt_client.auth_log_in()
            self.log_right("qBittorrent connection successful.")
            print("qBittorrent connection successful.")
            return True
        except Exception as e:
            self.log_right(f"qBittorrent connection error: {e}")
            print(f"qBittorrent connection error: {e}")
            return False

    def init_hf_state(self):
        try:
            if HF_TOKEN:
                login(token=HF_TOKEN)
                files = self.hf_api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
                folder_counts = {}
                for f in files:
                    parts = f.split('/')
                    if len(parts) > 1 and parts[0].startswith("pushshift_"):
                        try:
                            idx = int(parts[0].split('_')[1])
                            folder_counts[idx] = folder_counts.get(idx, 0) + 1
                        except ValueError: pass
                
                if not folder_counts:
                    self.current_folder_index = 1
                    self.current_folder_file_count = 0
                else:
                    max_idx = max(folder_counts.keys())
                    count = folder_counts[max_idx]
                    if count >= MAX_FILES_PER_HF_FOLDER:
                        self.current_folder_index = max_idx + 1
                        self.current_folder_file_count = 0
                    else:
                        self.current_folder_index = max_idx
                        self.current_folder_file_count = count
                self.log_right(f"HF State: Folder pushshift_{self.current_folder_index}, Files: {self.current_folder_file_count}")
                print(f"HF State: Folder pushshift_{self.current_folder_index}, Files: {self.current_folder_file_count}")
        except Exception as e:
            self.log_right(f"Error initializing HF state: {e}")
            print(f"Error initializing HF state: {e}")

    def format_date(self, timestamp):
        try:
            if timestamp is None: return None
            return datetime.utcfromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
        except: return None

    def has_font_error(self, text):
        if not isinstance(text, str): return True 
        return '\ufffd' in text

    def safe_int(self, value, default=0):
        try:
            if value is None: return default
            return int(value)
        except: return default

    def process_submission_row(self, data):
        author = data.get('author')
        if author in ('[deleted]', '[removed]', None): return None
        
        name = data.get('name')
        if name is None: return None

        title = data.get('title')
        if self.has_font_error(title): return None

        selftext = data.get('selftext')
        if selftext in ('[deleted]', '[removed]'): 
            return None
        if not selftext: 
            selftext = "None"

        clean_data = {}
        clean_data['author'] = author
        clean_data['created_utc'] = self.format_date(data.get('created_utc'))
        clean_data['crosspost_parent'] = data.get('crosspost_parent') if data.get('crosspost_parent') is not None else "None"
        clean_data['domain'] = data.get('domain') if data.get('domain') is not None else "None"
        clean_data['num_comments'] = self.safe_int(data.get('num_comments'), 0)
        clean_data['num_crossposts'] = self.safe_int(data.get('num_crossposts'), 0)
        clean_data['score'] = self.safe_int(data.get('score'), 0)
        clean_data['name'] = name
        clean_data['subreddit'] = data.get('subreddit')
        clean_data['subreddit_id'] = data.get('subreddit_id')
        clean_data['subreddit_subscribers'] = self.safe_int(data.get('subreddit_subscribers'), 0)
        clean_data['title'] = title
        clean_data['selftext'] = selftext
        
        uv_ratio = data.get('upvote_ratio')
        try: clean_data['upvote_ratio'] = float(uv_ratio) if uv_ratio is not None else 1.0
        except: clean_data['upvote_ratio'] = 1.0
        
        return clean_data

    def process_comment_row(self, data):
        author = data.get('author')
        if author in ('[deleted]', '[removed]', None): return None
        
        body = data.get('body')
        if body in ('[deleted]', '[removed]', None): return None
        
        l_id = data.get('link_id')
        if l_id is None: return None
        l_id = str(l_id) 

        name = data.get('name')
        if name is None: return None

        p_id = data.get('parent_id')
        if p_id is None: return None
        if not isinstance(p_id, str): p_id = l_id

        clean_data = {}
        clean_data['author'] = author
        clean_data['body'] = body
        clean_data['created_utc'] = self.format_date(data.get('created_utc'))
        clean_data['controversiality'] = self.safe_int(data.get('controversiality'), 0)
        clean_data['score'] = self.safe_int(data.get('score'), 0)
        clean_data['name'] = name
        clean_data['parent_id'] = p_id
        clean_data['subreddit'] = data.get('subreddit')
        clean_data['subreddit_id'] = data.get('subreddit_id')
        clean_data['link_id'] = l_id
        
        return clean_data

    def process_clean_zst(self, input_path):
        thread_name = threading.current_thread().name
        filename = os.path.basename(input_path)
        if "_submissions" in filename: ftype = "SUBMISSION"
        elif "_comments" in filename: ftype = "COMMENT"
        else: return None

        if not os.path.exists(TEMP_UPLOAD_DIR):
            try: os.makedirs(TEMP_UPLOAD_DIR)
            except: pass
            
        output_filename = filename.replace('.zst', '_cleaned.parquet')
        output_path = os.path.join(TEMP_UPLOAD_DIR, output_filename)
        
        dctx = zstd.ZstdDecompressor()
        
        batch_data = []
        BATCH_SIZE = 2000000 
        writer = None
        reader = None
        
        CHUNK_SIZE = 128 * 1024 * 1024 
        buffer = b""

        try:
            file_size = os.path.getsize(input_path)
            
            with open(input_path, 'rb') as ifh:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"[{thread_name}] {filename}", leave=False) as pbar:
                    with dctx.stream_reader(ifh) as reader:
                        while True:
                            chunk = reader.read(CHUNK_SIZE)
                            
                            if not chunk:
                                break

                            pbar.update(ifh.tell() - pbar.n)
                            chunk = buffer + chunk
                            lines = chunk.split(b'\n')
                            buffer = lines.pop()

                            for line in lines:
                                if not line: continue
                                try:
                                    row = orjson.loads(line)
                                    
                                    cleaned_row = None
                                    if ftype == "SUBMISSION": cleaned_row = self.process_submission_row(row)
                                    else: cleaned_row = self.process_comment_row(row)
                                    
                                    if cleaned_row:
                                        batch_data.append(cleaned_row)
                                    
                                    if len(batch_data) >= BATCH_SIZE:
                                        df = pd.DataFrame(batch_data)
                                        table = pa.Table.from_pandas(df)
                                        if writer is None:
                                            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                                        writer.write_table(table)
                                        batch_data = [] 
                                except: continue
                            
                            if not self.is_running: break

                        if buffer:
                            try:
                                row = orjson.loads(buffer)
                                cleaned_row = None
                                if ftype == "SUBMISSION": cleaned_row = self.process_submission_row(row)
                                else: cleaned_row = self.process_comment_row(row)
                                if cleaned_row:
                                    batch_data.append(cleaned_row)
                            except: pass

                if batch_data:
                    df = pd.DataFrame(batch_data)
                    table = pa.Table.from_pandas(df)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                    writer.write_table(table)
            
            if writer:
                writer.close()
                del dctx, reader, writer
                gc.collect() 
                print(f"[{thread_name}] Cleaned: {os.path.basename(output_path)}")
                return output_path
            else:
                del dctx, reader, writer
                gc.collect()
                if os.path.exists(output_path):
                    try: os.remove(output_path)
                    except: pass
                print(f"[{thread_name}] Finished {filename}: 100% (No valid data)")
                return None 

        except Exception as e:
            self.log_left(f"Parquet conversion error: {e}")
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except: pass
            return None

    def initialize_queue(self, torrent_hash):
        self.log_right("Scanning files to freeze queue...")
        print("Scanning files to freeze queue...")
        files = self.qbt_client.torrents_files(torrent_hash)
        
        reset_ids = []
        self.safe_file_indices = set()
        
        for f in files:
            if f['priority'] > 0:
                self.safe_file_indices.add(f['index'])
                if f['progress'] < 1:
                    reset_ids.append(f['index'])
        
        if reset_ids:
            self.log_right(f"Freezing {len(reset_ids)} files to Priority 0.")
            print(f"Freezing {len(reset_ids)} files to Priority 0.")
            self.qbt_client.torrents_file_priority(torrent_hash, file_ids=reset_ids, priority=0)
        else:
            self.log_right("No pending files needed freezing.")
            print("No pending files needed freezing.")
            
        return len(self.safe_file_indices) > 0

    def flush_pending_uploads(self):
        with self.lock:
            if not self.pending_uploads: return
            files_to_upload_tuples = self.pending_uploads[:]
            self.pending_uploads = [] 
            self.current_pending_size = 0
        
        count = len(files_to_upload_tuples)
        self.log_left(f"Batch Uploading {count} files...")
        print(f"\nBatch Uploading {count} files...")
        
        if not os.path.exists(TEMP_UPLOAD_DIR):
            os.makedirs(TEMP_UPLOAD_DIR)
        
        uploaded_parquet_paths = [] 
        
        try:
            for parquet_path, zst_path, f_idx in files_to_upload_tuples:
                if not os.path.exists(parquet_path): continue
                
                if self.current_folder_file_count >= MAX_FILES_PER_HF_FOLDER:
                    self.current_folder_index += 1
                    self.current_folder_file_count = 0
                    self.log_right(f"Switching to new folder: pushshift_{self.current_folder_index}")
                    print(f"Switching to new folder: pushshift_{self.current_folder_index}")

                uploaded_parquet_paths.append(parquet_path)
                self.current_folder_file_count += 1

            if not uploaded_parquet_paths: return

            remote_folder = f"pushshift_{self.current_folder_index}"
            

            upload_success = False
            max_retries = 10
            
            for attempt in range(max_retries):
                try:
                    self.hf_api.upload_folder(
                        folder_path=TEMP_UPLOAD_DIR,
                        path_in_repo=remote_folder,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        commit_message=f"Batch upload {len(uploaded_parquet_paths)} files"
                    )
                    upload_success = True
                except Exception as e:
                    self.log_left(f"Upload failed (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        self.log_left("Retrying in 15 seconds...")
                        print(f"Upload failed. Retrying in 15 seconds...")
                        time.sleep(15)
                    else:
                        self.log_left("Max retries reached. Returning files to queue.")
                        print("Max retries reached. Returning files to queue.")

            if not upload_success:
                with self.lock:
                    self.pending_uploads.extend(files_to_upload_tuples)
                return

            self.log_left(f"Batch upload success!")
            print(f"Batch upload success!")
            self.update_hf_progress(f"Folder {self.current_folder_index}: {self.current_folder_file_count}")

            self.log_left("Adding uploaded files to cleanup queue...")
            for parquet_path, zst_path, f_idx in files_to_upload_tuples:
                try: 
                    if os.path.exists(parquet_path): os.remove(parquet_path)
                except: pass
                
                try: self.qbt_client.torrents_file_priority(self.current_torrent_hash, file_ids=f_idx, priority=0)
                except: pass
                
                with self.lock:
                    self.deletion_queue.add(zst_path)

        except Exception as e:
            self.log_left(f"Batch processing error: {e}")
            print(f"Batch processing error: {e}")
            with self.lock:
                self.pending_uploads.extend(files_to_upload_tuples)
        
        finally:
            if os.path.exists(TEMP_UPLOAD_DIR):
                for filename in os.listdir(TEMP_UPLOAD_DIR):
                    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        pass

    def janitor_loop(self):
        self.log_right("Delete thread started ")
        while self.is_running:
            pending_files = []
            with self.lock:
                if self.deletion_queue:
                    pending_files = list(self.deletion_queue)
            
            if not pending_files:
                time.sleep(5)
                continue
            
            for file_path in pending_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        self.log_left(f"Deleted: {os.path.basename(file_path)}")
                        print(f"Deleted: {os.path.basename(file_path)}")
                        with self.lock:
                            self.deletion_queue.discard(file_path)
                    else:
                        with self.lock:
                            self.deletion_queue.discard(file_path)
                except PermissionError:
                    pass
                except Exception as e:
                    pass
            
            time.sleep(5)

    def producer_loop(self):
        while self.is_running:
            if self.is_paused:
                time.sleep(1)
                continue

            try:
                files = self.qbt_client.torrents_files(self.current_torrent_hash)
                
                pending_disk_size = 0
                active_downloads_size = 0
                candidate_files = []
                
                with self.lock:
                    safe_indices_copy = self.safe_file_indices.copy()
                    processed_indices_copy = self.processed_files_indices.copy()

                for f in files:
                    if f['progress'] == 1 and f['index'] not in processed_indices_copy and f['index'] in safe_indices_copy:
                        pending_disk_size += f['size']
                    elif f['priority'] > 0 and f['progress'] < 1:
                        active_downloads_size += f['size']
                    elif f['priority'] == 0 and f['progress'] < 1 and f['index'] not in processed_indices_copy and f['index'] in safe_indices_copy:
                        candidate_files.append(f)

                if pending_disk_size > MAX_DISK_QUEUE_BYTES:
                    self.log_right(f"Disk buffer full ({pending_disk_size / 1024**3:.2f} GB). Pausing downloads.")
                    print(f"\rDisk buffer full ({pending_disk_size / 1024**3:.2f} GB). Pausing downloads.", end="")
                    time.sleep(5)
                    continue

                if active_downloads_size > 0:
                    time.sleep(2)
                    total_downloaded = sum(f['progress'] * f['size'] for f in files if f['priority'] > 0)
                    total_active_size = sum(f['size'] for f in files if f['priority'] > 0)
                    percent = (total_downloaded / total_active_size * 100) if total_active_size > 0 else 0
                    dl_gb = total_downloaded / (1024**3)
                    total_gb = total_active_size / (1024**3)
                    self.update_progress(percent, f"Downloading: {percent:.1f}% ({dl_gb:.2f}/{total_gb:.2f} GB)")
                    continue

                if candidate_files:
                    new_batch_ids = []
                    acc_size = 0
                    
                    for f in candidate_files:
                        new_batch_ids.append(f['index'])
                        acc_size += f['size']
                        if acc_size >= TARGET_BATCH_SIZE_BYTES: break
                    
                    if new_batch_ids:
                        self.qbt_client.torrents_file_priority(self.current_torrent_hash, file_ids=new_batch_ids, priority=7)
                        
                        with self.lock:
                            self.safe_file_indices.update(new_batch_ids)
                        
                        self.log_right(f"Added {len(new_batch_ids)} files to download queue.")
                        print(f"\nStarting NEW BATCH: {len(new_batch_ids)} files ({acc_size / 1024**3:.2f} GB)")
                
                disk_percent = (pending_disk_size / MAX_DISK_QUEUE_BYTES * 100)
                self.update_disk(disk_percent, f"Disk Queue: {pending_disk_size / 1024**3:.2f} / {MAX_DISK_QUEUE_GB} GB")

            except Exception as e:
                self.log_right(f"Producer error: {e}")
            
            time.sleep(2)

    def consumer_loop(self):
        while self.is_running:
            if self.is_paused:
                time.sleep(1)
                continue

            try:
                target_file_info = None
                
                with self.lock:
                    files = self.qbt_client.torrents_files(self.current_torrent_hash)
                    
                    if not target_file_info:
                        for f in files:
                            if f['progress'] == 1 and f['index'] not in self.processed_files_indices and f['index'] in self.safe_file_indices:
                                local_path = os.path.join(self.save_path, f['name'])
                                if os.path.exists(local_path):
                                    target_file_info = (local_path, f['index'])
                                    self.processed_files_indices.add(f['index'])
                                    break 
                
                if not target_file_info:
                    time.sleep(2)
                    continue

                path, f_idx = target_file_info
                
                if not self.is_running: break
                
                thread_name = threading.current_thread().name

                if path.endswith(".zst"):
                    cleaned = self.process_clean_zst(path)
                    
                    if cleaned:
                        if os.path.exists(cleaned):
                            self.log_left(f"[{thread_name}] Cleaned: {os.path.basename(cleaned)}")
                            fsize = os.path.getsize(cleaned)
                            
                            trigger_upload = False
                            with self.lock:
                                self.current_pending_size += fsize
                                self.pending_uploads.append((cleaned, path, f_idx))
                                if len(self.pending_uploads) >= UPLOAD_BATCH_SIZE or self.current_pending_size >= (MAX_PENDING_SIZE_GB * 1024**3):
                                    trigger_upload = True
                            
                            try: self.qbt_client.torrents_file_priority(self.current_torrent_hash, file_ids=f_idx, priority=0)
                            except: pass
                            
                            if trigger_upload:
                                self.flush_pending_uploads()
                        else:
                            self.log_left(f"[{thread_name}] Error: Output file missing: {cleaned}")
                    else:
                        try:
                            with open(TRASH_LIST_PATH, "a", encoding="utf-8") as f:
                                f.write(os.path.basename(path) + "\n")
                            self.log_left(f"[{thread_name}] Added to Trash List: {os.path.basename(path)}")
                        except Exception as e:
                            print(f"Error writing to trash list: {e}")

                        try:
                            self.qbt_client.torrents_file_priority(self.current_torrent_hash, file_ids=f_idx, priority=0)
                        except: pass
                        
                        with self.lock:
                            self.deletion_queue.add(path)

                else:
                    try: 
                        self.qbt_client.torrents_file_priority(self.current_torrent_hash, file_ids=f_idx, priority=0)
                        with self.lock:
                            self.deletion_queue.add(path)
                    except: pass
                
                trigger_upload = False
                with self.lock:
                    if len(self.pending_uploads) >= UPLOAD_BATCH_SIZE or self.current_pending_size >= (MAX_PENDING_SIZE_GB * 1024**3):
                        trigger_upload = True
                
                if trigger_upload:
                    self.flush_pending_uploads()

            except Exception as e:
                self.log_left(f"Consumer error: {e}")
                time.sleep(2)

    def start_pipeline(self):
        if not self.connect(): return
        torrents = self.qbt_client.torrents_info()
        if not torrents:
            self.log_right("No torrent found!")
            print("No torrent found!")
            return

        torrent = torrents[0]
        self.current_torrent_hash = torrent.hash
        self.save_path = torrent.save_path
        self.log_right(f"Managing Torrent: {torrent.name}")
        print(f"Managing Torrent: {torrent.name}")

        self.initialize_queue(self.current_torrent_hash)
        
        self.qbt_client.torrents_resume(torrent_hashes=self.current_torrent_hash)

        producer_thread = threading.Thread(target=self.producer_loop, daemon=True)
        producer_thread.start()
        self.threads = [producer_thread]

        janitor_thread = threading.Thread(target=self.janitor_loop, daemon=True)
        janitor_thread.start()
        self.threads.append(janitor_thread)

        for i in range(NUM_THREADS):
            t = threading.Thread(target=self.consumer_loop, daemon=True, name=f"Consumer-{i+1}")
            t.start()
            self.threads.append(t)
            self.log_right(f"Started Consumer Thread {i+1}")

    def safe_stop(self):
        self.is_running = False
        if self.pending_uploads:
            self.log_left("Stopping... Flushing pending uploads...")
            print("Stopping... Flushing pending uploads...")
            self.flush_pending_uploads()
        
        try:
            self.qbt_client.torrents_pause(torrent_hashes=self.current_torrent_hash)
        except: pass

        with self.lock:
            pending_delete = list(self.deletion_queue)
            self.deletion_queue.clear()
            
        if pending_delete:
            self.log_left("Performing final cleanup of source files...")
            print("Performing final cleanup of source files...")
            for file_path in pending_delete:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        self.log_left(f"Final Clean Deleted: {os.path.basename(file_path)}")
                        print(f"Final Clean Deleted: {os.path.basename(file_path)}")
                except Exception as e:
                    self.log_left(f"Could not delete {os.path.basename(file_path)} on exit.")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reddit Data Pipeline Processor")
        self.geometry("950x650")
        
        self.processor = DataProcessor(self, self.log_left_msg, self.log_right_msg, self.update_progress, self.update_disk, self.update_hf_progress)
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        control_frame = ttk.LabelFrame(self, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        self.btn_start = ttk.Button(control_frame, text="START PIPELINE", command=self.start_process)
        self.btn_start.pack(side="left", padx=5)
        self.btn_pause = ttk.Button(control_frame, text="PAUSE", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", padx=5)
        self.btn_stop = ttk.Button(control_frame, text="STOP", command=self.stop_process, state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_label.pack(side="right", padx=10)

        progress_frame = ttk.LabelFrame(self, text="Status", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(progress_frame, text="Download Progress:").pack(anchor="w")
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=2)
        self.progress_text = ttk.Label(progress_frame, text="0%")
        self.progress_text.pack(anchor="w")

        ttk.Label(progress_frame, text="Disk Queue (Processed Buffer):").pack(anchor="w", pady=(5,0))
        self.disk_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.disk_bar.pack(fill="x", pady=2)
        self.disk_text = ttk.Label(progress_frame, text="0 GB")
        self.disk_text.pack(anchor="w")

        hf_frame = ttk.LabelFrame(self, text="HF Upload Status", padding=10)
        hf_frame.pack(fill="x", padx=10, pady=5)
        self.hf_status_text = ttk.Label(hf_frame, text="Not uploading")
        self.hf_status_text.pack()

        logs_container = ttk.Frame(self)
        logs_container.pack(fill="both", expand=True, padx=10, pady=5)

        left_frame = ttk.LabelFrame(logs_container, text="Processing Logs", padding=5)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0,5))
        self.log_left_area = scrolledtext.ScrolledText(left_frame, height=10, state='disabled')
        self.log_left_area.pack(fill="both", expand=True)

        right_frame = ttk.LabelFrame(logs_container, text="Download/System Logs", padding=5)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5,0))
        self.log_right_area = scrolledtext.ScrolledText(right_frame, height=10, state='disabled')
        self.log_right_area.pack(fill="both", expand=True)

    def log_left_msg(self, message):
        self.after(0, self._append_log_left, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    def _append_log_left(self, msg):
        self.log_left_area.config(state='normal')
        self.log_left_area.insert(tk.END, msg)
        self.log_left_area.see(tk.END)
        self.log_left_area.config(state='disabled')

    def log_right_msg(self, message):
        self.after(0, self._append_log_right, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    def _append_log_right(self, msg):
        self.log_right_area.config(state='normal')
        self.log_right_area.insert(tk.END, msg)
        self.log_right_area.see(tk.END)
        self.log_right_area.config(state='disabled')

    def update_progress(self, percent, text):
        self.after(0, self._update_pb, percent, text)
    def _update_pb(self, percent, text):
        self.progress_bar['value'] = percent
        self.progress_text.config(text=text)

    def update_disk(self, percent, text):
        self.after(0, self._update_disk, percent, text)
    def _update_disk(self, percent, text):
        self.disk_bar['value'] = percent
        self.disk_text.config(text=text)

    def update_hf_progress(self, text):
        self.after(0, self._update_hf, text)
    def _update_hf(self, text):
        self.hf_status_text.config(text=text)

    def start_process(self):
        if self.processor.is_running: return
        self.processor.is_running = True
        self.processor.is_paused = False
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal", text="PAUSE")
        self.btn_stop.config(state="normal")
        self.status_label.config(text="RUNNING PIPELINE", foreground="green")
        self.thread = threading.Thread(target=self.processor.start_pipeline, daemon=True)
        self.thread.start()

    def toggle_pause(self):
        if self.processor.is_paused:
            self.processor.is_paused = False
            self.btn_pause.config(text="PAUSE")
            self.status_label.config(text="RUNNING", foreground="green")
        else:
            self.processor.is_paused = True
            self.btn_pause.config(text="RESUME")
            self.status_label.config(text="PAUSED", foreground="orange")

    def stop_process(self):
        self.status_label.config(text="STOPPING...", foreground="red")
        self.processor.safe_stop()
        self.on_stopped_by_user()

    def on_stopped_by_user(self):
        self.processor.is_paused = False
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.status_label.config(text="STOPPED", foreground="red")

    def on_closing(self):
        if self.processor.is_running:
            if messagebox.askokcancel("Exit", "Program is running. Are you sure you want to stop and exit?"):
                self.processor.safe_stop()
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()