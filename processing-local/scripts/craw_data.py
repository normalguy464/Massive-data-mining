"""
Mục tiêu:
- Chỉ lấy file submission trong 1 folder cụ thể, ví dụ pushshift_1
- Giới hạn tổng dung lượng khoảng 10-15 GB
- Tránh quét toàn bộ repo vì rất chậm với dataset lớn

Lý do dùng HfFileSystem.ls thay vì list_repo_tree toàn repo:
- Chỉ nhìn vào đúng folder cần dùng
- Ít metadata hơn
- Nhanh hơn đáng kể trong trường hợp repo rất lớn
"""

import time
from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download

# =========================
# 1) CẤU HÌNH
# =========================

REPO_ID = "anhchanghoangsg/reddit_pushshift_dataset_cleaned"
REPO_TYPE = "dataset"

# Chỉ quét đúng folder này
TARGET_FOLDER = "pushshift_1"

# Thư mục local để lưu dữ liệu
LOCAL_DIR = "./data/raw"

# Tổng dung lượng muốn tải
TARGET_SIZE_GB = 8
TARGET_BYTES = TARGET_SIZE_GB * 1024**3

# Chỉ lấy submission
SUBMISSION_KEYWORDS = ["submission", "submissions"]

# small_first:
# - lấy file nhỏ trước để dễ gom nhiều file trong quota 10-15GB
PICK_STRATEGY = "small_first"

# True thì tải luôn, False thì chỉ in danh sách file dự kiến
SHOULD_DOWNLOAD = True


# =========================
# 2) HÀM HỖ TRỢ
# =========================

def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def is_submission_file(path: str) -> bool:
    """
    Chỉ nhận:
    - file trong đúng folder TARGET_FOLDER
    - file parquet
    - tên có chứa submission/submissions

    Lý do:
    - đảm bảo không kéo comments
    - tránh file ngoài folder mục tiêu
    """
    p = path.lower()
    return (
        p.startswith(f"datasets/{REPO_ID.lower()}/{TARGET_FOLDER.lower()}/")
        and p.endswith(".parquet")
        and any(k in p for k in SUBMISSION_KEYWORDS)
    )


def list_files_in_one_folder(repo_id: str, target_folder: str):
    """
    Chỉ liệt kê file trong một folder từ xa.

    Lý do:
    - nhanh hơn rất nhiều so với quét cả repo
    - đúng với nhu cầu của bạn: chỉ cần pushshift_1
    """
    fs = HfFileSystem()

    # detail=True để lấy metadata như size
    # đường dẫn dataset trên HfFileSystem có dạng:
    # datasets/<repo_id>/<folder>
    folder_path = f"datasets/{repo_id}/{target_folder}"
    entries = fs.ls(folder_path, detail=True)

    files = []
    for entry in entries:
        # entry thường là dict với các field như name, size, type
        path = entry.get("name")
        size = entry.get("size")
        entry_type = entry.get("type")

        # Chỉ lấy file thực sự
        if entry_type != "file":
            continue

        # Bỏ qua nếu không có size
        if size is None:
            continue

        files.append({
            "path": path,
            "size": size,
        })

    return files


def choose_files(candidate_files, target_bytes, strategy="small_first"):
    """
    Chọn file sao cho tổng dung lượng không vượt quá quota.

    Lý do:
    - bạn muốn khoảng 10-15GB chứ không tải toàn bộ submission
    """
    if strategy == "small_first":
        sorted_files = sorted(candidate_files, key=lambda x: x["size"])
    elif strategy == "large_first":
        sorted_files = sorted(candidate_files, key=lambda x: x["size"], reverse=True)
    else:
        raise ValueError("strategy phải là 'small_first' hoặc 'large_first'")

    picked = []
    total = 0

    for f in sorted_files:
        if total + f["size"] <= target_bytes:
            picked.append(f)
            total += f["size"]

    return picked, total


def convert_fs_path_to_repo_filename(fs_path: str, repo_id: str) -> str:
    """
    Chuyển path kiểu:
      datasets/<repo_id>/pushshift_1/abc.parquet
    thành:
      pushshift_1/abc.parquet

    Lý do:
    - hf_hub_download cần filename là đường dẫn bên trong repo
    """
    prefix = f"datasets/{repo_id}/"
    if not fs_path.startswith(prefix):
        raise ValueError(f"Path không hợp lệ: {fs_path}")
    return fs_path[len(prefix):]


def print_plan(files, total_bytes):
    print("=" * 80)
    print("DANH SÁCH FILE DỰ KIẾN TẢI")
    print("=" * 80)

    for i, f in enumerate(files, 1):
        print(f"{i:>3}. {f['path']}  |  {bytes_to_gb(f['size']):.2f} GB")

    print("-" * 80)
    print(f"Tổng số file: {len(files)}")
    print(f"Tổng dung lượng: {bytes_to_gb(total_bytes):.2f} GB")
    print("=" * 80)


def download_files(repo_id: str, repo_type: str, files, local_dir: str):
    """
    Tải từng file một.

    Lý do:
    - dễ resume
    - dễ biết file nào lỗi
    - không phải mở snapshot lớn
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, 1):
        filename_in_repo = convert_fs_path_to_repo_filename(f["path"], repo_id)
        print(f"[{i}/{len(files)}] Đang tải: {filename_in_repo} ({bytes_to_gb(f['size']):.2f} GB)")

        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename_in_repo,
            local_dir=local_dir,
        )

    print("Tải xong.")


# =========================
# 3) CHẠY CHÍNH
# =========================

def main():
    print(f"Đang liệt kê file trong folder: {TARGET_FOLDER}")

    all_files_in_folder = list_files_in_one_folder(REPO_ID, TARGET_FOLDER)
    print(f"Số file tìm thấy trong folder: {len(all_files_in_folder)}")

    submission_files = [f for f in all_files_in_folder if is_submission_file(f["path"])]
    print(f"Số file submission: {len(submission_files)}")

    if not submission_files:
        print("Không thấy file submission nào. Có thể naming của repo khác dự kiến.")
        return

    total_submission_bytes = sum(f["size"] for f in submission_files)
    print(f"Tổng dung lượng submission trong folder này: {bytes_to_gb(total_submission_bytes):.2f} GB")

    picked_files, picked_total = choose_files(
        submission_files,
        TARGET_BYTES,
        strategy=PICK_STRATEGY,
    )

    if not picked_files:
        print("Không chọn được file nào trong quota hiện tại.")
        return

    print_plan(picked_files, picked_total)

    if SHOULD_DOWNLOAD:
        download_files(REPO_ID, REPO_TYPE, picked_files, LOCAL_DIR)
    else:
        print("Chưa tải, mới chỉ lập kế hoạch.")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed_seconds = time.perf_counter() - start_time
    print(f"craw_data.py finished in {elapsed_seconds:.2f}s ({elapsed_seconds / 60.0:.2f}m).")
"""
Mục tiêu:
- Chỉ lấy file submission trong 1 folder cụ thể, ví dụ pushshift_1
- Giới hạn tổng dung lượng khoảng 10-15 GB
- Tránh quét toàn bộ repo vì rất chậm với dataset lớn

Lý do dùng HfFileSystem.ls thay vì list_repo_tree toàn repo:
- Chỉ nhìn vào đúng folder cần dùng
- Ít metadata hơn
- Nhanh hơn đáng kể trong trường hợp repo rất lớn
"""

from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download

# =========================
# 1) CẤU HÌNH
# =========================

REPO_ID = "anhchanghoangsg/reddit_pushshift_dataset_cleaned"
REPO_TYPE = "dataset"

# Chỉ quét đúng folder này
TARGET_FOLDER = "pushshift_1"

# Thư mục local để lưu dữ liệu
LOCAL_DIR = "./reddit_subset"

# Tổng dung lượng muốn tải
TARGET_SIZE_GB = 12
TARGET_BYTES = TARGET_SIZE_GB * 1024**3

# Chỉ lấy submission
SUBMISSION_KEYWORDS = ["submission", "submissions"]

# small_first:
# - lấy file nhỏ trước để dễ gom nhiều file trong quota 10-15GB
PICK_STRATEGY = "small_first"

# True thì tải luôn, False thì chỉ in danh sách file dự kiến
SHOULD_DOWNLOAD = True


# =========================
# 2) HÀM HỖ TRỢ
# =========================

def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def is_submission_file(path: str) -> bool:
    """
    Chỉ nhận:
    - file trong đúng folder TARGET_FOLDER
    - file parquet
    - tên có chứa submission/submissions

    Lý do:
    - đảm bảo không kéo comments
    - tránh file ngoài folder mục tiêu
    """
    p = path.lower()
    return (
        p.startswith(f"datasets/{REPO_ID.lower()}/{TARGET_FOLDER.lower()}/")
        and p.endswith(".parquet")
        and any(k in p for k in SUBMISSION_KEYWORDS)
    )


def list_files_in_one_folder(repo_id: str, target_folder: str):
    """
    Chỉ liệt kê file trong một folder từ xa.

    Lý do:
    - nhanh hơn rất nhiều so với quét cả repo
    - đúng với nhu cầu của bạn: chỉ cần pushshift_1
    """
    fs = HfFileSystem()

    # detail=True để lấy metadata như size
    # đường dẫn dataset trên HfFileSystem có dạng:
    # datasets/<repo_id>/<folder>
    folder_path = f"datasets/{repo_id}/{target_folder}"
    entries = fs.ls(folder_path, detail=True)

    files = []
    for entry in entries:
        # entry thường là dict với các field như name, size, type
        path = entry.get("name")
        size = entry.get("size")
        entry_type = entry.get("type")

        # Chỉ lấy file thực sự
        if entry_type != "file":
            continue

        # Bỏ qua nếu không có size
        if size is None:
            continue

        files.append({
            "path": path,
            "size": size,
        })

    return files


def choose_files(candidate_files, target_bytes, strategy="small_first"):
    """
    Chọn file sao cho tổng dung lượng không vượt quá quota.

    Lý do:
    - bạn muốn khoảng 10-15GB chứ không tải toàn bộ submission
    """
    if strategy == "small_first":
        sorted_files = sorted(candidate_files, key=lambda x: x["size"])
    elif strategy == "large_first":
        sorted_files = sorted(candidate_files, key=lambda x: x["size"], reverse=True)
    else:
        raise ValueError("strategy phải là 'small_first' hoặc 'large_first'")

    picked = []
    total = 0

    for f in sorted_files:
        if total + f["size"] <= target_bytes:
            picked.append(f)
            total += f["size"]

    return picked, total


def convert_fs_path_to_repo_filename(fs_path: str, repo_id: str) -> str:
    """
    Chuyển path kiểu:
      datasets/<repo_id>/pushshift_1/abc.parquet
    thành:
      pushshift_1/abc.parquet

    Lý do:
    - hf_hub_download cần filename là đường dẫn bên trong repo
    """
    prefix = f"datasets/{repo_id}/"
    if not fs_path.startswith(prefix):
        raise ValueError(f"Path không hợp lệ: {fs_path}")
    return fs_path[len(prefix):]


def print_plan(files, total_bytes):
    print("=" * 80)
    print("DANH SÁCH FILE DỰ KIẾN TẢI")
    print("=" * 80)

    for i, f in enumerate(files, 1):
        print(f"{i:>3}. {f['path']}  |  {bytes_to_gb(f['size']):.2f} GB")

    print("-" * 80)
    print(f"Tổng số file: {len(files)}")
    print(f"Tổng dung lượng: {bytes_to_gb(total_bytes):.2f} GB")
    print("=" * 80)


def download_files(repo_id: str, repo_type: str, files, local_dir: str):
    """
    Tải từng file một.

    Lý do:
    - dễ resume
    - dễ biết file nào lỗi
    - không phải mở snapshot lớn
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, 1):
        filename_in_repo = convert_fs_path_to_repo_filename(f["path"], repo_id)
        print(f"[{i}/{len(files)}] Đang tải: {filename_in_repo} ({bytes_to_gb(f['size']):.2f} GB)")

        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename_in_repo,
            local_dir=local_dir,
        )

    print("Tải xong.")


# =========================
# 3) CHẠY CHÍNH
# =========================

def main():
    print(f"Đang liệt kê file trong folder: {TARGET_FOLDER}")

    all_files_in_folder = list_files_in_one_folder(REPO_ID, TARGET_FOLDER)
    print(f"Số file tìm thấy trong folder: {len(all_files_in_folder)}")

    submission_files = [f for f in all_files_in_folder if is_submission_file(f["path"])]
    print(f"Số file submission: {len(submission_files)}")

    if not submission_files:
        print("Không thấy file submission nào. Có thể naming của repo khác dự kiến.")
        return

    total_submission_bytes = sum(f["size"] for f in submission_files)
    print(f"Tổng dung lượng submission trong folder này: {bytes_to_gb(total_submission_bytes):.2f} GB")

    picked_files, picked_total = choose_files(
        submission_files,
        TARGET_BYTES,
        strategy=PICK_STRATEGY,
    )

    if not picked_files:
        print("Không chọn được file nào trong quota hiện tại.")
        return

    print_plan(picked_files, picked_total)

    if SHOULD_DOWNLOAD:
        download_files(REPO_ID, REPO_TYPE, picked_files, LOCAL_DIR)
    else:
        print("Chưa tải, mới chỉ lập kế hoạch.")


if __name__ == "__main__":
    main()
