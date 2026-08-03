#!/usr/bin/env python3
"""多线程分块下载 HF 模型文件(每块带断点续传 + 无限重试),用法:
  python dl_parallel.py <url> <out_path> [threads]
"""
import os
import sys
import time
import threading
import urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAX_RETRIES = 100000

proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")

def opener():
    if proxy:
        ph = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(ph)
    return urllib.request.build_opener()

def get_size():
    req = urllib.request.Request(URL, method="HEAD")
    with opener().open(req, timeout=30) as r:
        return int(r.headers["Content-Length"])

def download_range(start, end, idx, progress, lock, stop):
    """下载 [start, end],失败/中断后从已写入位置继续,直到完成或 stop 置位。"""
    path = OUT + f".part{idx}"
    while not stop.is_set():
        try:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if start + have > end:
                break  # 本块已完成
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start+have}-{end}"})
            with opener().open(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    with lock:
                        progress[idx] += len(chunk)
            if os.path.getsize(path) >= end - start + 1:
                break
            print(f"  [t{idx}] range done but short, retry", flush=True)
        except Exception as e:
            with lock:
                if not getattr(e, "retried", False):
                    print(f"  [t{idx}] {type(e).__name__}: retrying...", flush=True)
                    e.retried = True
            time.sleep(1)

def main():
    total = get_size()
    print(f"total: {total/1e9:.2f} GB, threads: {THREADS}", flush=True)
    chunk = total // THREADS
    ranges = [(i * chunk, (i + 1) * chunk - 1 if i < THREADS - 1 else total - 1) for i in range(THREADS)]
    progress = [0] * THREADS
    lock = threading.Lock()
    stop = threading.Event()
    threads = [threading.Thread(target=download_range, args=(s, e, i, progress, lock, stop), daemon=True)
               for i, (s, e) in enumerate(ranges)]
    t0 = time.time()
    for t in threads:
        t.start()
    last_report = 0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        done = sum(progress)
        dt = time.time() - t0
        if done - last_report > 20_000_000 or dt > 60:
            print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB  {done/dt/1e6:.2f} MB/s  {dt:.0f}s", flush=True)
            last_report = done
    for t in threads:
        t.join()
    # 校验各块大小
    ok = all(os.path.getsize(OUT + f".part{i}") == e - s + 1 for i, (s, e) in enumerate(ranges))
    if not ok:
        print("ERROR: some parts incomplete", flush=True)
        sys.exit(1)
    with open(OUT, "wb") as out:
        for i in range(THREADS):
            with open(OUT + f".part{i}", "rb") as p:
                out.write(p.read())
            os.remove(OUT + f".part{i}")
    print(f"done -> {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)", flush=True)

if __name__ == "__main__":
    main()
