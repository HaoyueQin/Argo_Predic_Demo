#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseTNT training dashboard — WSL-native, zero blocking.

Monitors a running `train_v4.py` training session and serves a web page
(default http://localhost:8080) with GPU/CPU/RAM stats, training progress,
loss curve, validation history and watchdog restart count.

Usage:
    python dashboard.py [--log <train log file>] [--history <history json>]
                        [--watchdog-log <watchdog log file>] [--port 8080]
                        [--patience 5]

Notes:
    - Linux/WSL only (reads /proc, nvidia-smi, pgrep).
    - The training log file is produced by redirecting train_v4.py output,
      e.g.: python src/train_v4.py ... 2>&1 | tee model_save_full_chunked/training.log
"""

import argparse
import http.server
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent

lock = threading.Lock()
_cpu_prev = {"total": 0, "idle": 0, "ts": 0}


def build_state(patience):
    return {
        "status": "idle", "all_done": False, "epoch": 0, "max_epoch": 0,
        "step": 0, "total_steps": 0, "train_pid": 0, "watchdog_pid": 0,
        "restart_count": 0, "max_restarts": 10,
        "epoch_remain_min": 0, "iter_speed": 0.0, "total_remain_min": 0,
        "gpu_name": "", "gpu_util": 0, "gpu_temp": 0,
        "gpu_vram_used": 0, "gpu_vram_total": 0, "gpu_power": 0.0,
        "cpu_name": "", "cpu_cores": 0, "cpu_util": 0.0,
        "ram_used_mb": 0, "ram_total_mb": 0,
        "train_loss": 0.0, "train_fde": 0.0, "val_fde": 0.0,
        "best_fde": 999.0, "best_epoch": 0, "patience_count": 0,
        "max_patience": patience, "completed_epochs": 0,
        "history_epochs": [], "history_losses": [],
        "live_steps": [], "live_losses": [], "log_tail": [], "last_update": "",
    }


def poll_all(state, log_path, hist_path, wd_path):
    while True:
        try:
            gpu = None
            cpu_name = ""
            cpu_u = 0.0
            ram_u = ram_t = 0
            log = {}
            hist = {}
            wd = {}
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,"
                     "memory.used,memory.total,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4)
                parts = [x.strip() for x in r.stdout.strip().split(",")]
                if len(parts) >= 6:
                    gpu = {"n": parts[0], "u": int(parts[1]), "t": int(parts[2]),
                           "vu": int(parts[3]), "vt": int(parts[4]),
                           "p": float(parts[5]) if parts[5].replace('.', '').replace('-', '').isdigit() else 0.0}
            except Exception:
                pass
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
            except Exception:
                pass
            try:
                with open("/proc/stat") as f:
                    parts = f.readline().split()
                vals = [int(x) for x in parts[1:]]
                total = sum(vals)
                idle = vals[3]
                now = time.time()
                if _cpu_prev["ts"] > 0:
                    dt = total - _cpu_prev["total"]
                    di = idle - _cpu_prev["idle"]
                    if dt > 0:
                        cpu_u = round(100 * (1 - di / dt), 1)
                _cpu_prev["total"] = total
                _cpu_prev["idle"] = idle
                _cpu_prev["ts"] = now
            except Exception:
                pass
            try:
                d = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        p = line.split(":")
                        if len(p) == 2:
                            d[p[0].strip()] = int(p[1].strip().split()[0])
                ram_t = d.get("MemTotal", 0) // 1024
                avail = d.get("MemAvailable", d.get("MemFree", 0)) // 1024
                ram_u = ram_t - avail
            except Exception:
                pass
            if os.path.exists(log_path):
                try:
                    txt = Path(log_path).read_text(errors="replace")
                    # train_v4 prints: "Epoch: 3/16  LR = ..."
                    em = re.findall(r"[Ee]poch[: ]+(\d+)/(\d+)", txt)
                    if em:
                        log["ep"], log["em"] = int(em[-1][0]), int(em[-1][1])
                    # tqdm progress bars: "3218/3218 [00:12<00:00, 25.6it/s]"
                    sm = re.findall(r"(\d+)/(\d+) \[", txt)
                    if sm:
                        log["st"], log["tt"] = int(sm[-1][0]), int(sm[-1][1])
                    lm = re.findall(r"loss=(\d+\.\d+)", txt)
                    if lm:
                        log["ls"] = float(lm[-1])
                    # train_v4 logs "Train FDE: x.xxx" and "minFDE=x.xxx"
                    fm = re.findall(r"Train FDE:?\s+(\d+\.\d+)|minFDE=(\d+\.\d+)", txt)
                    flat = [m[0] or m[1] for m in fm if any(m)]
                    if flat:
                        log["fd"] = float(flat[-1])
                    vm = re.findall(r"minFDE=(\d+\.\d+)", txt)
                    if vm:
                        log["vf"] = float(vm[-1])
                    sm = re.findall(r"(\d+\.\d+)s/it", txt)
                    if sm and float(sm[-1]) > 0:
                        log["sp"] = round(1.0 / float(sm[-1]), 2)
                    else:
                        im = re.findall(r"(\d+\.\d+)it/s", txt)
                        if im:
                            log["sp"] = float(im[-1])
                    # completed epochs from validation lines
                    log["cp"] = len(re.findall(r"\[Val\] Epoch \d+:", txt))
                    # "Finish." only counts if the log was written recently — a
                    # stale log from an earlier run must not show as "done"
                    fresh = (time.time() - os.path.getmtime(log_path)) < 600
                    log["dn"] = "Finish." in txt and "TRAINING_ALL_DONE" not in txt and fresh
                    lines = txt.strip().split("\n")
                    log["ac"] = bool(re.findall(r"\d+%\|", lines[-1])) if lines else False
                    log["tl"] = [l.strip()[:130] for l in lines[-12:] if l.strip()]
                    # live loss curve since the last epoch header
                    epoch_matches = list(re.finditer(r"[Ee]poch[: ]+\d+/\d+", txt))
                    live_txt = txt[epoch_matches[-1].start():] if epoch_matches else txt
                    all_losses = [float(x) for x in re.findall(r"loss=(\d+\.\d+)", live_txt)]
                    if len(all_losses) > 10:
                        step = max(1, len(all_losses) // 200)
                        log["ll"] = all_losses[::step]
                        if step > 1:  # avoid duplicating the last point
                            log["ll"].append(all_losses[-1])
                        log["ls_arr"] = [i * step for i in range(len(log["ll"]) - 1)] + [len(all_losses) - 1]
                except Exception:
                    pass
            if os.path.exists(hist_path):
                try:
                    h = json.loads(Path(hist_path).read_text())
                    hist = {"bf": h.get("best_loss", 999), "be": h.get("best_epoch", 0),
                            "pc": h.get("no_improve_count", 0),
                            "es": h.get("epochs", []), "ls": h.get("losses", [])}
                except Exception:
                    pass
            if os.path.exists(wd_path):
                try:
                    txt = Path(wd_path).read_text(errors="replace")
                    rm = re.findall(r"Retry (\d+)/(\d+)", txt)
                    if rm:
                        wd = {"r": int(rm[-1][0]), "m": int(rm[-1][1])}
                except Exception:
                    pass
            tp = wp = 0
            try:
                r = subprocess.run(["pgrep", "-f", "train_v4"], capture_output=True,
                                   text=True, timeout=2)
                pids = [int(x) for x in r.stdout.strip().split("\n") if x]
                # exclude this dashboard process (its own cmdline may contain the
                # keyword via --log/--watchdog-log arguments)
                pids = [p for p in pids if p != os.getpid()]
                if pids:
                    tp = pids[0]
            except Exception:
                pass
            try:
                r = subprocess.run(["pgrep", "-f", "watchdog"], capture_output=True,
                                   text=True, timeout=2)
                pids = [int(x) for x in r.stdout.strip().split("\n") if x]
                pids = [p for p in pids if p != os.getpid()]
                if pids:
                    wp = pids[0]
            except Exception:
                pass
            with lock:
                if gpu:
                    state.update(gpu_name=gpu["n"], gpu_util=gpu["u"], gpu_temp=gpu["t"],
                                 gpu_vram_used=gpu["vu"], gpu_vram_total=gpu["vt"],
                                 gpu_power=gpu["p"])
                if cpu_name:
                    state.update(cpu_name=cpu_name, cpu_cores=os.cpu_count() or 0)
                state.update(cpu_util=cpu_u, ram_used_mb=ram_u, ram_total_mb=ram_t,
                             train_pid=tp, watchdog_pid=wp)
                if log:
                    if log.get("ep") is not None:
                        state["epoch"] = log["ep"]
                    if log.get("em") is not None:
                        state["max_epoch"] = log["em"]
                    st = log.get("st", 0)
                    if st:
                        state["step"] = st
                    tt = log.get("tt", 0)
                    if tt:
                        state["total_steps"] = tt
                    if log.get("ls"):
                        state["train_loss"] = log["ls"]
                    if log.get("fd"):
                        state["train_fde"] = log["fd"]
                    if log.get("vf"):
                        state["val_fde"] = log["vf"]
                    if log.get("sp"):
                        state["iter_speed"] = log["sp"]
                    state["completed_epochs"] = log.get("cp") or 0
                    state["all_done"] = log.get("dn") or False
                    if log.get("tl"):
                        state["log_tail"] = log["tl"]
                    if state["all_done"]:
                        state["status"] = "done"
                    elif log.get("ac"):
                        state["status"] = "running"
                    elif state["completed_epochs"] > 0 and tp == 0:
                        state["status"] = "crashed"
                    elif state["completed_epochs"] > 0:
                        state["status"] = "stopped"
                    else:
                        state["status"] = "idle"
                    s = state["iter_speed"]
                    ts = state["total_steps"]
                    st = state["step"]
                    cur = state["epoch"]
                    mx = state["max_epoch"]
                    if s > 0 and ts > 0 and st > 0:
                        rem = ts - st
                        state["epoch_remain_min"] = round(rem / s / 60, 1)
                        pep = ts / s
                        tr = rem / s + (mx - cur) * pep
                        state["total_remain_min"] = round(tr / 60, 1)
                if hist:
                    state.update(best_fde=hist.get("bf", 999), best_epoch=hist.get("be", 0),
                                 patience_count=hist.get("pc", 0),
                                 history_epochs=hist.get("es", []),
                                 history_losses=hist.get("ls", []))
                if log.get("ll"):
                    state.update(live_steps=log["ls_arr"], live_losses=log["ll"])
                if wd:
                    state.update(restart_count=wd.get("r", 0), max_restarts=wd.get("m", 10))
                state["last_update"] = time.strftime("%H:%M:%S")
        except Exception:
            pass
        time.sleep(1.0)


def make_handler(state):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/state" or self.path.startswith("/api/state?"):
                with lock:
                    payload = json.dumps(state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                html = PAGE.encode()
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

        def log_message(self, *args):
            pass

    return Handler


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>DenseTNT Dashboard</title>
<style>
body{font-family:Consolas,monospace;background:#1e1e2e;color:#cdd6f4;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.card{background:#313244;border-radius:8px;padding:12px}
.card h3{margin:0 0 6px 0;font-size:13px;color:#a6adc8}
.val{font-size:22px;font-weight:bold}
.bar{height:8px;background:#45475a;border-radius:4px;overflow:hidden;margin-top:6px}
.bar>div{height:100%;background:#89b4fa}
#log{background:#11111b;border-radius:8px;padding:10px;margin-top:10px;font-size:12px;white-space:pre-wrap;max-height:220px;overflow-y:auto}
canvas{width:100%;height:220px;background:#11111b;border-radius:8px;margin-top:10px}
.row{display:flex;gap:24px;align-items:baseline;flex-wrap:wrap}
</style>
</head>
<body>
<h1>DenseTNT Training Dashboard</h1>
<div class="row">
<span>Status: <b id="st">-</b></span>
<span>Epoch: <b id="ep">-</b>/<b id="mx">-</b></span>
<span>Step: <b id="sp">-</b>/<b id="tt">-</b></span>
<span>Speed: <b id="spd">-</b></span>
<span>Restarts: <b id="rc">-</b>/<b id="mr">-</b></span>
<span>ETA: <b id="eta">-</b></span>
</div>
<div class="grid">
<div class="card"><h3>GPU</h3><div class="val" id="gn">-</div><div id="gu">util -</div><div id="gt">temp -</div><div id="gv">vram -</div><div id="gp">power -</div><div class="bar"><div id="gbar" style="width:0%"></div></div></div>
<div class="card"><h3>CPU</h3><div class="val" id="cn">-</div><div id="cu">util -</div><div id="cm">mem -</div><div class="bar"><div id="cbar" style="width:0%"></div></div></div>
<div class="card"><h3>Train Loss</h3><div class="val" id="tl">-</div><div id="tf">train FDE -</div><div id="vf">val FDE -</div></div>
<div class="card"><h3>Best (val FDE)</h3><div class="val" id="bf">-</div><div id="be">best epoch -</div><div id="pc">patience -</div></div>
</div>
<canvas id="cv"></canvas>
<div id="log"></div>
<script>
const s=document.getElementById('st'),ep=document.getElementById('ep'),mx=document.getElementById('mx'),
sp=document.getElementById('sp'),tt=document.getElementById('tt'),spd=document.getElementById('spd'),
rc=document.getElementById('rc'),mr=document.getElementById('mr'),eta=document.getElementById('eta'),
gn=document.getElementById('gn'),gu=document.getElementById('gu'),gt=document.getElementById('gt'),
gv=document.getElementById('gv'),gp=document.getElementById('gp'),gbar=document.getElementById('gbar'),
cn=document.getElementById('cn'),cu=document.getElementById('cu'),cm=document.getElementById('cm'),
cbar=document.getElementById('cbar'),tl=document.getElementById('tl'),tf=document.getElementById('tf'),
vf=document.getElementById('vf'),bf=document.getElementById('bf'),be=document.getElementById('be'),
pc=document.getElementById('pc'),log=document.getElementById('log'),cv=document.getElementById('cv');
async function poll(){try{
const r=await fetch('/api/state');const d=await r.json();
s.textContent=d.status;ep.textContent=d.epoch;mx.textContent=d.max_epoch;
sp.textContent=d.step;tt.textContent=d.total_steps;spd.textContent=d.iter_speed?d.iter_speed.toFixed(2)+' it/s':'-';
rc.textContent=d.restart_count;mr.textContent=d.max_restarts;
eta.textContent=d.total_remain_min?d.total_remain_min.toFixed(1)+' min left':'-';
gn.textContent=d.gpu_name||'-';gu.textContent='util '+d.gpu_util+'%';gt.textContent='temp '+d.gpu_temp+'C';
gv.textContent=d.gpu_vram_used+'/'+d.gpu_vram_total+' MB';gp.textContent=d.gpu_power?d.gpu_power.toFixed(0)+' W':'-';
gbar.style.width=Math.min(100,d.gpu_util)+'%';
cn.textContent=d.cpu_name||'-';cu.textContent='util '+d.cpu_util+'%';cm.textContent=d.ram_used_mb+'/'+d.ram_total_mb+' MB';
cbar.style.width=Math.min(100,d.cpu_util)+'%';
tl.textContent=d.train_loss?d.train_loss.toFixed(3):'-';tf.textContent='train FDE '+(d.train_fde?d.train_fde.toFixed(2):'-');
vf.textContent='val FDE '+(d.val_fde?d.val_fde.toFixed(2):'-');
bf.textContent=d.best_fde<900?d.best_fde.toFixed(3):'-';be.textContent='best epoch '+d.best_epoch;
pc.textContent='patience '+d.patience_count+'/'+d.max_patience;
log.textContent=(d.log_tail||[]).join('\\n');
draw(d);
}catch(e){}}
function draw(d){
const c=cv.getContext('2d');const W=cv.width=800,H=cv.height=220;c.clearRect(0,0,W,H);
const ls=d.live_losses||[],xs=d.live_steps||[];
if(ls.length<2)return;
const mxv=Math.max(...ls),mnv=Math.min(...ls),rg=(mxv-mnv)||1;
c.strokeStyle='#89b4fa';c.lineWidth=1.5;c.beginPath();
ls.forEach((v,i)=>{const x=xs[i]/Math.max(...xs,1)*(W-20)+10,y=H-15-(v-mnv)/rg*(H-40);i?c.lineTo(x,y):c.moveTo(x,y)});
c.stroke();
}
setInterval(poll,1000);poll();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='DenseTNT training dashboard')
    parser.add_argument('--log', default=None, help='training log file')
    parser.add_argument('--history', default=None, help='training_history.json path')
    parser.add_argument('--watchdog-log', default=None, help='watchdog log file')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    args = parser.parse_args()

    out_dir = PROJ / 'model_save_full_chunked'
    log_path = str(args.log or out_dir / 'training.log')
    hist_path = str(args.history or out_dir / 'training_history.json')
    wd_path = str(args.watchdog_log or out_dir / 'watchdog.log')

    state = build_state(args.patience)
    threading.Thread(target=poll_all, args=(state, log_path, hist_path, wd_path),
                     daemon=True).start()

    # Bind to loopback only: the page shows live training state (log tail,
    # PIDs) with no authentication — exposing it on 0.0.0.0 would leak to the LAN.
    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(state))
    print(f'[Dashboard] http://localhost:{args.port}  (log: {log_path})')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
