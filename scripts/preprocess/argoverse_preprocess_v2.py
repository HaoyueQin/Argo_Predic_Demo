"""
===============================================================================
Argoverse v1.1 数据预处理与清洗脚本
===============================================================================
功能：
  1. 读取原始 Argoverse v1.1 CSV 数据
  2. 数据校验（AV/AGENT 存在性、轨迹完整性）
  3. 异常检测与修复（速度异常、帧间跳跃）
  4. 场景分类（直行/左转/右转/复杂）
  5. 坐标变换（方案B：以 Agent 为中心旋转）
  6. 输出两种格式：
     - data_cleaned/*.csv → DenseTNT 使用（原始世界坐标，清洗后）
     - data_processed/*.npz → LSTM/Kalman 使用（Agent-centered 旋转坐标）

用法：python argoverse_preprocess_v2.py [--data-dir DIR] [--out-dir DIR]

日期：2026-06-02
===============================================================================
"""

import os
import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────────────────────────────────────
# 环境配置：限制 OpenBLAS/MKL 线程数，防止多进程时内存爆炸
# ─────────────────────────────────────────────────────────────────────────────
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置（基于仓库根，可通过命令行覆盖）
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")          # 原始数据（Argoverse 官方布局: raw/{train,val}/data）
CLEANED_DIR = os.path.join(REPO_ROOT, "data", "cleaned")   # 清洗后（DenseTNT）
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")  # 处理后（LSTM/Kalman）
REPORT_DIR = os.path.join(REPO_ROOT, "outputs", "reports")    # 报告输出

# ─────────────────────────────────────────────────────────────────────────────
# 常量定义
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_FRAMES = 50        # 总帧数（5秒 × 10Hz）
HISTORY_FRAMES = 20      # 历史帧数（2秒）
MAX_SPEED_MS = 50.0      # 最大合理速度 (m/s)，约 180 km/h
MAX_JUMP_DIST = 5.0      # 最大合理帧间位移 (m)
MIN_TRAVEL_DIST = 0.5    # 最小移动距离 (m)，低于此视为静止
MIN_HISTORY_LEN = 20     # 最少历史帧数


def process_one(args):
    """
    处理单个场景的完整流程。

    输入: (csv_path, split_name, cleaned_dir, processed_dir)
    输出: 处理结果元组

    流程:
      1. 读取 CSV → 解析轨迹
      2. 校验 AV/AGENT 存在性
      3. 补齐到 50 帧
      4. 异常检测与修复（速度异常、帧间跳跃）
      5. 静止检测
      6. 场景分类（直行/左转/右转/复杂）
      7. 输出 DenseTNT CSV（原始世界坐标）
      8. 坐标变换 → 输出 LSTM/Kalman NPZ（旋转坐标）
    """
    csv_path, split_name, cleaned_dir, processed_dir = args
    scene_id = os.path.splitext(os.path.basename(csv_path))[0]

    try:
        # ── Step 1: 读取 CSV ──
        # Argoverse v1.1 CSV 格式: TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            rows = [r for r in reader if len(r) >= 5]

        # ── Step 2: 解析轨迹 ──
        # 按 TRACK_ID 分组，每条轨迹包含时间戳、坐标、对象类型
        tracks = defaultdict(list)
        city_name = "MIA"  # 默认城市
        for row in rows:
            try:
                ts = float(row[0])      # 时间戳（秒）
                tid = row[1]            # 轨迹 ID
                otype = row[2]          # 对象类型: AV/AGENT/OTHERS
                x, y = float(row[3]), float(row[4])  # 世界坐标（米）
                city = row[5] if len(row) > 5 else "MIA"  # 城市: MIA(迈阿密)/PIT(匹兹堡)
                tracks[tid].append({"ts": ts, "x": x, "y": y, "type": otype, "city": city})
                city_name = city
            except (ValueError, IndexError):
                continue

        # ── Step 3: 校验 AV 和 AGENT 存在性 ──
        # AV = ego vehicle（自动驾驶车），AGENT = 预测目标
        av_id = agent_id = None
        for tid, tlist in tracks.items():
            types = set(r["type"] for r in tlist)
            if "AV" in types:
                av_id = tid
            if "AGENT" in types:
                agent_id = tid

        if av_id is None:
            return ("skip_no_av", scene_id)
        if agent_id is None:
            return ("skip_no_agent", scene_id)

        # ── Step 4: 按时间排序轨迹 ──
        def get_track(tid):
            """获取排序后的轨迹：返回 (时间戳数组, 坐标数组)"""
            data = sorted(tracks[tid], key=lambda r: r["ts"])
            ts_arr = np.array([r["ts"] for r in data])
            xy_arr = np.array([[r["x"], r["y"]] for r in data])
            return ts_arr, xy_arr

        av_ts, av_xy = get_track(av_id)
        agent_ts, agent_xy = get_track(agent_id)

        # ── Step 4b: 时间戳单位归一化（仅用于物理量计算）──
        # Argoverse 1 官方 CSV 的 TIMESTAMP 为微秒（10Hz，间隔约 1e5）。
        # 异常检测按秒计算，这里把微秒统一换算为秒；输出 CSV 仍保留原始
        # 时间戳（DenseTNT 加载时做相对化处理，对单位不敏感）。
        av_ts_raw, agent_ts_raw = av_ts, agent_ts
        dts = np.diff(np.concatenate([av_ts, agent_ts]))
        median_dt = float(np.median(dts)) if len(dts) else 0.0
        if median_dt > 1.0:  # not seconds → microseconds
            av_ts = av_ts * 1e-6
            agent_ts = agent_ts * 1e-6

        # ── Step 5: 检查历史帧数 ──
        if len(agent_xy) < MIN_HISTORY_LEN:
            return ("skip_short", scene_id)

        # ── Step 6: 轨迹完整性 ──
        # AGENT 轨迹不足 50 帧（历史 ≥20 帧但未来被截断）时直接跳过：补齐会
        # 人为制造"静止未来"（短轨迹被当作停车），且 DenseTNT 训练端要求
        # AGENT 恰好 50 行（dataset 内 assert len(AGENT)==50），补齐的 CSV
        # 同样会被丢弃；旧实现以补齐后的长度遍历原始时间戳会越界崩溃。
        if len(agent_xy) < TOTAL_FRAMES:
            return ("skip_short", scene_id)

        # 切分历史帧和未来帧
        hist_xy = agent_xy[:HISTORY_FRAMES].copy()      # 帧 0-19
        gt_xy = agent_xy[HISTORY_FRAMES:TOTAL_FRAMES].copy()  # 帧 20-49
        hist_ts = agent_ts[:HISTORY_FRAMES]

        # ── Step 7: 异常检测与修复 ──
        # 检测连续帧之间的速度异常和位移跳跃
        anomalies = {"speed": 0, "jump": 0}
        for xy_set, ts_set in [(hist_xy, hist_ts), (gt_xy, agent_ts[HISTORY_FRAMES:TOTAL_FRAMES])]:
            if len(xy_set) < 2:
                continue
            # 计算帧间位移
            disp = np.sqrt(np.sum(np.diff(xy_set, axis=0) ** 2, axis=1))
            # 计算帧间时间差
            gaps = np.diff(ts_set)
            # 只对连续帧（时间间隔 0.05-0.2s）进行检测
            ok = (gaps > 0.05) & (gaps < 0.2)
            if ok.any():
                # 速度 = 位移 / 时间
                speeds = disp[ok] / gaps[ok]
                anomalies["speed"] += int(np.sum(speeds > MAX_SPEED_MS))
                # 检测帧间跳跃
                jump_mask = disp[ok] > MAX_JUMP_DIST
                anomalies["jump"] += int(jump_mask.sum())
                # 修复跳跃点：用前一帧坐标填充
                for idx in np.where(ok)[0][jump_mask]:
                    if idx + 1 < len(xy_set):
                        xy_set[idx + 1] = xy_set[idx]

        # ── Step 8: 静止检测 ──
        # 如果 AGENT 整体移动距离 < 0.5m，标记为静止
        total_disp = np.sqrt(np.sum((gt_xy[-1] - hist_xy[0]) ** 2))
        is_stationary = total_disp < MIN_TRAVEL_DIST

        # ── Step 9: 场景分类 ──
        # 基于 AGENT 轨迹的朝向变化进行分类
        all_pts = np.vstack([hist_xy, gt_xy])
        scene_type = "straight"
        if len(all_pts) >= 10:
            # 计算起始方向和结束方向
            sd = all_pts[10] - all_pts[0]     # 前 10 帧的方向
            ed = all_pts[-1] - all_pts[-10]   # 后 10 帧的方向
            n1, n2 = np.linalg.norm(sd), np.linalg.norm(ed)
            if n1 > 1e-6 and n2 > 1e-6:
                sd, ed = sd / n1, ed / n2  # 归一化
                # 计算朝向变化角度
                cross_val = sd[0] * ed[1] - sd[1] * ed[0]  # 叉积
                dot_val = np.dot(sd, ed)                    # 点积
                hc = np.arctan2(cross_val, dot_val)         # 朝向变化角（弧度）
                # 分类
                if abs(hc) >= np.radians(15):
                    scene_type = "left_turn" if hc > 0 else "right_turn"
                elif abs(hc) >= np.radians(5):
                    scene_type = "complex"

        # ── Step 10: 输出 DenseTNT CSV ──
        # 格式：无表头，每行 TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME
        # 坐标系：原始世界坐标（DenseTNT 模型内部有自己的坐标变换）
        cleaned_path = os.path.join(cleaned_dir, f"{scene_id}.csv")
        with open(cleaned_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # 写入 AV 轨迹（原始时间戳，见 Step 4b）
            for j in range(len(av_ts)):
                w.writerow([av_ts_raw[j], av_id, "AV", av_xy[j, 0], av_xy[j, 1], city_name])
            # 写入 AGENT 轨迹（清洗后，原始时间戳）
            full_agent = np.vstack([hist_xy, gt_xy])
            for j in range(min(len(agent_ts), TOTAL_FRAMES)):
                w.writerow([agent_ts_raw[j], agent_id, "AGENT", full_agent[j, 0], full_agent[j, 1], city_name])
            # 写入 OTHERS 轨迹
            for tid, tlist in tracks.items():
                if tid in (av_id, agent_id):
                    continue
                for r in tlist:
                    w.writerow([r["ts"], tid, r["type"], r["x"], r["y"], r["city"]])

        # ── Step 11: 坐标变换（方案B：以 Agent 为中心旋转）──
        # 原点：AGENT 在 frame 19（历史最后一帧）的位置
        # 旋转：AGENT 在 frame 18→19 的运动方向对齐 x 轴
        cent_x, cent_y = hist_xy[-1]  # 原点坐标
        dx = hist_xy[-1, 0] - hist_xy[-2, 0]
        dy = hist_xy[-1, 1] - hist_xy[-2, 1]
        angle = np.arctan2(dy, dx)    # 旋转角度
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)

        def transform(pts):
            """将世界坐标变换为 Agent-centered 旋转坐标"""
            s = pts - np.array([cent_x, cent_y])  # 平移
            return np.column_stack([
                s[:, 0] * cos_a - s[:, 1] * sin_a,  # 旋转
                s[:, 0] * sin_a + s[:, 1] * cos_a
            ])

        hist_local = transform(hist_xy)  # 历史轨迹（旋转坐标）
        gt_local = transform(gt_xy)      # 未来轨迹（旋转坐标）

        # ── Step 12: 输出 LSTM/Kalman NPZ ──
        pt_path = os.path.join(processed_dir, f"{scene_id}.npz")
        np.savez(pt_path,
            scene_id=int(scene_id),           # 场景 ID
            hist=hist_local.astype(np.float32),       # 历史轨迹 (20,2) 旋转坐标
            gt=gt_local.astype(np.float32),           # 未来轨迹 (30,2) 旋转坐标
            hist_global=hist_xy.astype(np.float32),   # 历史轨迹 (20,2) 世界坐标
            gt_global=gt_xy.astype(np.float32),       # 未来轨迹 (30,2) 世界坐标
            city=city_name,                           # 城市名
            cent_x=cent_x, cent_y=cent_y, angle=angle,  # 逆变换参数
            scene_type=scene_type,                    # 场景类型
            is_stationary=is_stationary,              # 是否静止
            speed_anomalies=anomalies["speed"],       # 速度异常点数
            jump_anomalies=anomalies["jump"],         # 跳跃异常点数
        )

        return ("ok", scene_id, scene_type, is_stationary, anomalies["speed"], anomalies["jump"])

    except Exception as e:
        return ("error", scene_id, str(e))


def main():
    """主函数：遍历 train/val 两个 split，调用 process_one 处理每个场景"""

    for split in ["train", "val"]:
        src = os.path.join(DATA_DIR, split, "data")
        if not os.path.exists(src):
            print(f"SKIP: {src} not found")
            continue

        # 获取所有 CSV 文件，按场景 ID 排序
        csv_files = sorted(Path(src).glob("*.csv"), key=lambda p: int(p.stem))
        total = len(csv_files)
        print(f"\n{'='*50}")
        print(f"处理 {split}: {total:,} 个场景")
        print(f"{'='*50}")

        # 创建输出目录
        cleaned_dir = os.path.join(CLEANED_DIR, split)
        processed_dir = os.path.join(PROCESSED_DIR, split)
        os.makedirs(cleaned_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # 断点续跑：跳过已处理的文件
        args_list = []
        for f in csv_files:
            sid = f.stem
            cleaned_path = os.path.join(cleaned_dir, f"{sid}.csv")
            processed_path = os.path.join(processed_dir, f"{sid}.npz")
            if os.path.exists(cleaned_path) and os.path.exists(processed_path):
                continue  # 两种格式都已存在，跳过
            args_list.append((str(f), split, cleaned_dir, processed_dir))

        skipped_existing = total - len(args_list)
        if skipped_existing > 0:
            print(f"  跳过已处理: {skipped_existing:,}, 待处理: {len(args_list):,}")
        if not args_list:
            print(f"  全部已处理!")
            continue

        # 统计变量
        stats = {
            "total": total, "valid": 0,
            "skipped": {"no_agent": 0, "no_av": 0, "short": 0, "error": 0},
            "anomaly_scenes": 0, "speed_anomalies": 0, "jump_anomalies": 0,
            "stationary": 0, "by_type": defaultdict(int),
        }

        # 断点续跑合并：加载上次运行保存的统计，与本次增量合并后再保存。
        # 否则续跑（跳过已存在文件）会用仅含本次处理的统计覆盖全量统计，
        # 导致报告中的保留率/过滤数失真（见 review M6）。
        prev_stats = {}
        stats_path = os.path.join(REPORT_DIR, f"stats_{split}.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    prev_stats = json.load(f)
            except Exception:
                prev_stats = {}

        # 多进程并行处理（限制为 4 进程，防止内存爆炸）
        workers = 4
        print(f"使用 {workers} 个进程并行处理...")
        t0 = time.time()

        with Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one, args_list, chunksize=500)):
                status = result[0]
                if status == "ok":
                    _, sid, stype, stationary, sp_anom, jmp_anom = result
                    stats["valid"] += 1
                    stats["by_type"][stype] += 1
                    if stationary:
                        stats["stationary"] += 1
                    if sp_anom > 0 or jmp_anom > 0:
                        stats["anomaly_scenes"] += 1
                    stats["speed_anomalies"] += sp_anom
                    stats["jump_anomalies"] += jmp_anom
                elif status == "skip_no_av":
                    stats["skipped"]["no_av"] += 1
                elif status == "skip_no_agent":
                    stats["skipped"]["no_agent"] += 1
                elif status == "skip_short":
                    stats["skipped"]["short"] += 1
                elif status == "error":
                    stats["skipped"]["error"] += 1

                # 进度输出
                if (i + 1) % 10000 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate / 60 if rate > 0 else 0
                    done = stats['valid'] + sum(stats['skipped'].values())
                    print(f"  [{i+1:,}/{total:,}] {rate:.0f}/s, ETA {eta:.1f}min, "
                          f"valid={stats['valid']:,}, skip={done-stats['valid']:,}")

        elapsed = time.time() - t0
        print(f"\n{split} 完成! 用时 {elapsed/60:.1f}分钟")

        # 合并历史统计（断点续跑场景）后保存
        if prev_stats:
            for k in ("valid", "anomaly_scenes", "speed_anomalies", "jump_anomalies", "stationary"):
                stats[k] += prev_stats.get(k, 0)
            for k, v in prev_stats.get("skipped", {}).items():
                stats["skipped"][k] = stats["skipped"].get(k, 0) + v
            for k, v in prev_stats.get("by_type", {}).items():
                stats["by_type"][k] = stats["by_type"].get(k, 0) + v
            stats["total"] = max(stats["total"], prev_stats.get("total", 0))

        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({**stats, "by_type": dict(stats["by_type"])},
                      f, indent=2, ensure_ascii=False, default=str)

    # 生成清洗报告
    print("\n生成清洗报告...")
    generate_report()

    print("\n✅ 全部完成!")
    print(f"  DenseTNT 数据: {CLEANED_DIR}")
    print(f"  LSTM/Kalman 数据: {PROCESSED_DIR}")
    print(f"  清洗报告: {REPORT_DIR}/data_cleaning_report.md")


def generate_report():
    """生成 Markdown 格式的清洗报告"""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 统计原始文件数
    raw_counts = {}
    for split in ["train", "val"]:
        src = os.path.join(DATA_DIR, split, "data")
        raw_counts[split] = len(list(Path(src).glob("*.csv"))) if os.path.exists(src) else 0

    # 读取统计数据
    all_stats = {}
    for split in ["train", "val"]:
        stats_path = os.path.join(REPORT_DIR, f"stats_{split}.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                all_stats[split] = json.load(f)

    # 生成报告内容
    lines = []
    lines.append("# Argoverse v1.1 数据预处理与清洗报告\n")
    lines.append("## 1. 数据集概况\n")
    lines.append("| 项目 | 训练集 | 验证集 | 合计 |")
    lines.append("|------|--------|--------|------|")
    t_raw = raw_counts.get("train", 0)
    v_raw = raw_counts.get("val", 0)
    t_val = all_stats.get("train", {}).get("valid", 0)
    v_val = all_stats.get("val", {}).get("valid", 0)
    lines.append(f"| 原始场景数 | {t_raw:,} | {v_raw:,} | {t_raw+v_raw:,} |")
    lines.append(f"| 有效场景数 | {t_val:,} | {v_val:,} | {t_val+v_val:,} |")
    lines.append(f"| 过滤场景数 | {t_raw-t_val:,} | {v_raw-v_val:,} | {t_raw+v_raw-t_val-v_val:,} |")
    if t_raw > 0 and v_raw > 0:
        lines.append(f"| 保留率 | {100*t_val/t_raw:.1f}% | {100*v_val/v_raw:.1f}% | "
                     f"{100*(t_val+v_val)/(t_raw+v_raw):.1f}% |")

    lines.append("\n## 2. 清洗流程\n")
    lines.append("### Step 1: 基本校验")
    lines.append("- CSV 格式完整性检查（至少 5 列）")
    lines.append("- 确认场景包含 AV 和 AGENT 对象类型\n")
    lines.append("### Step 2: 轨迹完整性")
    lines.append(f"- AGENT 历史帧数 < {MIN_HISTORY_LEN} → 跳过\n")
    lines.append("### Step 3: 异常检测与修复")
    lines.append(f"- 速度异常阈值: {MAX_SPEED_MS} m/s ({MAX_SPEED_MS*3.6:.0f} km/h)")
    lines.append(f"- 帧间跳跃阈值: {MAX_JUMP_DIST} m")
    lines.append("- 修复方式: 跳跃点用前一帧坐标填充\n")
    lines.append("### Step 4: 场景分类")
    lines.append("- 直行: 朝向变化 < 15°")
    lines.append("- 左转: 朝向变化 > 15° | 右转: < -15°")
    lines.append("- 复杂: 中间过渡区域\n")
    lines.append("### Step 5: 坐标变换 (方案 B)")
    lines.append("- 以 AGENT 在 frame 19 的位置为原点")
    lines.append("- 以 AGENT 朝向为 x 轴正方向旋转\n")

    lines.append("## 3. 清洗结果\n")
    for split_name, split_key in [("训练集", "train"), ("验证集", "val")]:
        s = all_stats.get(split_key, {})
        raw = raw_counts.get(split_key, 0)
        valid = s.get("valid", 0)
        if valid == 0:
            continue
        lines.append(f"### {split_name}\n")
        lines.append(f"| 指标 | 数量 | 占比 |")
        lines.append(f"|------|------|------|")
        lines.append(f"| 原始场景 | {raw:,} | 100% |")
        lines.append(f"| 有效场景 | {valid:,} | {100*valid/raw:.1f}% |")
        skipped = s.get("skipped", {})
        for reason, count in skipped.items():
            reason_cn = {"no_agent": "缺少 AGENT", "no_av": "缺少 AV",
                        "short": "历史帧不足", "error": "解析错误"}.get(reason, reason)
            lines.append(f"| 过滤: {reason_cn} | {count:,} | {100*count/raw:.2f}% |")
        lines.append(f"| 有速度异常场景 | {s.get('anomaly_scenes',0):,} | "
                     f"{100*s.get('anomaly_scenes',0)/max(valid,1):.1f}% |")
        lines.append(f"| 速度异常点总数 | {s.get('speed_anomalies',0):,} | — |")
        lines.append(f"| 跳跃点总数 | {s.get('jump_anomalies',0):,} | — |")
        lines.append(f"| 静止车辆 | {s.get('stationary',0):,} | "
                     f"{100*s.get('stationary',0)/max(valid,1):.1f}% |")

        lines.append(f"\n**场景类型分布:**\n")
        lines.append(f"| 类型 | 数量 | 占比 |")
        lines.append(f"|------|------|------|")
        by_type = s.get("by_type", {})
        for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {t} | {cnt:,} | {100*cnt/max(valid,1):.1f}% |")
        lines.append("")

    lines.append("## 4. 输出格式说明\n")
    lines.append("### `data_cleaned/{train,val}/*.csv` — DenseTNT 格式")
    lines.append("- 无表头 CSV: `TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME`")
    lines.append("- AGENT 使用清洗后坐标，其他对象保留原始坐标\n")
    lines.append("### `data_processed/{train,val}/*.npz` — LSTM/Kalman 格式")
    lines.append("- NumPy compressed 格式，包含:")
    lines.append("  - `hist` (20,2) / `gt` (30,2): agent-centered 旋转坐标")
    lines.append("  - `hist_global` / `gt_global`: 原始世界坐标")
    lines.append("  - `cent_x`, `cent_y`, `angle`: 逆变换参数")
    lines.append("  - `city`, `scene_type`, `is_stationary`\n")
    lines.append("## 5. 坐标方案对比 (2000 样本)\n")
    lines.append("| 方案 | X 范围 | Y 范围 | 终点距离 | 结论 |")
    lines.append("|------|--------|--------|----------|------|")
    lines.append("| A: AV 中心旋转 | 13.4m | 20.3m | 40.6m | 偏大 |")
    lines.append("| **B: Agent 中心旋转** | **14.2m** | **19.0m** | **29.4m** | **✓** |")
    lines.append("| C: 仅平移 | 16.1m | 20.3m | 29.4m | 范围大 |")

    report_path = os.path.join(REPORT_DIR, "data_cleaning_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告: {report_path}")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="Argoverse 1.1 preprocessing & cleaning")
    _parser.add_argument("--data-dir", default=DATA_DIR,
                         help="raw data root (default: <repo>/data/raw, "
                              "expected layout raw/{train,val}/data)")
    _parser.add_argument("--out-dir", default=REPO_ROOT,
                         help="output root for cleaned/processed/reports (default: <repo>)")
    _args = _parser.parse_args()
    DATA_DIR = _args.data_dir
    CLEANED_DIR = os.path.join(_args.out_dir, "data", "cleaned")
    PROCESSED_DIR = os.path.join(_args.out_dir, "data", "processed")
    REPORT_DIR = os.path.join(_args.out_dir, "outputs", "reports")
    main()
