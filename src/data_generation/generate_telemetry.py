"""
Synthetic Audio Telemetry Data Generator
=========================================
Generates realistic device audio session telemetry data simulating
user behavior across Apple's consumer electronics portfolio.

Produces three core datasets:
  1. Device registry — hardware/firmware metadata per device
  2. Audio sessions — per-session telemetry events (ANC levels,
     spatial audio, voice isolation, etc.)
  3. System adjustments — granular time-series of automatic system
     parameter changes within sessions
"""

import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import load_config, resolve_path

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE_TYPES = {
    "AirPods_Pro": {"anc": True, "spatial": True, "transparency": True,
                     "conversation_awareness": True, "conversation_boost": True},
    "AirPods_Max": {"anc": True, "spatial": True, "transparency": True,
                     "conversation_awareness": True, "conversation_boost": False},
    "iPhone_15_Pro": {"anc": False, "spatial": True, "transparency": False,
                       "conversation_awareness": False, "conversation_boost": False},
    "iPhone_16_Pro": {"anc": False, "spatial": True, "transparency": False,
                       "conversation_awareness": False, "conversation_boost": False},
    "iPad_Pro_M4": {"anc": False, "spatial": True, "transparency": False,
                     "conversation_awareness": False, "conversation_boost": False},
    "MacBook_Pro_16": {"anc": False, "spatial": True, "transparency": False,
                        "conversation_awareness": False, "conversation_boost": False},
    "HomePod_2": {"anc": False, "spatial": True, "transparency": False,
                   "conversation_awareness": False, "conversation_boost": False},
    "AppleWatch_Ultra2": {"anc": False, "spatial": False, "transparency": False,
                           "conversation_awareness": False, "conversation_boost": False},
    "AppleTV_4K": {"anc": False, "spatial": True, "transparency": False,
                    "conversation_awareness": False, "conversation_boost": False},
    "VisionPro": {"anc": True, "spatial": True, "transparency": True,
                   "conversation_awareness": True, "conversation_boost": False},
}

AUDIO_MODES = [
    "media_playback", "phone_call", "facetime", "siri_interaction",
    "notification", "spatial_audio_content", "voice_memo",
    "music_streaming", "podcast", "video_conference",
]

FIRMWARE_VERSIONS = ["6A300", "6A301", "6A306", "6B29", "6C101", "6D47", "7A182"]

ENVIRONMENT_TYPES = [
    "quiet_indoor", "noisy_indoor", "outdoor_urban", "outdoor_nature",
    "commute_transit", "commute_car", "office_open_plan", "office_private",
    "gym", "airplane",
]

REGIONS = ["NA_West", "NA_East", "NA_Central", "EU_West", "EU_East",
           "APAC_East", "APAC_South", "LATAM", "MEA"]


def generate_devices(n_devices: int = 50_000) -> pd.DataFrame:
    """Generate device registry with hardware specs and firmware."""
    device_types = list(DEVICE_TYPES.keys())
    weights = [0.25, 0.08, 0.18, 0.15, 0.07, 0.08, 0.05, 0.04, 0.03, 0.07]

    records = []
    for _ in range(n_devices):
        dtype = np.random.choice(device_types, p=weights)
        records.append({
            "device_id": str(uuid.uuid4()),
            "device_type": dtype,
            "firmware_version": random.choice(FIRMWARE_VERSIONS),
            "os_version": f"1{random.randint(7,8)}.{random.randint(0,5)}",
            "region": random.choice(REGIONS),
            "registration_date": fake.date_between(
                start_date="-3y", end_date="-30d"
            ).isoformat(),
            "has_anc": DEVICE_TYPES[dtype]["anc"],
            "has_spatial_audio": DEVICE_TYPES[dtype]["spatial"],
            "has_transparency": DEVICE_TYPES[dtype]["transparency"],
            "has_conversation_awareness": DEVICE_TYPES[dtype]["conversation_awareness"],
            "has_conversation_boost": DEVICE_TYPES[dtype]["conversation_boost"],
        })
    return pd.DataFrame(records)


def generate_sessions(
    devices_df: pd.DataFrame,
    n_sessions: int = 500_000,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Generate audio session telemetry events."""
    device_ids = devices_df["device_id"].values
    device_map = devices_df.set_index("device_id").to_dict("index")

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    date_range_days = (end - start).days

    records = []
    for _ in range(n_sessions):
        did = random.choice(device_ids)
        dev = device_map[did]

        session_start = start + timedelta(
            days=random.randint(0, date_range_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        duration_sec = int(np.random.lognormal(mean=6.5, sigma=1.2))
        duration_sec = min(max(duration_sec, 5), 28800)  # 5s to 8h

        env = random.choice(ENVIRONMENT_TYPES)
        ambient_noise_db = _ambient_noise_for_env(env)

        audio_mode = random.choice(AUDIO_MODES)

        # ANC level depends on device capability + environment noise
        anc_level = 0.0
        if dev["has_anc"]:
            if ambient_noise_db > 55:
                anc_level = min(1.0, 0.4 + (ambient_noise_db - 55) / 80 +
                                np.random.normal(0, 0.08))
            else:
                anc_level = max(0.0, np.random.normal(0.15, 0.1))

        # Spatial audio engagement
        spatial_enabled = dev["has_spatial_audio"] and random.random() < 0.65
        head_tracking_active = spatial_enabled and random.random() < 0.45

        # Transparency mode
        transparency_active = (dev["has_transparency"] and
                                not (anc_level > 0.5) and
                                random.random() < 0.3)

        # Conversation awareness triggered
        conv_awareness_triggered = (dev["has_conversation_awareness"] and
                                     audio_mode in ["media_playback", "music_streaming", "podcast"] and
                                     random.random() < 0.15)

        # Voice isolation quality score (0-100)
        voice_isolation_score = None
        if audio_mode in ["phone_call", "facetime", "video_conference", "siri_interaction"]:
            base_score = 85 - (ambient_noise_db - 40) * 0.3
            voice_isolation_score = round(
                max(20, min(100, base_score + np.random.normal(0, 5))), 1
            )

        # Volume level and adjustments
        volume_level = round(np.clip(np.random.beta(2, 3) * 100, 5, 100), 1)
        volume_adjustments = int(np.random.poisson(2))

        # Bluetooth signal quality
        bt_signal_quality = None
        if dev["device_type"].startswith("AirPods"):
            bt_signal_quality = round(
                np.clip(np.random.normal(85, 12), 30, 100), 1
            )

        # Battery drain rate (% per hour)
        battery_drain_rate = round(
            np.random.exponential(3.5) + (anc_level * 2) +
            (1.5 if spatial_enabled else 0), 2
        )

        # User satisfaction proxy (implicit from session behavior)
        satisfaction_proxy = _compute_satisfaction_proxy(
            duration_sec, volume_adjustments, anc_level,
            voice_isolation_score, ambient_noise_db
        )

        records.append({
            "session_id": str(uuid.uuid4()),
            "device_id": did,
            "device_type": dev["device_type"],
            "session_start": session_start.isoformat(),
            "session_end": (session_start + timedelta(seconds=duration_sec)).isoformat(),
            "duration_seconds": duration_sec,
            "audio_mode": audio_mode,
            "environment_type": env,
            "ambient_noise_db": round(ambient_noise_db, 1),
            "anc_level": round(max(0, anc_level), 3),
            "anc_adjustments_count": int(np.random.poisson(1.5)) if anc_level > 0 else 0,
            "spatial_audio_enabled": spatial_enabled,
            "head_tracking_active": head_tracking_active,
            "transparency_mode_active": transparency_active,
            "conversation_awareness_triggered": conv_awareness_triggered,
            "voice_isolation_score": voice_isolation_score,
            "volume_level": volume_level,
            "volume_adjustments": volume_adjustments,
            "bluetooth_signal_quality": bt_signal_quality,
            "battery_drain_rate_pct_hr": battery_drain_rate,
            "audio_codec": random.choice(["AAC", "AAC-ELD", "ALAC", "Opus", "LC3"]),
            "sample_rate_hz": random.choice([16000, 44100, 48000, 96000]),
            "bitrate_kbps": random.choice([64, 128, 256, 320, 512, 1411]),
            "latency_ms": round(max(1, np.random.exponential(15)), 1),
            "audio_glitches": int(np.random.poisson(0.3)),
            "firmware_version": dev["firmware_version"],
            "region": dev["region"],
            "satisfaction_proxy": round(satisfaction_proxy, 2),
        })

    return pd.DataFrame(records)


def generate_system_adjustments(
    sessions_df: pd.DataFrame,
    max_adjustments_per_session: int = 8,
) -> pd.DataFrame:
    """Generate granular time-series of system auto-adjustments within sessions."""
    anc_sessions = sessions_df[sessions_df["anc_level"] > 0].sample(
        n=min(100_000, len(sessions_df[sessions_df["anc_level"] > 0])),
        random_state=42,
    )

    records = []
    for _, session in anc_sessions.iterrows():
        session_start = datetime.fromisoformat(session["session_start"])
        duration = session["duration_seconds"]

        # Draw the offsets once per session and sort, so timestamps are
        # distinct and monotonically increasing within the session.
        n_adj = min(random.randint(1, max_adjustments_per_session),
                    max(1, duration - 1))
        offsets = sorted(random.sample(range(1, max(2, duration)), n_adj))

        for offset_sec in offsets:
            adj_type = random.choice([
                "anc_level_change", "transparency_toggle",
                "spatial_recalibration", "volume_auto_adjust",
                "codec_switch", "eq_adaptation",
            ])

            records.append({
                "adjustment_id": str(uuid.uuid4()),
                "session_id": session["session_id"],
                "device_id": session["device_id"],
                "timestamp": (session_start + timedelta(seconds=offset_sec)).isoformat(),
                "adjustment_type": adj_type,
                "parameter_name": _param_for_adjustment(adj_type),
                "old_value": round(random.uniform(0, 1), 3),
                "new_value": round(random.uniform(0, 1), 3),
                "trigger": random.choice([
                    "auto_adaptive", "noise_spike", "user_manual",
                    "conversation_detected", "environment_change",
                    "motion_detected", "head_gesture",
                ]),
                "ambient_noise_at_adjustment": round(
                    session["ambient_noise_db"] + np.random.normal(0, 5), 1
                ),
                "response_time_ms": round(max(1, np.random.exponential(25)), 1),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ambient_noise_for_env(env: str) -> float:
    """Return realistic ambient noise level (dB) per environment."""
    noise_map = {
        "quiet_indoor": (25, 8), "noisy_indoor": (55, 10),
        "outdoor_urban": (65, 12), "outdoor_nature": (35, 8),
        "commute_transit": (75, 10), "commute_car": (60, 8),
        "office_open_plan": (50, 8), "office_private": (30, 6),
        "gym": (70, 10), "airplane": (80, 5),
    }
    mean, std = noise_map.get(env, (50, 10))
    return max(15, np.random.normal(mean, std))


def _param_for_adjustment(adj_type: str) -> str:
    params = {
        "anc_level_change": "anc_attenuation_db",
        "transparency_toggle": "transparency_gain",
        "spatial_recalibration": "head_tracking_offset_deg",
        "volume_auto_adjust": "output_volume_pct",
        "codec_switch": "codec_bitrate_kbps",
        "eq_adaptation": "eq_band_gain_db",
    }
    return params.get(adj_type, "unknown")


def _compute_satisfaction_proxy(
    duration: int, vol_adj: int, anc: float,
    voice_score: float | None, noise: float,
) -> float:
    """Heuristic satisfaction score from implicit behavioral signals."""
    score = 50.0
    # Longer sessions → higher satisfaction
    score += min(25, np.log1p(duration) * 3)
    # Fewer volume adjustments → less friction
    score -= vol_adj * 2
    # Good ANC in noisy environments
    if noise > 60 and anc > 0.5:
        score += 10
    # Voice isolation quality
    if voice_score is not None:
        score += (voice_score - 60) * 0.2
    return np.clip(score + np.random.normal(0, 3), 0, 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    gen = cfg["data_generation"]
    output_dir = resolve_path(cfg["pipeline"]["raw_data_path"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating device registry ({gen['num_devices']:,} devices)...")
    devices = generate_devices(n_devices=gen["num_devices"])
    devices.to_parquet(os.path.join(output_dir, "devices.parquet"), index=False)
    print(f"  → {len(devices):,} devices written")

    print(f"Generating audio sessions ({gen['num_sessions']:,} sessions)...")
    sessions = generate_sessions(
        devices, n_sessions=gen["num_sessions"],
        start_date=gen["date_range"]["start"], end_date=gen["date_range"]["end"],
    )
    sessions.to_parquet(os.path.join(output_dir, "sessions.parquet"), index=False)
    print(f"  → {len(sessions):,} sessions written")

    print("Generating system adjustments...")
    adjustments = generate_system_adjustments(sessions)
    adjustments.to_parquet(os.path.join(output_dir, "system_adjustments.parquet"), index=False)
    print(f"  → {len(adjustments):,} adjustments written")

    # Also write summary stats
    print("\n=== Dataset Summary ===")
    print(f"Devices:      {len(devices):>10,}")
    print(f"Sessions:     {len(sessions):>10,}")
    print(f"Adjustments:  {len(adjustments):>10,}")
    print(f"Device types: {devices['device_type'].nunique()}")
    print(f"Date range:   {sessions['session_start'].min()[:10]} → {sessions['session_start'].max()[:10]}")
    print(f"Regions:      {devices['region'].nunique()}")
    print(f"\nOutput: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
