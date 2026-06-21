#!/usr/bin/env python3
"""
Sterling Wake Word Trainer
==========================
Trains a custom openWakeWord-compatible ONNX model that recognises any
combination of wake phrases you specify (e.g. "hey sterling", "sterling",
"ster", "ling").

How it works
------------
1.  edge-tts generates spoken audio for each wake phrase across multiple
    voices and speaking speeds — this is your positive training data.
2.  edge-tts generates audio for ~150 diverse English phrases that are NOT
    the wake word — negative training data.
3.  Both sets are augmented with Gaussian noise at several SNR levels.
4.  openWakeWord's ONNX embedding pipeline converts every 2-second audio
    clip into a [16, 96] feature array (exactly what the wake word model
    expects as input).
5.  A scikit-learn MLPClassifier is trained on the flattened [1536] features.
6.  The trained weights are baked into an ONNX graph with the right I/O
    shape ([1, 16, 96] → [1, 1]) so Sterling can load it directly.

Output
------
    assets/hey_sterling.onnx

Usage
-----
    source ster/bin/activate
    python scripts/train_wake_word.py
    python scripts/train_wake_word.py --phrases "hey sterling" sterling ster ling
    python scripts/train_wake_word.py --out assets/custom_wake.onnx
    python scripts/train_wake_word.py --quick     # fewer samples, fast test run

Requirements (all already installed)
-------------------------------------
    edge-tts, openwakeword, onnxruntime, onnx, scikit-learn, numpy, scipy
"""

import argparse
import asyncio
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# ─── ONNX graph building ─────────────────────────────────────────────────────
import onnx
from onnx import helper, numpy_helper, TensorProto

# ─── openWakeWord feature extractor ──────────────────────────────────────────
import openwakeword.utils
from openwakeword.utils import AudioFeatures


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PHRASES = ["hey sterling", "sterling", "ster", "ling"]

# Clip duration: 2 seconds at 16 kHz → exactly [16, 96] embedding window
CLIP_SAMPLES = 32000
SAMPLE_RATE = 16000

# TTS voices used for positive samples (diverse accent/gender coverage)
TTS_VOICES = [
    "en-US-ChristopherNeural",   # male, US
    "en-US-JennyNeural",         # female, US
    "en-GB-RyanNeural",          # male, UK (Sterling's own voice)
    "en-US-GuyNeural",           # male, US
    "en-US-AriaNeural",          # female, US
    "en-AU-WilliamNeural",       # male, Australian
    "en-IE-ConnorNeural",        # male, Irish
]

# Speaking rate variations for positive samples (+robustness to speed)
TTS_RATES = ["-15%", "+0%", "+20%"]

# Gaussian noise SNR levels (dB) added to each sample (positive & negative)
NOISE_SNR_LEVELS = [20, 10, 5]  # quieter = more noise

# MLP hidden layer sizes
MLP_HIDDEN = (128, 64)

# Opset version for the exported ONNX model
ONNX_OPSET = 13


# ─────────────────────────────────────────────────────────────────────────────
# Negative phrase bank
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_PHRASES = [
    # Greetings & small talk
    "hello there", "good morning", "how are you", "nice to meet you",
    "have a great day", "see you later", "take care", "you're welcome",
    "thank you very much", "no problem at all",
    # Questions
    "what time is it", "where are we going", "can you hear me",
    "what is the weather today", "how long will that take",
    "what did you say", "could you say that again",
    "how many people are coming", "who called", "when does it start",
    # Commands & requests
    "turn on the lights", "play some music", "set a timer for ten minutes",
    "open the garage door", "lock the front door", "turn off the television",
    "dim the lights please", "start the dishwasher", "order more coffee",
    "send a message", "call my phone", "navigate home",
    # Numbers & dates
    "one two three four five", "the year two thousand and twenty five",
    "january the first", "at three thirty pm", "forty seven degrees",
    "nine hundred dollars", "a hundred and twelve miles",
    # Technology & names
    "amazon echo", "google assistant", "apple siri", "microsoft cortana",
    "open artificial intelligence", "large language model",
    "neural network", "machine learning", "deep learning",
    "computer vision", "natural language processing",
    # Random sentences
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "the rain in spain stays mainly in the plain",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the early bird catches the worm",
    "actions speak louder than words",
    "every cloud has a silver lining",
    "a stitch in time saves nine",
    # Short words that could confuse ("sterl" doesn't exist but "ster"-adjacent)
    "faster", "blister", "master", "plaster", "disaster",
    "mister", "sister", "oyster", "cluster", "roster",
    "longer", "finger", "singer", "linger", "ginger",
    "spring", "string", "strong", "bring", "ring",
    "sterling silver", "a sterling example", "the sterling pound",
    # More natural conversation
    "i would like a cup of coffee please",
    "let me check my calendar",
    "the meeting starts at nine am",
    "please remind me in one hour",
    "i will be back in five minutes",
    "the project deadline is next friday",
    "can you turn down the volume",
    "i need to charge my phone",
    "what is on the menu today",
    "check the front door camera",
    "is anyone home right now",
    "what movies are playing tonight",
    "add milk to the shopping list",
    "what is the score of the game",
    "play my morning playlist",
    "increase the thermostat by two degrees",
    "how is traffic on the highway",
    "wake me up at seven tomorrow",
    "send a text to john saying i am on my way",
    "how many calories are in an apple",
    "what is the capital of france",
    "translate hello into spanish",
    "set the alarm for six thirty am",
    "find a good italian restaurant nearby",
    "tell me a joke",
    "what is the latest news",
    "read my emails",
    "skip this song",
    "lower the volume",
    # Phonetically close but wrong
    "always sterling", "a stirring performance", "strong feeling",
    "start the engine", "is it certain", "curtain call",
    "concerning news", "earning a living", "burning sensation",
    "starring role", "turning point", "learning curve",
    "earning potential", "discerning taste", "determining factor",
]


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _tts_bytes_async(text: str, voice: str, rate: str) -> bytes:
    """Run edge-tts and return raw MP3 bytes."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def tts_to_audio(text: str, voice: str, rate: str = "+0%") -> np.ndarray:
    """
    Synthesise text → 16 kHz int16 numpy array via edge-tts + ffmpeg.
    Returns the raw audio (variable length).
    """
    mp3_bytes = asyncio.run(_tts_bytes_async(text, voice, rate))

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, "audio.mp3")
        wav_path = os.path.join(tmp, "audio.wav")

        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)

        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-sample_fmt", "s16", wav_path],
            capture_output=True, check=True,
        )

        _, audio = wavfile.read(wav_path)

    return audio.astype(np.int16)


def pad_to_clip(audio: np.ndarray, label: int) -> np.ndarray:
    """
    Resize audio to exactly CLIP_SAMPLES (2 seconds).
    - Positive (label=1): silence first, audio at the END  (model sees word last)
    - Negative (label=0): pad/truncate however — the content fills the window
    """
    target = CLIP_SAMPLES
    n = len(audio)

    if n >= target:
        # Truncate — keep the END for positives, START for negatives
        if label == 1:
            return audio[-target:].astype(np.int16)
        else:
            return audio[:target].astype(np.int16)
    else:
        if label == 1:
            # Pad with silence at the beginning so the word ends at the clip boundary
            pad = np.zeros(target - n, dtype=np.int16)
            return np.concatenate([pad, audio])
        else:
            # Pad with silence at the end
            pad = np.zeros(target - n, dtype=np.int16)
            return np.concatenate([audio, pad])


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian white noise at the given signal-to-noise ratio (dB)."""
    rms_signal = np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-9)
    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    noise = np.random.randn(len(audio)).astype(np.float32) * rms_noise
    noisy = audio.astype(np.float32) + noise
    return np.clip(noisy, -32768, 32767).astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(clips: np.ndarray, af: AudioFeatures) -> np.ndarray:
    """
    clips : (N, 32000) int16
    returns : (N, 16, 96) float32
    """
    return af.embed_clips(clips, batch_size=32)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_onnx_from_mlp(clf: MLPClassifier, out_path: str):
    """
    Converts a trained sklearn MLPClassifier into an ONNX model with:
      Input  : 'x.1'  [1, 16, 96]  float32
      Output : '53'   [1,  1]       float32  (sigmoid probability)

    This is the exact I/O signature openWakeWord expects.
    """
    n_layers = len(clf.coefs_)

    # Build node + initializer lists
    nodes = []
    initializers = []

    # ── Step 1: Reshape [1, 16, 96] → [1, 1536] ──────────────────────────────
    shape_val = numpy_helper.from_array(
        np.array([1, 16 * 96], dtype=np.int64), name="reshape_shape"
    )
    initializers.append(shape_val)
    nodes.append(helper.make_node("Reshape", ["x.1", "reshape_shape"], ["flat"]))

    # ── Steps 2..N-1: hidden Gemm + ReLU ─────────────────────────────────────
    prev = "flat"
    for i in range(n_layers - 1):
        W = clf.coefs_[i].astype(np.float32)    # [n_in, n_out]
        b = clf.intercepts_[i].astype(np.float32)  # [n_out]

        w_name = f"W{i}"
        b_name = f"b{i}"
        h_name = f"h{i}"
        r_name = f"r{i}"

        initializers.append(numpy_helper.from_array(W, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))

        # Gemm: Y = A @ B + C  (transB=0 → B shape [n_in, n_out])
        nodes.append(helper.make_node(
            "Gemm", [prev, w_name, b_name], [h_name],
            alpha=1.0, beta=1.0, transA=0, transB=0
        ))
        nodes.append(helper.make_node("Relu", [h_name], [r_name]))
        prev = r_name

    # ── Final layer: Gemm + Sigmoid ───────────────────────────────────────────
    W_out = clf.coefs_[-1].astype(np.float32)
    b_out = clf.intercepts_[-1].astype(np.float32)

    initializers.append(numpy_helper.from_array(W_out, name="W_out"))
    initializers.append(numpy_helper.from_array(b_out, name="b_out"))

    nodes.append(helper.make_node(
        "Gemm", [prev, "W_out", "b_out"], ["logit"],
        alpha=1.0, beta=1.0, transA=0, transB=0
    ))
    nodes.append(helper.make_node("Sigmoid", ["logit"], ["53"]))

    # ── Assemble graph ────────────────────────────────────────────────────────
    graph = helper.make_graph(
        nodes,
        "hey_sterling_wake_word",
        inputs=[helper.make_tensor_value_info("x.1",  TensorProto.FLOAT, [1, 16, 96])],
        outputs=[helper.make_tensor_value_info("53", TensorProto.FLOAT, [1, 1])],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"  Saved ONNX model → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def self_test(onnx_path: str, af: AudioFeatures, pos_audio: np.ndarray, neg_audio: np.ndarray):
    """
    Quick sanity check: run a few positive and negative clips through the
    freshly exported ONNX model and print the scores.
    """
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)

    def score_clip(audio_1d: np.ndarray) -> float:
        clip = pad_to_clip(audio_1d, label=1)  # shape doesn't affect onnx, just size
        feat = af.embed_clips(clip[None, :])    # [1, 16, 96]
        result = session.run(None, {"x.1": feat.astype(np.float32)})
        return float(result[0][0, 0])

    print("\n  Self-test scores (expect positive > 0.5, negative < 0.5):")
    for i, a in enumerate(pos_audio[:3]):
        s = score_clip(a)
        ok = "✓" if s > 0.4 else "✗"
        print(f"    {ok} Positive [{i}]: {s:.3f}")
    for i, a in enumerate(neg_audio[:3]):
        s = score_clip(a)
        ok = "✓" if s < 0.6 else "✗"
        print(f"    {ok} Negative [{i}]: {s:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    phrases: list[str],
    voices: list[str],
    rates: list[str],
    neg_phrases: list[str],
    out_path: str,
    quick: bool,
):
    t0 = time.time()

    # Reduce scope for quick test mode
    if quick:
        voices = voices[:2]
        rates = ["+ 0%"]
        neg_phrases = neg_phrases[:30]
        print("  [quick mode — fewer samples, for testing only]")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  STEP 1 — Ensure openWakeWord base models are downloaded")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    openwakeword.utils.download_models(["hey_jarvis"])  # pulls melspec + embedding too
    print("  Base models ready.")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  STEP 2 — Generate POSITIVE samples ({len(phrases)} phrases × "
          f"{len(voices)} voices × {len(rates)} speeds)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Wake phrases:", ", ".join(f'"{p}"' for p in phrases))

    pos_raw = []   # list of 1-D int16 arrays (variable length)
    fail_count = 0

    for phrase in phrases:
        for voice in voices:
            for rate in rates:
                try:
                    audio = tts_to_audio(phrase, voice, rate)
                    pos_raw.append(audio)
                    print(f"  ✓  '{phrase}' | {voice} | {rate}")
                except Exception as e:
                    fail_count += 1
                    print(f"  ✗  '{phrase}' | {voice} | {rate}  — {e}")

    if not pos_raw:
        print("\nERROR: No positive samples could be generated. "
              "Check that edge-tts and ffmpeg are installed.")
        sys.exit(1)

    print(f"\n  Generated {len(pos_raw)} positive clips "
          f"({'  ' + str(fail_count) + ' failed' if fail_count else 'all succeeded'})")

    # Pad every positive clip to exactly CLIP_SAMPLES (wake word at the end)
    pos_clipped = np.array([pad_to_clip(a, label=1) for a in pos_raw], dtype=np.int16)

    # Augment: add noise copies
    pos_augmented = [pos_clipped]
    for snr in NOISE_SNR_LEVELS:
        noisy = np.array([add_noise(a, snr) for a in pos_clipped], dtype=np.int16)
        pos_augmented.append(noisy)
    pos_all = np.concatenate(pos_augmented, axis=0)

    print(f"  Total positives after augmentation: {len(pos_all)}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  STEP 3 — Generate NEGATIVE samples ({len(neg_phrases)} phrases × "
          f"{min(3, len(voices))} voices)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    neg_raw = []
    neg_voices = voices[:3]  # use 3 voices for negatives to keep it reasonable

    for phrase in neg_phrases:
        for voice in neg_voices:
            try:
                audio = tts_to_audio(phrase, voice, "+0%")
                neg_raw.append(audio)
            except Exception:
                pass

    # Add silence and pure noise clips
    print(f"  Adding silence + noise clips...")
    for _ in range(30):
        neg_raw.append(np.zeros(CLIP_SAMPLES, dtype=np.int16))          # silence
    for snr in [30, 20, 10]:
        for _ in range(30):
            noise_clip = add_noise(np.zeros(CLIP_SAMPLES, dtype=np.int16), snr)
            neg_raw.append(noise_clip)

    neg_clipped = np.array([pad_to_clip(a, label=0) for a in neg_raw], dtype=np.int16)

    # Augment negatives (smaller factor — we already have many)
    neg_augmented = [neg_clipped]
    for snr in NOISE_SNR_LEVELS[:2]:
        noisy = np.array([add_noise(a, snr) for a in neg_clipped], dtype=np.int16)
        neg_augmented.append(noisy)
    neg_all = np.concatenate(neg_augmented, axis=0)

    print(f"  Total negatives after augmentation: {len(neg_all)}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  STEP 4 — Extract openWakeWord embeddings [N, 16, 96]")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    af = AudioFeatures(inference_framework="onnx")

    print(f"  Extracting features from {len(pos_all)} positive clips...")
    X_pos = extract_features(pos_all, af)   # [N, 16, 96]
    print(f"  Extracting features from {len(neg_all)} negative clips...")
    X_neg = extract_features(neg_all, af)   # [N, 16, 96]

    # Flatten for sklearn: [N, 16*96]
    X_pos_flat = X_pos.reshape(len(X_pos), -1)
    X_neg_flat = X_neg.reshape(len(X_neg), -1)

    X = np.concatenate([X_pos_flat, X_neg_flat], axis=0)
    y = np.concatenate([
        np.ones(len(X_pos_flat), dtype=int),
        np.zeros(len(X_neg_flat), dtype=int),
    ])

    X, y = shuffle(X, y, random_state=42)

    print(f"  Dataset: {len(X_pos_flat)} positive  /  {len(X_neg_flat)} negative")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  STEP 5 — Train MLPClassifier  hidden={MLP_HIDDEN}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    clf = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN,
        activation="relu",
        solver="adam",
        alpha=1e-4,          # L2 regularisation
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    accuracy = (y_pred == y).mean()
    pos_recall = (y_pred[y == 1] == 1).mean()
    neg_precision = (y_pred[y == 0] == 0).mean()

    print(f"  Training accuracy  : {accuracy:.1%}")
    print(f"  Positive recall    : {pos_recall:.1%}  (how often wake word fires)")
    print(f"  Negative precision : {neg_precision:.1%}  (how often silence is quiet)")

    if pos_recall < 0.70:
        print("\n  WARNING: Positive recall is low. Consider:")
        print("    • Adding more TTS voices (--voices flag or edit TTS_VOICES)")
        print("    • Recording real speech samples (see README note)")
        print("    • Using a slightly lower threshold in config.yaml (0.4)")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  STEP 6 — Build and save ONNX model")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    build_onnx_from_mlp(clf, out_path)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  STEP 7 — Self-test")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    self_test(out_path, af, pos_raw, neg_raw)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min.")
    print(f"\n  ✅  Model saved to: {out_path}")
    print(textwrap.dedent(f"""
  Now update config.yaml:
  ─────────────────────────────────────────
  wake_word:
    model_path: "{out_path}"
    threshold: 0.5        # lower (0.4) = more sensitive
    debounce_time: 1.0
  ─────────────────────────────────────────
  Say: 'hey sterling' (or any trained phrase) to activate.
    """))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom Sterling wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scripts/train_wake_word.py
          python scripts/train_wake_word.py --phrases "hey sterling" sterling ster
          python scripts/train_wake_word.py --out assets/ster.onnx --quick
        """),
    )
    parser.add_argument(
        "--phrases", nargs="+", default=DEFAULT_PHRASES,
        metavar="PHRASE",
        help="Wake phrases to train on (default: hey sterling, sterling, ster, ling)",
    )
    parser.add_argument(
        "--out", default="assets/hey_sterling.onnx",
        metavar="PATH",
        help="Output ONNX model path (default: assets/hey_sterling.onnx)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test run — fewer samples, fewer voices (not for production use)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "━" * 62)
    print("  STERLING WAKE WORD TRAINER")
    print("━" * 62)
    print(f"  Phrases : {args.phrases}")
    print(f"  Output  : {args.out}")
    print(f"  Voices  : {len(TTS_VOICES)}")
    print(f"  Speeds  : {TTS_RATES}")

    train(
        phrases=args.phrases,
        voices=TTS_VOICES,
        rates=TTS_RATES,
        neg_phrases=NEGATIVE_PHRASES,
        out_path=args.out,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
