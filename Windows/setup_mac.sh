#!/usr/bin/env bash
# ============================================================
# Sterling — Automated Setup Script (M1 Mac)
# ============================================================
# Run this once to install all dependencies.
# Usage: bash setup_mac.sh

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

banner() {
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  STERLING — M1 Mac Setup${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""
}

step() {
    echo -e "${BOLD}${GREEN}▶ $1${RESET}"
}

warn() {
    echo -e "${YELLOW}⚠  $1${RESET}"
}

error() {
    echo -e "${RED}✗  $1${RESET}"
}

ok() {
    echo -e "${GREEN}✓  $1${RESET}"
}

banner

# ── 1. Check prerequisites ────────────────────────────────────────────────────

step "Checking prerequisites..."

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error "This setup script is for macOS only."
    exit 1
fi

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "This machine does not appear to be Apple Silicon (M1/M2/M3)."
    warn "The script will continue, but performance may differ."
fi

# Check Python 3.11+
if ! command -v python3.11 &>/dev/null; then
    if ! python3 --version 2>&1 | grep -q "3.1[1-9]"; then
        warn "Python 3.11+ not found."
        echo "  Installing via Homebrew..."
        if command -v brew &>/dev/null; then
            brew install python@3.11
        else
            error "Homebrew not found. Install from https://brew.sh then re-run this script."
            exit 1
        fi
    fi
fi
ok "Python: $(python3 --version)"

# Check / install Homebrew dependencies
step "Checking Homebrew dependencies..."

if ! command -v brew &>/dev/null; then
    error "Homebrew is required. Install from https://brew.sh"
    exit 1
fi

deps=(portaudio ffmpeg)
for dep in "${deps[@]}"; do
    if brew list "$dep" &>/dev/null; then
        ok "$dep already installed"
    else
        echo "  Installing $dep..."
        brew install "$dep"
        ok "$dep installed"
    fi
done

# ── 2. Virtual environment ────────────────────────────────────────────────────

step "Setting up Python virtual environment (ster/)..."

if [ ! -d "ster" ]; then
    python3.11 -m venv ster
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# Activate
source ster/bin/activate
ok "Virtual environment activated"

# ── 3. Python dependencies ────────────────────────────────────────────────────

step "Installing Python packages..."
pip install --upgrade pip --quiet
pip install -r requirements_mac.txt
ok "Python packages installed"

# ── 4. Ollama ─────────────────────────────────────────────────────────────────

step "Checking Ollama..."

if ! command -v ollama &>/dev/null; then
    warn "Ollama not found."
    echo ""
    echo "  Install Ollama:"
    echo "  Option A (Homebrew): brew install ollama"
    echo "  Option B (Installer): https://ollama.ai/download"
    echo ""
    read -p "  Press Enter after installing Ollama, or Ctrl+C to cancel..."
fi

if command -v ollama &>/dev/null; then
    ok "Ollama: $(ollama --version 2>/dev/null | head -1)"

    # Start Ollama in background if not running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "  Starting Ollama server..."
        ollama serve &>/dev/null &
        sleep 3
    fi

    # Pull model
    echo "  Pulling llama3.2 (this may take a few minutes on first run)..."
    ollama pull llama3.2
    ok "llama3.2 model ready"
fi

# ── 5. Pre-download base openWakeWord models ───────────────────────────────

step "Pre-downloading openWakeWord base models..."
echo "  These are the audio embedding models shared by all wake words (~5 MB)."

python3 -c "
import openwakeword.utils
openwakeword.utils.download_models(['hey_jarvis'])
print('  Models ready.')
"
ok "Base models downloaded"

# ── 5b. Train custom Sterling wake word ────────────────────────────────────

step "Training custom 'Sterling' wake word model..."
echo "  This trains Sterling, Hey Sterling, Ster, and Ling as wake phrases."
echo "  Uses edge-tts (7 voices × 3 speeds) + noise augmentation."
echo "  Takes ~5–10 minutes. Output: assets/hey_sterling.onnx"
echo ""
read -p "  Train now? [Y/n] " train_now
train_now=${train_now:-Y}

if [[ "$train_now" =~ ^[Yy]$ ]]; then
    python3 scripts/train_wake_word.py
    ok "Wake word model trained and saved to assets/hey_sterling.onnx"
else
    warn "Skipped. Run later with: python scripts/train_wake_word.py"
    warn "Until then, Sterling will fall back to the 'hey_jarvis' pre-trained model."
    # Patch config.yaml to use hey_jarvis as fallback if training was skipped
    if [ -f "config.yaml" ]; then
        sed -i '' 's|model_path: "assets/hey_sterling.onnx"|model: "hey_jarvis"  # fallback until custom model is trained|g' config.yaml 2>/dev/null || true
    fi
fi

# ── 6. Config file ────────────────────────────────────────────────────────────

step "Setting up configuration..."

if [ ! -f "config.yaml" ]; then
    cp config.yaml.example config.yaml
    ok "config.yaml created from example"
else
    ok "config.yaml already exists"
fi

# ── 7. Final instructions ─────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  Setup complete! No API keys required.${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "${GREEN}  Wake word: 'Hey Sterling' / 'Sterling' / 'Ster' / 'Ling'${RESET}"
echo "  Powered by openWakeWord + custom-trained ONNX model."
echo "  No API key. Fully offline."
echo ""
echo -e "${YELLOW}  Optional tweaks in config.yaml:${RESET}"
echo "    wake_word.threshold: 0.5       # lower = more sensitive"
echo "    wake_word.vad_threshold: 0.5   # enable VAD in noisy rooms"
echo "    wake_word.debounce_time: 1.0   # seconds between re-triggers"
echo ""
echo "  Re-train wake word at any time:"
echo "    python scripts/train_wake_word.py"
echo "    python scripts/train_wake_word.py --phrases 'hey sterling' sterling"
echo ""
echo -e "${YELLOW}  (Optional) Connect HuskyLens2 via USB:${RESET}"
echo "    → Set vision.enabled: true in config.yaml"
echo ""
echo -e "${GREEN}  Launch Sterling:${RESET}"
echo "     source ster/bin/activate"
echo "     python main.py"
echo ""
echo "  Say 'Hey Jarvis' to wake Sterling, then speak your request."
echo ""
