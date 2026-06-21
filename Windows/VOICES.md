# Sterling — Edge-TTS Voice Reference

Set the voice in **`config.yaml`** under `tts.voice`:

```yaml
tts:
  voice: "en-GB-RyanNeural"   # ← change this
```

> All voices below are English-language Edge-TTS neural voices.
> Rate and pitch can also be tuned in `config.yaml`:
> - `rate`: `-50%` (slow) → `+100%` (fast) — default `+5%`
> - `pitch`: `-50Hz` (deeper) → `+50Hz` (higher) — default `-5Hz`

---

## 🇬🇧 British English (en-GB) — *Current default region*

| Voice ID | Gender | Notes |
|---|---|---|
| `en-GB-RyanNeural` | Male | ⭐ **Current default** — Jarvis-like, professional |
| `en-GB-ThomasNeural` | Male | Warm British male |
| `en-GB-SoniaNeural` | Female | Clear, professional |
| `en-GB-LibbyNeural` | Female | Friendly, natural |
| `en-GB-MaisieNeural` | Female | Younger, upbeat |

---

## 🇺🇸 American English (en-US)

| Voice ID | Gender | Style | Notes |
|---|---|---|---|
| `en-US-AndrewNeural` | Male | Conversation | Warm, confident, authentic |
| `en-US-AndrewMultilingualNeural` | Male | Conversation | Same as Andrew + multilingual |
| `en-US-BrianNeural` | Male | Conversation | Approachable, casual, sincere |
| `en-US-ChristopherNeural` | Male | News | Reliable, authoritative |
| `en-US-EricNeural` | Male | News | Rational, clear |
| `en-US-GuyNeural` | Male | News | Passionate delivery |
| `en-US-RogerNeural` | Male | News | Lively |
| `en-US-SteffanNeural` | Male | News | Rational, steady |
| `en-US-AriaNeural` | Female | News | Positive, confident |
| `en-US-AvaNeural` | Female | Conversation | Expressive, caring |
| `en-US-AvaMultilingualNeural` | Female | Conversation | Same as Ava + multilingual |
| `en-US-EmmaNeural` | Female | Conversation | Cheerful, clear |
| `en-US-JennyNeural` | Female | General | Friendly, considerate |
| `en-US-MichelleNeural` | Female | News | Friendly, pleasant |

---

## 🇦🇺 Australian English (en-AU)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-AU-WilliamMultilingualNeural` | Male | Friendly, multilingual |
| `en-AU-NatashaNeural` | Female | Friendly, positive |

---

## 🇨🇦 Canadian English (en-CA)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-CA-LiamNeural` | Male | Friendly, positive |
| `en-CA-ClaraNeural` | Female | Friendly, positive |

---

## 🇮🇪 Irish English (en-IE)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-IE-ConnorNeural` | Male | Irish accent |
| `en-IE-EmilyNeural` | Female | Irish accent |

---

## 🇮🇳 Indian English (en-IN)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-IN-PrabhatNeural` | Male | Indian accent |
| `en-IN-NeerjaNeural` | Female | Indian accent |
| `en-IN-NeerjaExpressiveNeural` | Female | Expressive variant |

---

## 🇳🇿 New Zealand (en-NZ)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-NZ-MitchellNeural` | Male | NZ accent |
| `en-NZ-MollyNeural` | Female | NZ accent |

---

## 🇸🇬 Singapore (en-SG)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-SG-WayneNeural` | Male | Singaporean accent |
| `en-SG-LunaNeural` | Female | Singaporean accent |

---

## 🇿🇦 South African (en-ZA)

| Voice ID | Gender | Notes |
|---|---|---|
| `en-ZA-LukeNeural` | Male | SA accent |
| `en-ZA-LeahNeural` | Female | SA accent |

---

## Other Regions

| Voice ID | Gender | Region |
|---|---|---|
| `en-HK-SamNeural` | Male | Hong Kong |
| `en-HK-YanNeural` | Female | Hong Kong |
| `en-KE-ChilembaNeural` | Male | Kenya |
| `en-KE-AsiliaNeural` | Female | Kenya |
| `en-NG-AbeoNeural` | Male | Nigeria |
| `en-NG-EzinneNeural` | Female | Nigeria |
| `en-PH-JamesNeural` | Male | Philippines |
| `en-PH-RosaNeural` | Female | Philippines |
| `en-TZ-ElimuNeural` | Male | Tanzania |
| `en-TZ-ImaniNeural` | Female | Tanzania |

---

## macOS `say` Fallback Voices

Used when Edge-TTS fails (offline). Set in `config.yaml` under `tts.fallback_voice`:

```yaml
tts:
  fallback_voice: "Alex"   # ← macOS say voice
```

| Voice | Gender | Notes |
|---|---|---|
| `Alex` | Male | ⭐ Default — clear American |
| `Daniel` | Male | British accent |
| `Fred` | Male | Robotic, old-school |
| `Tom` | Male | Natural American |
| `Samantha` | Female | Natural American |
| `Victoria` | Female | American |
| `Fiona` | Female | Scottish accent |
| `Moira` | Female | Irish accent |
| `Tessa` | Female | South African |

> Run `say -v ?` in Terminal to see all voices installed on your Mac.

---

## Quick Test

Test any voice without restarting Sterling:

```bash
source ster/bin/activate
edge-tts --voice "en-GB-ThomasNeural" --text "Sterling online. All systems nominal." --write-media /tmp/test.mp3 && afplay /tmp/test.mp3
```
