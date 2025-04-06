Goal:
We want to detect AI-generated human speech (deepfakes), possibly in real-time, during real conversations.
We looked at 3 strong models for this task:

🔹 1. Wav2Vec2.0
What it is:

A powerful model from Facebook that learns directly from raw audio (no need for spectrograms or MFCCs).

Pretrained on a lot of speech, so it already understands audio well.

Why it’s good:

Works with raw waveforms (less pre-processing).

Already trained on speech — we just need to fine-tune it.

Good for real-time use.

Results:

Very low error rate (EER around 1–3%) when fine-tuned.

Limitations:

Needs a classifier on top to make decisions (e.g., spoof or bonafide).

Can be heavy on memory.

🔹 2. RawNet2
What it is:

A model designed specifically for detecting fake audio.

Uses CNN and GRU (a type of RNN) to understand patterns in raw waveforms.

Why it’s good:

Built for spoof detection.

No need to convert audio to spectrograms.

Top performer in many audio deepfake competitions.

Results:

Error rate (EER) is ~1–2%, very strong.

Limitations:

Training from scratch takes time and resources.

Slightly bigger model, might not be ideal for real-time unless optimized.

🔹 3. LCNN (Lightweight CNN)
What it is:

A smaller, faster CNN model that uses spectrograms as input.

Often used as a simple baseline for spoof detection.

Why it’s good:

Lightweight and fast — works on low-power devices.

Easy to train and modify.

Results:

EER between 4–6%, decent for small models.

Limitations:

Needs spectrograms — extra step before training.

Doesn’t learn long-term patterns as well as others.
