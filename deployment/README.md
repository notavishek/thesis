---
title: Multilingual Hate Detection (Thesis)
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
---

# Multilingual Hate Speech Detection System

This is a research prototype for detecting hate speech in **Bengali**, **English**, and **Banglish** (Romanized Bengali).

## Model Details
- **Architecture:** XLM-RoBERTa Large (Fine-tuned)
- **Tasks:** Multi-task learning (Hate Type, Target Group, Severity)
- **Training Data:** Unified dataset of ~100k samples from 6 sources.

## How to Use
1. Type a sentence in the input box.
2. Click "Analyze".
3. View the predicted Hate Type, Target Group, and Severity.

## Disclaimer
This model is for academic research purposes only. It may produce incorrect or biased results. Do not use for automated content moderation without human review.
