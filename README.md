# 🤖 Self-Healing Classification DAG using LangGraph

## 📌 Overview

This project demonstrates a sentiment classification system that can correct itself when unsure. Built with:

- Fine-tuned DistilBERT
- DAG-based logic for fallback decisions
- CLI interface for human-in-the-loop interaction

---

## 🧠 Features

- Sentiment analysis on movie reviews (IMDb)
- Fallback node when confidence is low
- CLI-based clarification prompt
- Full log tracking of predictions and fallbacks

---

## 🛠️ Setup

```bash
pip install transformers datasets torch
