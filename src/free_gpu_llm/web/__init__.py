"""Modular web retrieval: DDGS search + trafilatura extraction.

This is external retrieval + prompt augmentation, NOT autonomous browsing
and NOT native model tool-calling. Kept modular (search / extract / retrieval
/ pipeline as separate stages) so it can later be swapped for real tool
calling without touching the chatbot UI. See docs/web-retrieval.md.
"""
