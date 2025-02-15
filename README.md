NotesAI 📚🤖

An AI-powered tool that helps students prepare for exams by summarizing educational content from PDFs, Wikipedia, and YouTube transcripts.

🚀 Objective

NotesAI simplifies study material by automatically summarizing long-form content into concise, easy-to-read notes. It utilizes AI models to extract key insights, reducing the time students spend reviewing large amounts of information.

❓ Need to Build This

Currently, AI platforms do not directly access content from links due to security and privacy restrictions. This means:

AI models cannot fetch content from Wikipedia directly; users must manually copy and paste the relevant text.

For YouTube videos, AI cannot summarize directly from links; users must first download the transcript and then process it for summarization and Q&A.

NotesAI addresses these limitations by providing a structured way to input text and generate summaries, ensuring a smoother study experience.

🎯 Why NotesAI?

Students often face these challenges:

Large study materials 📚 – Too much content to read before exams.

Time constraints ⏳ – Need quick and effective revision.

Retention difficulties 🧠 – Struggle to remember key points.

NotesAI solves these problems by leveraging LLama AI models and Groq API to generate summaries and answer questions from various sources.

🛠️ Features

✅ PDF Summarization: Extracts and summarizes text from PDFs..✅ Wikipedia Summarization: Users can manually input Wikipedia content for structured summarization.✅ YouTube Transcript Summarization: Requires users to first download the transcript, which is then processed for key insights.✅ Question & Answer System: Uses AI to answer user-generated questions from the summarized content.✅ AI-Powered Processing: Utilizes LLama-3.1-8B-Instant and Groq API for fast and efficient summarization.

🏷️ Tech Stack

Python 🐍

Jupyter Notebook 📚

PyMuPDF (fitz) & OCR (pytesseract) 📝

BeautifulSoup4 (Web Scraping Wikipedia - Manual Input Required) 🌐

YouTube Transcript API (Requires Manual Transcript Download) 🎥

Groq API & LLama-3.1-8B-Instant 🤙

Requests, Re (Regex), and Logging
