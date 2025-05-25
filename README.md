# AI Image Captioning

A local, privacy-first tool that automatically generates concise, high-quality captions for your images.
Whether your photos live on your hard drive or in Google Photos, this package lets you add searchable,
descriptive text to every picture—no cloud uploads required.

## Introduction

### The problem — too many photos

If you have accumulated thousands of photos over the years, you know how hard it is to keep track of what’s
in each image. You might have an organized folder structure (I do—over 17 000 photos!), but most people don’t.
Modern smartphones capture hundreds of images without prompting you to add labels or descriptions, and browsing
manually is time-consuming.

While services like Google Photos or OneDrive offer automated albums, facial grouping, or “this day last year”
reminders, they often rely on cloud processing and can’t export structured captions for your own use. Plus, you
may not want your personal photos leaving your device.

### The goal

This package fills the gap by offering:

- **Local, private processing**  
  No images or text ever leave your machine—100 % offline by default.

- **LLM-powered captions**  
  Leverage Ollama and LangChain to generate concise, factual photo descriptions.

- **OCR integration**  
  Extract any visible text from images to guide the captioning model.

- **Google Photos ingestion**  
  Optionally pull down photos from your Google account, batch process them, and save locally.

- **Structured metadata**  
  Organize captions, extracted text, timestamps and validation flags via a Pydantic model.

By adding captions and searchable metadata to your entire photo library, you gain better searchability,
discover forgotten memories, and build your own searchable “printed” album—without ever sacrificing privacy.

---

## Installation

First, ensure you have Python 3.8 or higher installed.

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/ai-image-captioning.git
   cd ai-image-captioning
   ```

2. **(Optional) Google Photos setup**

If you plan to ingest from Google Photos, place your OAuth credentials in credentials.json at the project root.
The first time you run the loader, you’ll be prompted to authorize and a token.json will be saved.
