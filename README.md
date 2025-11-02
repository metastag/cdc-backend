# Basic Flask server

This project contains a minimal Flask server and a Python virtual environment.

Quick start (PowerShell):

1. Activate the virtual environment:

   .\venv\Scripts\Activate.ps1

2. Install dependencies (if you didn't already):

   pip install -r requirements.txt

3. Run the server:

   .\venv\Scripts\python.exe app.py

4. Open http://127.0.0.1:5000/ in your browser (or use curl / Invoke-WebRequest).

API
---

This server exposes a small set of endpoints for analyzing and storing short journal entries.

POST /api/analyze
- Request JSON: { "text": string }
- Response JSON: { "distortions": Distortion[], "overallScore": number, "positivePatterns": string[] }

POST /api/entries
- Create and save a journal entry.
- Request JSON: { "text": string, "mood"?: string, "meta"?: object }
- Response JSON (201): { "id": string, "text": string, "createdAt": ISOString, "summary"?: string }

GET /api/entries
- List saved entries. Response: [{ "id": string, "text": string, "createdAt": ISOString, "score"?: number, "summary"?: string }, ...]

GET /api/analysis/:entryId
- Fetch analysis for a saved entry. If analysis is not yet computed the endpoint will return 202 {"status":"pending"} and compute analysis in the background; once available it will return the same shape as POST /api/analyze.

Example (PowerShell):
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/entries -Body (ConvertTo-Json @{text='I always fail at everything'}) -ContentType 'application/json'

