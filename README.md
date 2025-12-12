---
title: Quiz Analyser
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# LLM Analysis Agent

An AI-powered autonomous agent that solves data-related quiz tasks using LangGraph and OpenAI's GPT model via AI Pipe proxy.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/healthz` | GET | Health check |
| `/solve` | POST | Submit task for solving |

## Usage

Send a POST request to `/solve` with the following JSON payload:

```json
{
  "email": "your_email@example.com",
  "secret": "your_secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

## Environment Variables

- `EMAIL` - Your student email
- `SECRET` - Your secret from the Google Form
- `API_KEY` - Your AI Pipe token from aipipe.org
