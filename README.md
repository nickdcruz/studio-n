# Marcus — Marketing Command

AI-powered marketing team orchestration. Brief Marcus, he briefs the specialists, they run in parallel, he reviews, you download.

## Stack

- FastAPI + uvicorn
- Anthropic API (claude-opus-4-6 for Marcus, claude-sonnet-4-6 for specialists)
- GitHub API (agent system prompts fetched dynamically from `nickdcruz/nicklaus-marketing-agents`)

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=
GITHUB_TOKEN=
GITHUB_REPO=nickdcruz/nicklaus-marketing-agents
HTTP_USER=
HTTP_PASS=
```

## Local Setup

```bash
bash setup.sh
```

Runs at `http://localhost:5050`

## Railway Deployment

Set the environment variables above in Railway's dashboard. The `railway.json` and `Procfile` handle the rest.

Note: The `outputs/` folder is ephemeral on Railway. Download outputs before redeployment.
