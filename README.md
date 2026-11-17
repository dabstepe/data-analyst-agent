# data-analyst

Conversational data analyst built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/overview).
Give it a CSV file from your Google Drive (a link or a file name), ask questions in
plain language, and it answers with real numbers from the data.

- **Fixed tools only, no code execution**: the agent runs with no sandbox and can only
  `load_csv`, `list_datasets`, `describe_columns`, `preview_rows`, `filter_rows` and
  `aggregate` (group-by with count/sum/mean/median/min/max/nunique, optional filters).
- **Google Drive**: the `google_drive_mcp` connector, acting as the session user, finds
  and downloads the file; the CSV text is handed to `load_csv`.
- **Memory**: the conversation session is durable; follow-ups reuse the loaded dataset
  without re-explaining.

## Local run

```bash
uv run --env-file .env agents chat
```

## Deploy (Mistral-managed)

```bash
git push -u origin HEAD
uv run --env-file .env agents deploy
```

Then chat with it by name: `uv run --env-file .env agents chat data-analyst`, or from
any program with `agents.Agent.load_from(agent_name="data-analyst")`.
