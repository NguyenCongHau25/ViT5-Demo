---
title: ViDialect API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# ViDialect API

Docker Space chạy FastAPI cho mô hình dịch phương ngữ.

## Endpoints

- `GET /`
- `GET /health`
- `POST /generate`

## Example request

```bash
curl -X POST "https://YOUR_SPACE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"răng rứa","max_length":128}'
```

## Example response

```json
{
  "result": "sao vậy"
}
```
