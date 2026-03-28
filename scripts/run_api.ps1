param(
    [int]$Port = 8000
)

uvicorn src.api.server:app --host 0.0.0.0 --port $Port --reload
