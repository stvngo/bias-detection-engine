"""
Main application entry point for the Bias Detection Engine
"""
import os
import uvicorn
from app.api.endpoints import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="warning"
    )
