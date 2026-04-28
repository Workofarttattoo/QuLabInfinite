import sys

sys.path.insert(0, "/app/qulab")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("qulab.mcp.ech0_mcp_lite_public:app", host="0.0.0.0", port=8000)
