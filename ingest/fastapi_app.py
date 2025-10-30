from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .cli import _ingest as ingest_impl, _register as register_impl

app = FastAPI(title="QuLab Infinite Ingestion API")

class IngestReq(BaseModel):
    source: str
    out: str

class RegisterReq(BaseModel):
    dataset: str
    name: str
    kind: str = "auto"

@app.post("/ingest")
def ingest(req: IngestReq):
    ingest_impl(req.source, req.out)
    return {"status": "ok", "out": req.out}

@app.post("/register")
def register(req: RegisterReq):
    register_impl(req.dataset, req.name, req.kind)
    return {"status": "ok", "name": req.name}
