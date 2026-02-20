from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from rag import Document, sha1_short
from webapp.config import AppConfig
from webapp.services.extractors import extract_text_from_path, is_allowed_file
from webapp.services.pipeline_factory import create_pipeline
from webapp.services.retrieval import fast_retrieve
from webapp.state import AppState


def register_routes(app: Flask, state: AppState, config: AppConfig, upload_dir: Path) -> None:
    def _snapshot_upload_progress() -> dict:
        with state.lock:
            progress = state.server.upload_progress
            return {
                "run_id": progress.run_id,
                "stage": progress.stage,
                "failed_stage": progress.failed_stage,
                "message": progress.message,
                "percent": progress.percent,
                "active": progress.active,
                "error": progress.error,
                "updated_at": progress.updated_at,
            }

    def _set_upload_progress(
        *,
        run_id: int,
        stage: str,
        message: str,
        percent: int,
        active: bool,
        error: str | None = None,
        failed_stage: str | None = None,
    ) -> None:
        with state.lock:
            progress = state.server.upload_progress
            if run_id < progress.run_id:
                return
            progress.run_id = run_id
            progress.stage = stage
            progress.failed_stage = failed_stage
            progress.message = message
            progress.percent = max(0, min(100, int(percent)))
            progress.active = active
            progress.error = error
            progress.updated_at = time.time()

    def _read_query_payload():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "")).strip()
        if not query:
            return None, (jsonify({"error": "Query is required."}), 400)

        with state.lock:
            pipeline = state.server.pipeline

        if pipeline is None:
            return None, (jsonify({"error": "Upload a file first."}), 400)

        return (pipeline, query), None

    @app.get("/")
    def index() -> str:
        return render_template("index.html", max_upload_mb=config.max_upload_mb)

    @app.get("/api/status")
    def status():
        with state.lock:
            progress = state.server.upload_progress
            return jsonify(
                {
                    "ready": state.server.pipeline is not None,
                    "indexed_file": state.server.indexed_file,
                    "chunk_count": state.server.chunk_count,
                    "upload_progress": {
                        "run_id": progress.run_id,
                        "stage": progress.stage,
                        "failed_stage": progress.failed_stage,
                        "message": progress.message,
                        "percent": progress.percent,
                        "active": progress.active,
                        "error": progress.error,
                        "updated_at": progress.updated_at,
                    },
                }
            )

    @app.get("/api/upload-progress")
    def upload_progress():
        return jsonify(_snapshot_upload_progress())

    @app.get("/api/chunk")
    def chunk():
        chunk_id = str(request.args.get("chunk_id", "")).strip()
        if not chunk_id:
            return jsonify({"error": "chunk_id is required."}), 400

        with state.lock:
            if state.server.pipeline is None:
                return jsonify({"error": "Upload a file first."}), 400
            ch = state.server.chunk_by_id.get(chunk_id)

        if ch is None:
            return jsonify({"error": "Chunk not found."}), 404

        return jsonify(
            {
                "chunk_id": ch.chunk_id,
                "source_path": ch.source_path,
                "chunk_index": ch.chunk_index,
                "text": ch.text,
            }
        )

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error": "No file provided."}), 400
        if not is_allowed_file(file.filename, config.allowed_extensions):
            return jsonify({"error": "Unsupported file type."}), 400

        with state.lock:
            progress = state.server.upload_progress
            progress.run_id += 1
            run_id = progress.run_id
            progress.stage = "uploading"
            progress.failed_stage = None
            progress.message = "Uploading file to server..."
            progress.percent = 8
            progress.active = True
            progress.error = None
            progress.updated_at = time.time()

        filename = secure_filename(file.filename)
        save_path = upload_dir / f"{int(time.time())}_{filename}"
        try:
            file.save(save_path)
        except Exception as exc:
            _set_upload_progress(
                run_id=run_id,
                stage="error",
                failed_stage="uploading",
                message="Upload failed while saving file.",
                percent=100,
                active=False,
                error=str(exc),
            )
            return jsonify({"error": "Failed to save uploaded file."}), 500

        _set_upload_progress(
            run_id=run_id,
            stage="uploaded",
            message="Upload complete. Preparing text extraction...",
            percent=20,
            active=True,
        )

        try:
            _set_upload_progress(
                run_id=run_id,
                stage="extracting",
                message="Extracting readable text from file...",
                percent=34,
                active=True,
            )
            text = extract_text_from_path(save_path)
        except RuntimeError as exc:
            _set_upload_progress(
                run_id=run_id,
                stage="error",
                failed_stage="extracting",
                message="Text extraction failed.",
                percent=100,
                active=False,
                error=str(exc),
            )
            return jsonify({"error": str(exc)}), 400
        if not text.strip():
            _set_upload_progress(
                run_id=run_id,
                stage="error",
                failed_stage="extracting",
                message="No text found in the uploaded file.",
                percent=100,
                active=False,
                error="Uploaded file is empty after decoding.",
            )
            return jsonify({"error": "Uploaded file is empty after decoding."}), 400

        try:
            _set_upload_progress(
                run_id=run_id,
                stage="pipeline",
                message="Initializing Gemini embedding and generation models...",
                percent=44,
                active=True,
            )
            pipeline = create_pipeline(config)
        except RuntimeError as exc:
            _set_upload_progress(
                run_id=run_id,
                stage="error",
                failed_stage="pipeline",
                message="Pipeline initialization failed.",
                percent=100,
                active=False,
                error=str(exc),
            )
            return jsonify({"error": str(exc)}), 500

        doc = Document(
            doc_id=sha1_short(f"{save_path}:{time.time_ns()}", 12),
            source_path=str(save_path),
            text=text,
        )

        def progress_callback(stage_name: str, payload: dict) -> None:
            if stage_name == "chunking_started":
                _set_upload_progress(
                    run_id=run_id,
                    stage="chunking",
                    message=f"Chunking {payload.get('total_docs', 1)} document(s)...",
                    percent=56,
                    active=True,
                )
                return
            if stage_name == "chunking_document":
                _set_upload_progress(
                    run_id=run_id,
                    stage="chunking",
                    message=f"Chunking document {payload.get('current_doc', 1)}/{payload.get('total_docs', 1)}...",
                    percent=58,
                    active=True,
                )
                return
            if stage_name == "chunking_completed":
                _set_upload_progress(
                    run_id=run_id,
                    stage="chunking",
                    message=f"Chunking complete: {payload.get('new_chunks', 0)} chunks generated.",
                    percent=62,
                    active=True,
                )
                return
            if stage_name == "embedding_started":
                new_chunks = payload.get("new_chunks", 0)
                _set_upload_progress(
                    run_id=run_id,
                    stage="embedding",
                    message=f"Generating embeddings for {new_chunks} chunks...",
                    percent=72,
                    active=True,
                )
                return
            if stage_name in {"embedding_completed", "indexing_started"}:
                _set_upload_progress(
                    run_id=run_id,
                    stage="indexing",
                    message="Inserting vectors in memory and building indexes...",
                    percent=88,
                    active=True,
                )
                return
            if stage_name == "complete":
                _set_upload_progress(
                    run_id=run_id,
                    stage="indexing",
                    message="Finalizing index...",
                    percent=95,
                    active=True,
                )

        start = time.perf_counter()
        try:
            pipeline.ingest_documents([doc], progress_callback=progress_callback)
        except Exception as exc:
            _set_upload_progress(
                run_id=run_id,
                stage="error",
                failed_stage="indexing",
                message="Indexing failed.",
                percent=100,
                active=False,
                error=str(exc),
            )
            return jsonify({"error": f"Ingestion failed: {exc}"}), 500
        ingest_seconds = round(time.perf_counter() - start, 3)

        with state.lock:
            state.server.pipeline = pipeline
            state.server.indexed_file = filename
            state.server.chunk_count = len(pipeline._chunks)
            state.server.chunk_by_id = {c.chunk_id: c for c in pipeline._chunks}

        _set_upload_progress(
            run_id=run_id,
            stage="complete",
            message="Index is ready. You can run queries now.",
            percent=100,
            active=False,
        )

        return jsonify(
            {
                "message": "File uploaded and indexed.",
                "filename": filename,
                "size_bytes": save_path.stat().st_size,
                "chunks_indexed": len(pipeline._chunks),
                "ingest_seconds": ingest_seconds,
                "run_id": run_id,
            }
        )

    @app.post("/api/retrieve")
    def retrieve():
        data, error = _read_query_payload()
        if error:
            return error
        pipeline, query = data

        try:
            result = fast_retrieve(pipeline, query, config.query_settings)
        except Exception as exc:
            return jsonify({"error": f"Retrieve failed: {exc}"}), 500

        return jsonify(result)

    @app.post("/api/answer")
    def answer():
        data, error = _read_query_payload()
        if error:
            return error
        pipeline, query = data

        try:
            result = pipeline.answer(
                query,
                mode="qa",
                dense_top_k=config.query_settings.dense_top_k,
                sparse_top_k=config.query_settings.sparse_top_k,
                fuse_top_k=config.query_settings.fuse_top_k,
                rerank_k=config.query_settings.rerank_k,
                keep_n=config.query_settings.keep_n,
                token_budget=config.query_settings.token_budget,
                neighbor_window=0,
                use_multi_query=True,
                use_llm_query_rewrite=False,
                use_reranker=False,
                compress_if_needed=True,
            )
        except Exception as exc:
            return jsonify({"error": f"Answer failed: {exc}"}), 500

        debug = result.get("debug", {})
        return jsonify(
            {
                "answer": result.get("answer", ""),
                "matches": result.get("closest_matches", [])[: config.query_settings.retrieve_top_k],
                "sources": result.get("used_chunks", []),
                "queries_used": debug.get("queries_used", []),
                "top_fused_score": debug.get("top_fused_score"),
            }
        )

    @app.post("/api/query")
    def query():
        return answer()
