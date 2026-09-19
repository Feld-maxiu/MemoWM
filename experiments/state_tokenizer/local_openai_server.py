"""A minimal OpenAI-compatible chat endpoint backed by a local Qwen3.5 VLM.

The WorldMemArena evaluator drives its answer and judge stages through the
OpenAI SDK, so running it against local weights needs an endpoint rather than a
library call. vLLM is not an option here: the only environment that ships it has
vllm 0.11.0 against torch 2.11, and 0.11.0 pins torch 2.8 -- the extension fails
to load with an undefined c10::cuda symbol. Fixing that means mutating a shared
conda environment that belongs to someone else, so this serves the same role
with the standard library and the transformers stack already proven on these
weights.

Only what the evaluator actually calls is implemented: ``POST
/v1/chat/completions`` with ``messages``/``max_tokens``/``temperature``, text or
``image_url`` content parts, and a ``usage`` block (the framework's token
accounting reads it, and the efficiency claim is measured from it).

Throughput comes from batching plus multiple worker processes. Each worker owns
one model copy and a queue; text-only requests are decoded together, image ones
alone. Workers share the port via SO_REUSEPORT because threads alone do not
scale -- generate()'s loop is Python-level and contends on the GIL.

    python -m experiments.state_tokenizer.local_openai_server \
        --model models/Qwen3.5-9B --gpus 0,1,2,3 --port 8000

Then point the evaluator at it:

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    OPENAI_API_KEY=local        # any non-empty string; cli.py:435 returns the
                                # gold answer outright when this is unset

Determinism: ``temperature=0`` maps to greedy decoding, which is what the judge
runs at. Sampled generation is seeded per request from the request body so a
rerun of the same conversation reproduces.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class Replica:
    """One model copy pinned to one device, serialized by its own lock."""

    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.lock = threading.Lock()

    @torch.inference_mode()
    def generate(self, messages, images, max_tokens: int, temperature: float,
                 seed: int, thinking: bool):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = float(temperature)
            torch.manual_seed(seed)
        output = self.model.generate(**inputs, **kwargs)
        completion = output[0, prompt_tokens:]
        content = self.processor.tokenizer.decode(completion, skip_special_tokens=True)
        return content, prompt_tokens, int(completion.shape[0])

    @torch.inference_mode()
    def generate_batch(self, jobs: list["Job"]):
        """Decode several text-only prompts in one pass.

        One sequence at a time leaves a 9B almost idle -- decode is
        memory-bandwidth bound and a batch of one wastes nearly all of it,
        measured at ~10 tok/s per worker against 30-40 for a single stream on
        an unshared card. Batching is where the throughput is, and the judge
        calls that dominate the run are text-only, so they batch cleanly.
        Requests carrying images fall back to the single path: their pixel
        tensors have per-request shapes that would need separate handling for
        no benefit at this share of traffic.

        Left padding, because generation continues from the last position and
        right padding would start every short prompt decoding from pad tokens.
        """
        texts = [
            self.processor.apply_chat_template(
                job.messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=job.thinking,
            )
            for job in jobs
        ]
        tokenizer = self.processor.tokenizer
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        finally:
            tokenizer.padding_side = previous_side
        width = int(inputs["input_ids"].shape[1])
        output = self.model.generate(
            **inputs,
            max_new_tokens=max(job.max_tokens for job in jobs),
            do_sample=False,
        )
        results = []
        for index, job in enumerate(jobs):
            completion = output[index, width:]
                                                                                
                                                                      
            ids = completion.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            content = tokenizer.decode(ids, skip_special_tokens=True)
            prompt_tokens = int(inputs["attention_mask"][index].sum())
            results.append((content, prompt_tokens, len(ids)))
        return results


def load_image(reference: str) -> Image.Image:
    """Accept a data URI, a file path, or a file:// URL."""
    if reference.startswith("data:"):
        _, _, payload = reference.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if reference.startswith("file://"):
        reference = reference[len("file://"):]
    return Image.open(reference).convert("RGB")



def normalize(messages):
    """Split an OpenAI message list into chat-template parts and PIL images.

    These have to come out together. ``normalize`` rewrites ``image_url`` parts
    into the ``{"type": "image"}`` placeholder the chat template expects, which
    drops the URL -- so anything that tries to collect images *after*
    normalising finds none and the model silently answers text-only. That bug
    shipped once: the Raw-Fused arm is configured ``mm_mode: image`` and was
    measured for a full pilot without ever receiving a screenshot.
    """
    out, images = [], []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    reference = url.get("url") if isinstance(url, dict) else url
                    if reference:
                        images.append(load_image(str(reference)))
                        parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
            out.append({"role": message.get("role", "user"), "content": parts})
        else:
            out.append({"role": message.get("role", "user"),
                        "content": [{"type": "text", "text": str(content or "")}]})
    return out, images


class Job:
    """One in-flight request, waiting on its own event."""

    __slots__ = ("messages", "images", "max_tokens", "temperature", "seed",
                 "thinking", "has_image", "done", "result", "error")

    def __init__(self, messages, images, max_tokens, temperature, seed, thinking):
        self.messages = messages
        self.images = images
        has_image = bool(images)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.thinking = thinking
        self.has_image = has_image
        self.done = threading.Event()
        self.result = None
        self.error = None


class Pool:
    """A single replica fed by a batching loop.

    Requests arrive on their own threads (one per connection) and are queued
    rather than run inline. The loop drains what is waiting, groups the
    text-only ones into a single decode, and runs image requests alone.
    """

    def __init__(self, replica: Replica, max_batch: int = 8, wait_seconds: float = 0.05):
        self.replica = replica
        self.max_batch = max_batch
        self.wait_seconds = wait_seconds
        self.queue: queue.Queue[Job] = queue.Queue()
        self.served = 0
        self.failed = 0
        self.counter_lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, messages, images, max_tokens, temperature, seed, thinking):
        job = Job(messages, images, max_tokens, temperature, seed, thinking)
        self.queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def _collect(self) -> list[Job]:
        first = self.queue.get()
        batch = [first]
        if first.has_image or first.temperature > 0:
            return batch
                                                                               
        deadline = self.wait_seconds
        while len(batch) < self.max_batch:
            try:
                job = self.queue.get(timeout=deadline)
            except queue.Empty:
                break
            if job.has_image or job.temperature > 0:
                                                                         
                self.queue.put(job)
                break
            batch.append(job)
            deadline = 0.005
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._collect()
            try:
                with self.replica.lock:
                    if len(batch) == 1:
                        job = batch[0]
                        results = [self.replica.generate(
                            job.messages, job.images, job.max_tokens,
                            job.temperature, job.seed, job.thinking)]
                    else:
                        results = self.replica.generate_batch(batch)
                for job, result in zip(batch, results):
                    job.result = result
            except Exception as error:                                    
                for job in batch:
                    job.error = error
            finally:
                for job in batch:
                    job.done.set()


def make_handler(pool: Pool, model_name: str, cap: int = 1536):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):                                         
            pass

        def _send(self, status: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
                                                                             
                                                                                
                                                                                
                                                                            
                                                                         
                                                                           
                                                               
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/models"):
                self._send(200, {"object": "list", "data": [
                    {"id": model_name},
                    {"id": f"{model_name}-think"},
                ]})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": f"no route {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                messages, images = normalize(request.get("messages") or [])
                if not messages:
                    raise ValueError("messages is required")
                max_tokens = int(
                    request.get("max_tokens") or request.get("max_completion_tokens") or 1024
                )
                                                                              
                                                                                 
                                                                              
                                                                                 
                                                                                
                                                                               
                                                                        
                                                                             
                if max_tokens > cap:
                    max_tokens = cap
                temperature = float(request.get("temperature") or 0.0)
                seed = int(request.get("seed") or 0)
                                                                              
                                                                                
                                                                            
                                                                            
                                                                         
                                                                              
                requested = str(request.get("model") or model_name)
                thinking = requested.endswith("-think")
                started = time.time()
                content, prompt_tokens, completion_tokens = pool.submit(
                    messages, images, max_tokens, temperature, seed, thinking
                )
                with pool.counter_lock:
                    pool.served += 1
                    served = pool.served
                print(f"[serve] #{served} {prompt_tokens}->{completion_tokens} tok "
                      f"{time.time() - started:.1f}s", flush=True)
                self._send(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(started),
                    "model": request.get("model") or model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })
            except Exception as error:                                       
                with pool.counter_lock:
                    pool.failed += 1
                traceback.print_exc()
                self._send(500, {"error": {"message": str(error), "type": type(error).__name__}})

    return Handler


def serve_one(model_path: str, device: str, host: str, port: int, name: str,
              cap: int = 1536, batch: int = 8) -> None:
    """One process, one replica, sharing the port with its siblings."""
    print(f"[serve] loading {model_path} on {device}", flush=True)
    pool = Pool(Replica(model_path, device), max_batch=batch)

    class Reusing(ThreadingHTTPServer):
                                                                               
                                                                             
                                                                           
                                                                             
                                                                     
        def server_bind(self):
            import socket
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            super(ThreadingHTTPServer, self).server_bind()

    server = Reusing((host, port), make_handler(pool, name, cap))
    server.daemon_threads = True
    print(f"[serve] {device} ready on http://{host}:{port}/v1", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0",
                        help="comma-separated CUDA indices; repeats give that card "
                             "more than one worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=8,
                        help="text-only requests decoded together per worker")
    parser.add_argument("--max-new-tokens-cap", type=int, default=1536,
                        help="hard ceiling on generated tokens; bounds runaway loops")
    args = parser.parse_args()

    devices = [f"cuda:{index.strip()}" for index in args.gpus.split(",") if index.strip()]
    name = Path(args.model).name
    if len(devices) == 1:
        serve_one(args.model, devices[0], args.host, args.port, name,
                  args.max_new_tokens_cap, args.max_batch)
        return

    import multiprocessing as mp

    context = mp.get_context("spawn")
    children = [
        context.Process(
            target=serve_one,
            args=(args.model, device, args.host, args.port, name,
                  args.max_new_tokens_cap, args.max_batch),
            daemon=False,
        )
        for device in devices
    ]
    for child in children:
        child.start()
    print(f"[serve] {name} on http://{args.host}:{args.port}/v1 "
          f"with {len(children)} worker process(es)", flush=True)
    try:
        for child in children:
            child.join()
    except KeyboardInterrupt:
        for child in children:
            child.terminate()


if __name__ == "__main__":
    main()
