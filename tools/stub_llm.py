"""Offline stand-in for the organisers' proxy (OpenAI chat-completions API).

Answers from a JSON script {substring of the last user message: reply}; logs
every request body so the wire format can be inspected.

    python tools/stub_llm.py --port 8089 --script script.json --log requests.jsonl
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--script", default="")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    script = json.load(open(args.script)) if args.script else {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if args.log:
                with open(args.log, "a") as f:
                    f.write(json.dumps({"path": self.path, "body": body}, ensure_ascii=False) + "\n")
            prompt = "\n".join(str(m.get("content")) for m in body["messages"])
            reply = next((v for k, v in script.items() if k in prompt), "42")
            n_in, n_out = len(prompt) // 3, len(reply) // 3 + 1
            data = json.dumps({
                "id": "stub", "object": "chat.completion", "created": 0, "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
