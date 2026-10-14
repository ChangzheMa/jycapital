"""通过本地 Codex daemon 的 WebSocket RPC 发送当前任务通知。"""

import base64
import hashlib
import json
import os
import queue
import struct
import subprocess
import threading
import time


class RpcProxy:
    def __init__(self, binary, thread_id, on_notification=None):
        self.thread_id = thread_id
        self.on_notification = on_notification
        self.request_id = 0
        self.write_lock = threading.Lock()
        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            [str(binary), "app-server", "proxy"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        key = base64.b64encode(os.urandom(16)).decode()
        header = ("GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                  "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                  f"Sec-WebSocket-Key: {key}\r\n\r\n").encode()
        self.process.stdin.write(header)
        self.process.stdin.flush()
        self.key = key
        threading.Thread(target=self._read, daemon=True).start()
        connected = self.messages.get(timeout=10)
        if connected != {"connected": True}:
            raise RuntimeError(connected)
        self.call("initialize", {
            "clientInfo": {"name": "jycapital_run_notifications", "version": "1.0.0"},
            "capabilities": {"experimentalApi": True},
        })
        self.notify("initialized", {})

    def _exact(self, length):
        data = bytearray()
        while len(data) < length:
            part = self.process.stdout.read(length - len(data))
            if not part:
                raise EOFError("proxy closed")
            data.extend(part)
        return bytes(data)

    def _read(self):
        try:
            header = b""
            while not header.endswith(b"\r\n\r\n") and len(header) < 8192:
                header += self._exact(1)
            expected = base64.b64encode(hashlib.sha1(
                (self.key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest())
            if not header.startswith(b"HTTP/1.1 101 ") or expected.lower() not in header.lower():
                raise RuntimeError("WebSocket upgrade failed")
            self.messages.put({"connected": True})
            fragments = bytearray()
            while True:
                first, second = self._exact(2)
                opcode = first & 15
                length = second & 127
                if length == 126:
                    length = struct.unpack("!H", self._exact(2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", self._exact(8))[0]
                if length > 32 * 1024 * 1024:
                    raise RuntimeError("unexpectedly large RPC frame")
                mask = self._exact(4) if second & 128 else None
                payload = self._exact(length)
                if mask:
                    payload = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
                if opcode == 8:
                    raise EOFError("WebSocket closed")
                if opcode == 9:
                    self._send_frame(payload, opcode=10)
                    continue
                if opcode == 10:
                    continue
                if opcode not in (0, 1):
                    continue
                fragments.extend(payload)
                if first & 128:
                    message = json.loads(fragments)
                    fragments.clear()
                    if "id" not in message and self.on_notification:
                        self.on_notification(message)
                    self.messages.put(message)
        except Exception as error:
            self.messages.put({"proxy_error": type(error).__name__, "message": str(error)})

    def _send_frame(self, payload, opcode=1):
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes([128 | opcode, 128 | length])
        elif length < 65536:
            header = bytes([128 | opcode, 128 | 126]) + struct.pack("!H", length)
        else:
            header = bytes([128 | opcode, 128 | 127]) + struct.pack("!Q", length)
        masked = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
        with self.write_lock:
            self.process.stdin.write(header + mask + masked)
            self.process.stdin.flush()

    def notify(self, method, params):
        self._send_frame(json.dumps({"method": method, "params": params}).encode())

    def call(self, method, params, timeout=20):
        self.request_id += 1
        current_id = self.request_id
        self._send_frame(json.dumps({"id": current_id, "method": method, "params": params}).encode())
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(method)
            message = self.messages.get(timeout=max(0.01, deadline - time.monotonic()))
            if "proxy_error" in message:
                raise RuntimeError(message)
            if message.get("id") == current_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result")

    def thread(self):
        return self.call("thread/read", {"threadId": self.thread_id, "includeTurns": False})["thread"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
