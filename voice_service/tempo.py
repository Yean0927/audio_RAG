"""One continuous FFmpeg atempo filter per reply, shared by live and saved audio."""
from __future__ import annotations

import asyncio
from contextlib import suppress


class StreamingTempo:
    def __init__(self, executable, speed, sample_rate, on_audio):
        self.executable = executable
        self.speed = speed
        self.sample_rate = sample_rate
        self.on_audio = on_audio
        self.queue = asyncio.Queue()
        self.chunks = []
        self.tasks = []
        self.process = None

    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            self.executable, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "s16le", "-ar", str(self.sample_rate), "-ac", "1",
            "-probesize", "32", "-analyzeduration", "0", "-blocksize", "4096",
            "-i", "pipe:0", "-af", f"atempo={self.speed}",
            "-f", "s16le", "-ar", str(self.sample_rate), "-ac", "1", "-flush_packets", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        self.tasks = [asyncio.create_task(self.write()), asyncio.create_task(self.read()),
                      asyncio.create_task(self.process.stderr.read())]
        return self

    def feed(self, pcm):
        for task in self.tasks:
            if task.done():
                task.result()
        self.queue.put_nowait(pcm)

    async def write(self):
        while (pcm := await self.queue.get()) is not None:
            self.process.stdin.write(pcm)
            await self.process.stdin.drain()
        self.process.stdin.close()
        with suppress(BrokenPipeError, ConnectionResetError):
            await self.process.stdin.wait_closed()

    async def read(self):
        pending = b""
        while chunk := await self.process.stdout.read(4096):
            pending += chunk
            size = len(pending) // 2 * 2
            if size:
                pcm, pending = pending[:size], pending[size:]
                self.chunks.append(pcm)
                if self.on_audio is not None:
                    self.on_audio(pcm)
        if pending:
            raise RuntimeError("FFmpeg 返回了不完整的 PCM 样本。")

    async def finish(self):
        self.queue.put_nowait(None)
        try:
            _, _, error = await asyncio.gather(*self.tasks)
        except (BrokenPipeError, ConnectionResetError) as error:
            raise RuntimeError("FFmpeg 语速处理提前结束。请检查 ffmpeg 安装。") from error
        if await self.process.wait():
            raise RuntimeError(f"FFmpeg 语速处理失败：{error.decode(errors='replace')[-1000:]}")
        if not self.chunks:
            raise RuntimeError("FFmpeg 语速处理未返回音频。")
        return b"".join(self.chunks)

    async def __aexit__(self, *exc):
        if self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.kill()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.process.wait()
