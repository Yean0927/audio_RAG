FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN test "$(grep -Fc 'else if (!embeds->user_text.empty()) {' tools/omni/omni.cpp)" -eq 1 \
    && sed -i 's/else if (!embeds->user_text.empty()) {/if (!embeds->user_text.empty() \&\& (!ctx_omni->duplex_mode || (embeds->vision_embed.empty() \&\& embeds->audio_embed.empty()))) {/' tools/omni/omni.cpp \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-omni-server --parallel 4

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
LABEL io.glm-voice-rag.omni-prefill="audio-text-v1"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/bin/ /app/bin/
COPY --from=build /src/tools/omni/assets/default_ref_audio/default_ref_audio.wav \
    /app/tools/omni/assets/default_ref_audio/default_ref_audio.wav
ENV LD_LIBRARY_PATH=/app/bin:/usr/local/cuda/lib64
EXPOSE 8006
ENTRYPOINT ["/app/bin/llama-omni-server"]
CMD ["--host", "0.0.0.0", "--port", "8006", "-m", "/models/MiniCPM-o-4_5-Q4_K_M.gguf", "-ngl", "99", "-c", "8192"]
