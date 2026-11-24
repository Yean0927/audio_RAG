# 浏览器语音对话

## 启动语义 VAD 通话

在服务端准备 Smart Turn v3.2 CPU 权重，然后在 `call_config.json` 中设置 `smart_turn_model`。默认路径为 `./smart-turn-v3.2-cpu.onnx`。相对路径从项目根目录解析，也可以填写服务器上的绝对路径。临时切换权重时，传入 `--smart-turn-model` 可覆盖配置。需要支持 ONNX IR 10 的 ONNX Runtime（1.18 或更新版本）。`setup_glm_server.py` 使用的 1.19.2 已满足要求。[模型权重](https://huggingface.co/pipecat-ai/smart-turn-v3/tree/f766f81)

```bash
.venv/bin/python -m pip install 'silero-vad==6.2.0'
curl -fL -o smart-turn-v3.2-cpu.onnx \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81/smart-turn-v3.2-cpu.onnx
bash start_voice_call.sh
```

首页 `http://localhost:7860/` 会进入 `/omni_vad`。点击「开始通话」后可以直接说话，也可以在客服思考或播放回复时插话。浏览器连续上传麦克风音频，由服务端 Silero 检测语音起点和停顿。约 224 毫秒停顿后用 CPU Smart Turn 判停，用户续说会使旧判定失效。静音达到约 1.5 秒时兜底提交。

「对话记录」会在用户录音旁标出提交原因，服务端日志也会记录 `commit_reason` 和 Smart Turn 确认结束时的概率。话轮提交前用 `CallSpeechFilter` 复检，过滤为空就恢复监听或继续播放被暂停的回复，通过复检的录音会裁剪首尾并调整音量。

通话默认用 `call_config.json` 配置的 `resources/customer_faq.md` 知识库，也可以通过 `knowledge_file` 指定自己的 UTF-8 TXT 或 Markdown 文件。同时启动多个模型时，用 `--port` 指定不同的端口，再访问对应端口的 `/omni_vad`。

本地 Omni 在整轮生成和解码完成后发送回复音频。服务器在模型生成步骤检查取消信号，正在执行的音频解码会在完成后处理取消。完整权重的 GPU 推理和真实麦克风插话效果需要在服务器验证。

### 通过 vLLM 运行 GLM 的语义 VAD 通话

首次安装时，在 Linux x86_64 服务器的项目根目录运行 `python3.12 -m tools.install_vllm_cu128`。启动脚本离线查找本地缓存的 GLM-4-Voice-9B 权重，监听 `127.0.0.1:18000`，模型名为 `glm-voice`。

```bash
bash start_vllm.sh
```

也可以用 `bash start_vllm.sh /path/to/model` 指定模型目录，或提前导出 `MODEL`。等 `/v1/models` 返回 `glm-voice` 后，另开终端启动通话服务。

```bash
bash start_vllm_call.sh
```

通话脚本从正在运行的 vLLM 服务读取本地快照路径，在 GPU 0 上运行语音组件，并使用余弦检索。`bash start_vllm_call.sh "$MODEL"` 仍可显式传入路径，变量为空时会自动查询 vLLM。语音服务通过 `/v1/completions` 传入 prompt token ID，再用返回的流式 token ID 合成音频。

GLM 的语音分词器会优先使用主模型旁边或项目下的 `glm-4-voice-tokenizer`、项目下的 `THUDM/glm-4-voice-tokenizer`，以及 Hugging Face 缓存中的 `zai-org` 或 `THUDM` 快照。已有权重放在其他目录时，用 `--tokenizer-path /path/to/glm-4-voice-tokenizer` 指定。组件都已准备好时，可用 `HF_HUB_OFFLINE=1 bash start_vllm_call.sh "$MODEL"` 让 Hugging Face 只读取本地文件。

在服务器本机打开 `http://127.0.0.1:7860/omni_vad`，也可以使用 `/omni_vad/vllm` 确认通话服务启用了 vLLM。若 `7860` 已被其他通话服务占用，用 `bash start_vllm_call.sh --port 7861` 指定空闲端口。从另一台机器访问时，可以使用下方的 HTTPS 入口，或通过 SSH 转发实际监听端口。

### 通过现有 Apache HTTPS 入口访问 vLLM 通话

局域网客户端打开 `https://192.168.10.10/omni_vad`。如果同一 HTTPS 入口的 `/api/health` 返回 `state: ready`，且 `/omni_vad/vllm` 可以打开，就可以使用现有入口。客户端需要信任服务器证书对应的本地根 CA，浏览器才能启用麦克风。

如果 Apache 返回 503，先在服务器上用 `ss -ltnp '( sport = :7860 or sport = :7861 )'` 确认通话进程的监听端口，再查看 Apache 错误日志核对代理目标。Apache 的代理目标应与通话进程的实际监听端口一致。若要将旧的 `7861` 实例迁到 `7860`，先确认 `7860` 空闲，再停止并重启通话进程。vLLM 推理服务继续在 `127.0.0.1:18000` 提供服务。

```bash
# 在服务器上检查 vLLM，并在单独的终端启动通话服务
curl -fsS http://127.0.0.1:18000/v1/models
bash start_vllm_call.sh
```

在服务器的另一终端检查实际监听端口的 `/api/health`，例如 `curl -fsS http://127.0.0.1:7860/api/health`，等待 `state` 为 `ready`。再从客户端访问 HTTPS 入口。

若本机健康检查成功但 Apache 仍返回 503，检查 HTTPS 站点的代理目标和 Apache 错误日志。若网页能加载但通话连不上，检查 Apache 是否代理 `/api/call` 的 WebSocket 升级，且启用了 `ProxyPreserveHost On`，使通话服务收到的 `Host` 与浏览器发出的 `Origin` 一致。

### 通过 Transformers 和 SDPA 运行 GLM

```bash
bash start_sdpa_call.sh
```

该脚本默认在 `cuda:0` 运行 GLM 4bit、语音组件和 SONAR，并按余弦相似度检索 FAQ。打开 `http://127.0.0.1:7860/omni_vad`，由 Smart Turn 语义判停。权重路径从 `call_config.json` 的 `smart_turn_model` 读取。可用 `bash start_sdpa_call.sh "$MODEL"` 指定本地模型快照，也可以提前导出 `MODEL`。

## 切换语音后端

`voice_call_server.py` 和 `voice_web_demo.py` 共用 `--backend glm|omni-local|ming|omni|minicpmo|minicpmo-local|minicpmo-gguf`。启动时选择后端。本地进程加载权重时可用 `--model-path` 指定权重目录。

| 参数 | 默认推理模型 | 文本与语音输入组件 | 语音输出组件 | 输出采样率 |
| --- | --- | --- | --- | --- |
| `--backend glm` | `zai-org/glm-4-voice-9b` | GLM 文本分词器、GLM-4-Voice-Tokenizer 的 Whisper-VQ 编码器 | 独立 GLM-4-Voice-Decoder | 22050 Hz |
| `--backend omni-local` | `Qwen/Qwen2.5-Omni-7B` | Qwen2.5-Omni Processor、Thinker 音频编码器 | 模型内的 Talker 和 Token2Wav | 24000 Hz |
| `--backend ming` | `Ming-3.0-audio-flash` | 蚂蚁百灵音频理解 API | `Ming-3.0-tts-0.5B` API | 24000 Hz |
| `--backend omni` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 硅基流动多模态对话 API | `FunAudioLLM/CosyVoice2-0.5B` API | 24000 Hz |
| `--backend minicpmo` | `MiniCPM-o-4.5-Realtime` | 面壁智能 Realtime Chat API | 同一 Realtime Chat API | 24000 Hz |
| `--backend minicpmo-local` | `openbmb/MiniCPM-o-4_5` | 自部署 vLLM-Omni Chat Completions API | MiniCPM-o Talker 和 Token2Wav | 24000 Hz |
| `--backend minicpmo-gguf` | `openbmb/MiniCPM-o-4_5-gguf` | 自部署 llama-omni-server | MiniCPM-o Talker 和 Token2Wav | 24000 Hz |

各后端共用 SONAR 语音检索、客服配置、会话历史、结束判断、取消请求和结果保存。GLM 逐块生成并播放音频。通话页的 Ming 和托管 Omni 先生成完整文字，再流式播放 TTS 音频。MiniCPM-o 在一轮请求内流式返回文字和语音。本地 Omni 在整轮生成与解码完成后发送音频。浏览器按后端采样率播放。

GLM 默认 NF4 量化，主模型放在 `cuda:0`，语音组件和 SONAR 放在 `cuda:1`。本地 Omni 默认 BF16，主模型及语音组件放在 `--llm-device`，SONAR 默认跟随该设备。`--speech-device` 可单独指定 SONAR 设备。`--replica-gpus 0,1` 为每张卡加载一套本地后端。

### 自部署 MiniCPM-o 4.5 通话

#### 单张 RTX 3090（24 GB）

通话服务直接连接 llama-omni-server 的 `/backend` WebSocket。下面的 Dockerfile 用 CUDA 12.8.1 为 RTX 3090 编译推理服务，并使用官方 Q4_K_M 权重。运行容器需要 Docker 和 NVIDIA Container Toolkit。 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) · [官方 GGUF 权重](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)

从通话项目的根目录执行。同步代码时把 `resources/minicpmo-gguf.Dockerfile` 一起复制到服务器。权重已下载时可以跳过 `hf download`。

```bash
project_dir="$PWD"
gguf_dir="$project_dir/../models/MiniCPM-o-4_5-gguf"
uvx --from huggingface-hub hf download openbmb/MiniCPM-o-4_5-gguf \
  --revision 47eeee5b877167846601d816a192f9a2ee62fc25 --local-dir "$gguf_dir" \
  --include 'MiniCPM-o-4_5-Q4_K_M.gguf' --include 'audio/*.gguf' \
  --include 'vision/*.gguf' --include 'tts/*.gguf' --include 'token2wav-gguf/*.gguf'

engine_dir="$project_dir/../llama.cpp-omni"
mkdir -p "$engine_dir"
curl -fL -o /tmp/llama-omni-source.tar.gz \
  https://codeload.github.com/tc-mb/llama.cpp-omni/tar.gz/350a835d5f8215145aa210a7df0f67e8877b5d6e
tar -xzf /tmp/llama-omni-source.tar.gz --strip-components=1 -C "$engine_dir"
docker build --network host --build-arg http_proxy --build-arg https_proxy \
  -f "$project_dir/resources/minicpmo-gguf.Dockerfile" -t minicpmo-gguf:cu128 "$engine_dir"
docker run -d --name minicpmo-gguf --gpus device=0 \
  -p 127.0.0.1:8006:8006 -v "$gguf_dir:/models:ro" minicpmo-gguf:cu128
```

等模型服务的 `/health` 返回 HTTP 200。通话服务使用现有的 `.venv`、`call_config.json`、Silero、Smart Turn 和 SONAR，默认在 CPU 上做余弦检索。

```bash
curl --noproxy 127.0.0.1 -fsS http://127.0.0.1:8006/health
bash start_minicpmo_call.sh --backend minicpmo-gguf --max-calls 1 --check
bash start_minicpmo_call.sh --backend minicpmo-gguf --max-calls 1
```

打开 `http://127.0.0.1:7865/omni_vad`。模型使用本轮录音和完整的首条 FAQ，上轮语音的上下文通过之前的客服答复文本传给模型。原 HTTPS 入口继续代理到 `7860` 时，可在旧通话进程退出后给启动命令加上 `--port 7860`。

#### 双 GPU，CUDA 12.9

为 vLLM-Omni 建立独立环境，通话服务继续使用项目的 `.venv`。下面在 Linux x86_64 上显式选用 vLLM 0.24.0 的 CUDA 12.9 wheel。两张 GPU 都需要有足够的空闲显存来加载各自的模型组件。 [官方 CUDA 12.9 wheel](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.24.0) · [vLLM-Omni 的 MiniCPM-o 部署文档](https://github.com/vllm-project/vllm-omni/blob/v0.24.0/examples/online_serving/minicpmo/README.md)

```bash
uv venv --python 3.12 .venv-minicpmo
uv pip install --python .venv-minicpmo/bin/python --torch-backend=cu129 \
  'https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  'vllm-omni==0.24.0' stepaudio2-minicpmo
.venv-minicpmo/bin/vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --trust-remote-code --host 127.0.0.1 --port 8099
```

等 `http://127.0.0.1:8099/v1/models` 返回 `openbmb/MiniCPM-o-4_5` 后，另开终端启动通话服务。确保 `call_config.json` 中 `smart_turn_model` 指向现有的 ONNX 权重，`knowledge_file` 指向要用的 FAQ。

```bash
bash start_minicpmo_call.sh --check
bash start_minicpmo_call.sh
```

打开 `http://127.0.0.1:7865/omni_vad`。Silero 和 Smart Turn 判停后，SONAR 默认在 CPU 上对复检后的录音做跨模态检索。启动脚本使用余弦相似度，把排名第一的 FAQ、本轮录音和会话历史交给 MiniCPM-o。通话页显示命中的 FAQ、模型回复并播放 24 kHz 语音。

若推理服务在其他地址，用 `--minicpmo-base-url http://主机:端口/v1` 指定。若 vLLM-Omni 用 `--served-model-name` 暴露了别名，通话侧用 `--model-path` 指定相同名称。推理服务开启 Bearer 鉴权时，在通话进程设置 `MINICPMO_VLLM_API_KEY`。

### 用托管 API 试测 Ming、Qwen3-Omni 和 MiniCPM-o

分别在百灵和硅基流动的控制台创建令牌。选择要测试的后端，在启动进程的环境中设置对应变量 `ANT_LING_API_KEY` 或 `SILICONFLOW_API_KEY`。服务端把本轮录音与命中的 FAQ 片段放在同一条用户消息中发送，然后将返回的文字交给语音合成接口。百灵每次只接收一段音频，Ming 的历史以此前回复文本提供。Qwen3-Omni 会带上配置保留的历史录音和回复。 [百灵音频理解文档](https://developer.ant-ling.com/zh-CN/docs/tutorials/multimodal-understanding/) · [硅基流动模型目录](https://cloud.siliconflow.cn/open/models?types=chat) · [硅基流动多模态文档](https://docs.siliconflow.cn/docs/userguide/capabilities/multimodal-vision)

MiniCPM-o 使用面壁智能国内平台签发的 `MAP_API_KEY`。逐轮 Chat 请求包含本轮录音、命中的 FAQ 和配置保留的历史。服务端使用整轮回复文字参与 ROUGE-L 评分。 [面壁智能国内控制台](https://platform.modelbest.cn/) · [Realtime Chat 文档](https://minicpmo45.modelbest.cn/docs/zh/realtime-api/chat/)

已有项目 Python 环境时，可先用录音页面测试中文回答与播报。打开相应本机端口，录音后点击「发送语音」。此页面默认关闭知识库，便于先验证音频上传、文字回答和播报。

```bash
python voice_web_demo.py --backend ming --port 7861
python voice_web_demo.py --backend omni --port 7862
python voice_web_demo.py --backend minicpmo --port 7865
```

没有项目环境时，可在 Python 3.11 虚拟环境安装录音页面所需的依赖。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy<2' torch==2.6.0 torchaudio==2.6.0 soundfile==0.12.1 \
  gradio==4.44.1 huggingface-hub==0.25.2 httpx==0.27.2 fastapi==0.115.3 pydantic==2.7.4 websockets==12.0
```

在已配置 SONAR 和 Smart Turn 的项目环境中，可通过现有通话页面测试 FAQ 检索、播放和插话。在分别配置令牌的终端执行以下命令，并打开对应端口的 `/omni_vad`。API 后端不加载对话或合成权重，SONAR 默认在 CPU 上运行。首次准备欢迎语等缓存音频会调用语音合成 API。

```bash
bash start_voice_call.sh --backend ming --port 7861
bash start_voice_call.sh --backend omni --port 7862
```

### 用同一批录音对比 FAQ 回答

在各终端配置对应 API Key，使用同一份客服配置和同一种检索排序启动需要评测的服务。

```bash
bash start_voice_call.sh --backend ming --config call_config.json --retrieval-metric cosine --port 7863
bash start_voice_call.sh --backend omni --config call_config.json --retrieval-metric cosine --port 7864
bash start_voice_call.sh --backend minicpmo --config call_config.json --retrieval-metric cosine --port 7865
```

将录音置于 `answer_data/test_audio`，等要评测的服务的 `/api/health` 返回 `state: ready` 后从项目根目录运行。每次对比使用同一批录音。

```bash
mkdir -p answer_data/evaluations
.venv/bin/python -m tools.evaluate_voice_exact_match \
  --url http://127.0.0.1:7863 --compare-url http://127.0.0.1:7864 \
  --json answer_data/evaluations/faq_comparison.json
```

下面以 MiniCPM-o 对比 Ming 为例。改用 `--compare-url http://127.0.0.1:7864` 即可对比 Omni，省略 `--compare-url` 则只测 MiniCPM-o。

```bash
.venv/bin/python -m tools.evaluate_voice_exact_match \
  --url http://127.0.0.1:7865 --compare-url http://127.0.0.1:7863 \
  --json answer_data/evaluations/minicpmo_vs_ming.json
```

报告保留原有的字符级 ROUGE-L，并汇总两边检索出同一条 FAQ 时的配对分数。每条记录包含实际检索的条目和模型回复，可以核对关键数字、办理条件和答案是否完整。

要计算检索命中率与标准答案 ROUGE-L，可追加 `--gold answer_data/faq_labels.json`。标注文件的格式为 `{"安装.wav": "是否包安装"}`，键为实际录音文件名，值为正确 FAQ 的首行标题，省略 `## 问题：` 前缀。使用自定义知识库时，加 `--faq-file` 指向通话服务使用的同一文件。部分录音也可以标注。

### Qwen2.5-Omni 环境与启动

Omni 使用支持 `Qwen2_5OmniForConditionalGeneration` 的 Transformers 4.53.2。项目原有 GLM 环境使用 4.44.1，因此提供独立依赖文件。音频输入通过 Omni Processor 处理，生成接口返回文字和 24 kHz 音频。[Transformers Qwen2.5-Omni 文档](https://huggingface.co/docs/transformers/v4.53.2/en/model_doc/qwen2_5_omni)

以下命令对应现有服务器使用的 PyTorch 2.6 / CUDA 12.4。

```bash
python3.11 -m venv .venv-omni
.venv-omni/bin/python -m pip install --upgrade pip
.venv-omni/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv-omni/bin/python -m pip install fairseq2==0.4.5 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
.venv-omni/bin/python -m pip install -r requirements-omni.txt
.venv-omni/bin/python -m pip check
.venv-omni/bin/python voice_call_server.py --backend omni-local --download-models
```

`--omni-speaker Chelsie|Ethan` 选择本地 Omni 的音色，默认为 Chelsie。`--max-new-tokens` 约束 Thinker 文字生成。手动调试页可运行 `.venv-omni/bin/python voice_web_demo.py --backend omni-local`。

GLM 的独立组件路径由 `--tokenizer-path`、`--flow-path`、`--flow-steps` 控制。本地 Omni 从自身模型加载音频组件，指定这些 GLM 专用参数时会在启动前报错。本地 Omni 当前仅支持 BF16 加载。

### 适配器与验证

`voice_service/backend.py` 管理共享的会话和 RAG，`voice_service/cli.py` 管理后端选择和命令行参数。`voice_service/glm.py` 实现 GLM 的加载与流式生成，`voice_service/audio_models.py` 提供本地 Qwen2.5-Omni，`voice_service/api.py` 提供 Ming、Qwen3-Omni 和 MiniCPM-o 的托管接口。`voice_call_server.py` 和 `voice_web_demo.py` 分别是通话和手动录音的入口。

回归测试使用 CPU 张量和模拟运行时，覆盖后端默认配置、组件隔离、取消语音回复、Omni 文字切片、RAG 记录和 24 kHz 通话协议。

```bash
python -m unittest discover -s tests -v
```

实际权重加载、CUDA 扩展编译、显存占用和语音质量需要在 GPU 服务器验证。

## 一键客服通话（新入口）

在项目目录执行：

```bash
source .venv/bin/activate
python -m pip install -r requirements-web.txt -c requirements-server.txt -c constraints-server.txt
bash start_voice_call.sh
```

如果旧的 `voice_web_demo.py` 还占用 7860 端口，先在其终端按 Ctrl+C 停止。
启动脚本默认在服务进程内启用在线加载。设置 `HF_HUB_OFFLINE=1` 或 `TRANSFORMERS_OFFLINE=1` 时，使用现有的本地模型文件。
缺少 Decoder 时自动下载 `config.yaml`、`flow.pt`、`hift.pt`；主模型和 Tokenizer 在加载时按需下载。
这解决了下载日志中的 `OfflineModeIsEnabled`。网络仍需能够访问 Hugging Face；如果继续遇到网络错误，保留完整终端日志以便诊断。

通过下面的 SSH 转发访问，麦克风使用 Mac 上的浏览器：

```bash
ssh -N -L 7860:127.0.0.1:7860 root@192.168.10.10
```

打开 `http://localhost:7860` 后会进入 `/omni_vad`。点击「开始通话」，接通后直接说话，由 Smart Turn 判断何时说完。客服播放语音时也可以直接插话。
右侧「对话记录」可逐轮回听语音并展开检索结果。点击某轮的「保存到数据飞轮」，会在运行语音服务的项目根目录下生成 `数据飞轮/<记录编号>/`，其中 `record.json` 包含回复文字、检索结果及音频文件名，已有的用户录音和客服语音分别保存为 `question.wav` 和 `reply.wav`。挂断后仍可以保存本次通话中的记录。
话轮提交前，服务端用 Silero VAD 复检人声，裁去首尾杂音并调整音量，再将录音用于检索和回答。可以在右侧「对话记录」里播放实际送入模型的片段。
启用知识库后，模型每轮只参考排名第一的条目。「检索结果」会显示这条结果的排名、分数和完整文本。开启 `--compare-retrieval` 时，页面也会显示另一种检索方式的第一名，并标明哪条用于生成回答。
同一按钮在通话中变为「挂断通话」。点击后立即关闭麦克风和通话连接，停止当前回复并取消在途生成。结束语默认开启。要关闭，在 `call_config.json` 中将 `farewell_enabled` 改为 `false` 并重启语音服务。改回 `true` 并重启即可恢复。
页面关闭或连接中断也会清理本地麦克风；服务端不再发送该通话的回复。

收到问题后，页面显示处理状态，正式回复的音频准备好后直接播放。默认关闭「正在为您查询」语音提示；需要恢复时，可在 `call_config.json` 中将 `query_prompt` 设为 `true` 并重启服务。服务器日志的 `first audio in` 记录服务端从开始处理请求到发送正式回复首段音频的用时。
浏览器采集音频时开启回声消除和降噪。
每个服务实例默认同时接待 5 路通话。名额用完后，新来电会收到忙线提示。用 `--max-calls` 可以调整上限，`/api/health` 的 `max_calls` 和 `available_slots` 分别显示总名额和空闲名额。
多路 GLM 通话可以共用一套模型并同时生成，语音解码按音频块串行处理。本地 Omni 在每个模型副本上逐路推理。使用 `--replica-gpus` 时，每路通话接入当前连接数最少的副本。

首次运行会从 `resources/call_config.example.json` 创建 `call_config.json`；后者用于服务器端配置，不出现在用户页面。
可以设置客服名称、开场白、中文/英文、客服话术和 UTF-8 知识库文件路径。
服务启动时用同一后端和音色设置准备开场白和已启用的雷士结束语，音频保存在 `answer_data/calls/greetings/`（自定义 `--output-dir` 时为该目录下的 `greetings/`）。接通时播放缓存的开场白。启用 `farewell_enabled` 时，浏览器在通话期间后台下载并解码结束语，挂断后沿用通话的音频上下文播放，播放完毕后释放。如果挂断时解码仍在进行，播放会等音频就绪。修改 `greeting` 并重启后会生成新的开场白音频，设为空字符串可关闭开场白。音频准备失败或回复只有文字时，页面显示相应文字。
浏览器会在开场白前播放约 250 毫秒静音，让麦克风接入后的音频输出稳定下来。若开场白仍缺字，可直接试听服务端缓存的 WAV：原文件也缺字时检查合成，原文件完整时检查浏览器播放或插话误触发。
新配置默认从 `resources/customer_faq.md` 加载在线客服 FAQ，并启用 SONAR 跨模态检索。
已有配置需在 `call_config.json` 中将 `knowledge_file` 改为 `resources/customer_faq.md` 才会切换。留空时进行自由客服对话。
修改配置后重启服务。默认仍为主模型 GPU 0 / 语音组件和可选 SONAR GPU 1。

只检查权重文件和客服配置，不加载 GPU 模型：

```bash
python voice_call_server.py --check
```

录音与结果仍保存在 `answer_data/calls/`，不随挂断删除。

可以用启动参数选择 SONAR 的检索方式：

```bash
bash start_voice_call.sh --retrieval-metric euclidean
bash start_voice_call.sh --retrieval-metric cosine
bash start_voice_call.sh --retrieval-metric cosine --compare-retrieval
```

默认使用平方欧氏距离，分数越小越靠前。余弦相似度的分数越大越靠前。
开启 `--compare-retrieval` 后，两种排序共用同一个语音向量和 FAQ 索引，各保留第一条。
同一轮的 `retrieval_comparison.json` 记录两种检索方式各自的第一名、分数和完整问答。`context.txt` 记录按 `--retrieval-metric` 选中并交给模型的条目。两种分数只在各自的排序内比较。

## 手动录音调试页

以下保留原先手动录音页面的说明，便于调试。

`setup_glm_server.py` 安装和检查环境；`voice_web_demo.py` 启动网页。
网页支持麦克风录音、上传音频、文字及语音回答、最近三轮上下文，以及可选的 SONAR 语音检索。
录完后点击发送，再等待完整回答；当前版本不提供边说边播或打断。

## 服务器准备

在服务器项目目录激活 `.venv` 后，先完成环境检查：

```bash
python setup_glm_server.py
python -m pip install -r requirements-web.txt -c requirements-server.txt -c constraints-server.txt
python -m pip check
huggingface-cli download zai-org/glm-4-voice-decoder --local-dir ./glm-4-voice-decoder
```

主模型与语音 Tokenizer 首次加载时自动从 Hugging Face 下载。开启知识库后，SONAR 会另行下载编码器。
也可用 `--model-path`、`--tokenizer-path`、`--flow-path` 指定已经下载的本地目录。

## 启动和连接

服务器：

```bash
python voice_web_demo.py
```

Mac 另开本地终端，建立 SSH 转发，并保持该终端打开：

```bash
ssh -N -L 7860:127.0.0.1:7860 root@192.168.10.10
```

Mac 浏览器打开 http://localhost:7860，允许麦克风权限，录音结束后点击发送。
浏览器的麦克风需要安全上下文；通过 SSH 转发访问 localhost 可满足这一要求。
服务默认只监听服务器的 127.0.0.1，不创建 Gradio 公网分享链接。

默认将主模型以 NF4 量化加载到 GPU 0，把语音 Tokenizer、Decoder 和可选的 SONAR 放到 GPU 1。
默认使用 Transformers 加载 GLM；双卡分配仅作用于此网页入口。

单卡可用 `--speech-device cuda:0`；显存充裕时可用 `--quantization bf16`。
默认每条录音最长 30 秒，回答最多生成 768 个文本与语音 token；可通过命令行参数调整。

### GLM 生成速度对照

在部署用的 NVIDIA GPU 上，每组配置使用相同的六个客服录音。`--call-samples` 读取 `answer_data/calls` 中灯具连接 APP、国外项目采购、加盟、品牌合作、如何采购和以旧换新的 `question.wav`。也可用 `--wav 路径1 路径2` 指定其他录音。

```bash
mkdir -p answer_data/profiles
python -m tools.profile_voice_gpu --call-samples --quantization 4bit --attn-implementation auto --json answer_data/profiles/glm-4bit-auto.json
python -m tools.profile_voice_gpu --call-samples --quantization 4bit --attn-implementation sdpa --json answer_data/profiles/glm-4bit-sdpa.json
python -m tools.profile_voice_gpu --call-samples --quantization bf16 --attn-implementation sdpa --json answer_data/profiles/glm-bf16-sdpa.json
```

先看 `auto` 实际选用的注意力实现。如果已选 SDPA，可跳过第二条。对比 4bit 和 BF16 时，指定同一种注意力实现，BF16 需要足够的显存。如果已安装兼容的 FlashAttention，可以保持 4bit，将 `--attn-implementation` 改为 `flash_attention_2` 单独测试注意力的收益。

每组配置只加载一次模型。脚本先取每条录音的重复测量中位数，再取录音间中位数。首块以 25 个语音 token 齐备为准，未达到门槛的样本会标出，比较时使用两组均成功的录音。`--json` 保存逐样本、逐轮数据，`--trace-dir` 可另录 CUDA trace，其中 `glm_generate` 标记 GLM 的生成范围。选定更快的组合后，将对应参数传给 `bash start_voice_call.sh`。

## 知识库

默认关闭知识库，可直接自由对话。需要检索时，勾选“使用知识库回答”，然后粘贴内容或上传 UTF-8 的 TXT/Markdown。
文件优先于粘贴文本；知识库限制为 1 MB / 20 万字。
FAQ 文件以 `## 问题：` 开始每条问答，随后另起一行写 `答案：`，条目之间空一行。每条不超过 600 字符时，可以在答案前逐行添加其他常见问法。
模型每轮只参考所选检索指标排名第一的结果（Top 1）。SONAR 按问题和常见问法检索 FAQ，同一条 FAQ 取各问法中的最佳分数，命中后将完整问答交给模型。普通文本仍按 600 字符分块，只将排名第一的片段交给模型。
每次请求只选择一种语言的编码器，切换语言时重新载入对应编码器。

录音、回答及参考内容保存在 `answer_data/web/<随机编号>/`。点击“开始新对话”清空页面上下文，磁盘记录仍保留。

## 验证范围

本地可验证依赖解析、网页构建、会话和提示词逻辑。实际权重下载、双 GPU 推理、量化质量、中文语音和 SONAR 检索需要在服务器验证。
