# 浏览器语音对话

## 各模型一键启动全双工通话

完成对应环境安装后，在项目目录选择一个脚本启动。

```bash
bash start_glm_call.sh
bash start_kimi_call.sh
bash start_omni_call.sh
```

三个脚本均启动 `voice_call_server.py`。浏览器打开 `http://localhost:7860/duplex`，点击「开始通话」后直接说话，客服思考或播放回复时也可以直接插话。检索完成后页面显示「本轮检索结果」，展开可查看实际用于回答的条目、排名和分数。首页 `http://localhost:7860/` 也会自动进入全双工页面。

默认使用 `call_config.json` 中的知识库，目前已配置为 `resources/customer_faq.md`。通过 `knowledge_file` 可以指定自己的 UTF-8 TXT/Markdown 文件，启用检索时需填写有效文件路径。三个脚本默认使用同一份客服配置和知识库。

GLM、Kimi 默认使用 `.venv/bin/python`，Omni 默认使用 `.venv-omni/bin/python`。可以通过 `VOICE_PYTHON` 指定其他已安装对应依赖的 Python 环境。脚本支持追加服务参数。

```bash
bash start_kimi_call.sh --model-path /models/Kimi-Audio-7B-Instruct
bash start_glm_call.sh --speech-device cuda:0
bash start_omni_call.sh --port 7861 --retrieval-metric cosine
VOICE_PYTHON=/opt/venvs/kimi/bin/python bash start_kimi_call.sh
```

默认端口为 7860，同时启动多个模型时为各服务指定不同的 `--port`，并访问对应端口下的 `/duplex`。`--compare-retrieval` 可额外展示另一种检索方式的对照排序。

Kimi/Omni 当前在整轮生成和解码完成后发送回复音频。服务器在模型生成步骤检查取消信号，正在执行的音频解码会在完成后处理取消。完整权重的 GPU 推理和真实麦克风插话效果需要在服务器验证。

### 对比流式 Silero + Smart Turn

在服务端下载 Smart Turn v3.2 CPU 权重，然后用任一通话启动脚本传入本地路径。需要支持 ONNX IR 10 的 ONNX Runtime（1.18 或更新版本）。`setup_glm_server.py` 使用的 1.19.2 已满足要求。[模型权重](https://huggingface.co/pipecat-ai/smart-turn-v3/tree/f766f81)

```bash
.venv/bin/python -m pip install 'silero-vad==6.2.0'
curl -fL -o smart-turn-v3.2-cpu.onnx \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81/smart-turn-v3.2-cpu.onnx
bash start_glm_call.sh --smart-turn-model ./smart-turn-v3.2-cpu.onnx
```

- `http://localhost:7860/duplex` 使用原有的浏览器起录与大模型判停。
- `http://localhost:7860/duplex?vad=semantic` 连续上传麦克风音频，由服务端 Silero 检测语音起点和停顿。约 224 毫秒停顿后用 CPU Smart Turn 判停，用户续说会使旧判定失效。静音达到约 1.5 秒时兜底提交。

挂断后切换页面即可用同一后端和知识库对比。语义 VAD 页的「查看本轮录音」会标出提交原因，服务端日志也会记录 `commit_reason` 和 Smart Turn 确认结束时的概率。语义 VAD 的录音保留浏览器上传的音量，由流式 Silero 裁剪首尾。

## 切换 GLM、Kimi-Audio 和 Omni

`voice_call_server.py` 和 `voice_web_demo.py` 共用 `--backend glm|kimi|omni`。选择后端后重启服务即可切换，`--model-path` 默认随后端选择，也接受完整的本地模型目录。这里的 Omni 使用项目已有的 Qwen2.5-Omni。

| 参数 | 默认推理模型 | 文本与语音输入组件 | 语音输出组件 | 输出采样率 |
| --- | --- | --- | --- | --- |
| `--backend glm` | `zai-org/glm-4-voice-9b` | GLM 文本分词器、GLM-4-Voice-Tokenizer 的 Whisper-VQ 编码器 | 独立 GLM-4-Voice-Decoder | 22050 Hz |
| `--backend kimi` | `moonshotai/Kimi-Audio-7B-Instruct` | Kimi 文本分词器、官方运行时加载的 GLM 语音分词器、模型目录内的 Whisper-large-v3 连续特征编码器 | 模型目录内的 Flow Matching Detokenizer 和 BigVGAN | 24000 Hz |
| `--backend omni` | `Qwen/Qwen2.5-Omni-7B` | Qwen2.5-Omni Processor、Thinker 音频编码器 | 模型内的 Talker 和 Token2Wav | 24000 Hz |

Kimi 按官方消息格式组合文本和音频，输入同时包含离散语音 token 与 Whisper 连续特征。完整模型目录包含 `whisper-large-v3/`、`audio_detokenizer/`、`vocoder/` 和主模型文件。官方运行时还会加载 `THUDM/glm-4-voice-tokenizer`。组件结构依据官方实现接入。[Kimi-Audio 官方代码](https://github.com/MoonshotAI/Kimi-Audio/tree/349251e1d8f4f98d58fda59246381faecd7392e0)

三个后端共用 SONAR 语音检索、客服配置、会话历史、结束判断、取消请求和结果保存。多轮记录使用音频路径和回复文字，Kimi/Omni 在各自的接口中重新构造消息。历史过长时按完整轮次移除最早记录。GLM 继续逐块生成并播放音频，Kimi/Omni 当前在整轮生成与解码完成后发送音频。浏览器按后端采样率播放。

GLM 默认 NF4 量化，主模型放在 `cuda:0`，语音组件和 SONAR 放在 `cuda:1`。Kimi/Omni 默认 BF16，主模型及其语音组件放在 `--llm-device`，SONAR 默认跟随该设备，可以通过 `--speech-device` 单独指定 SONAR 的设备。`--replica-gpus 0,1` 为每张卡加载完整的一套后端。

### Kimi-Audio 环境与启动

在已经按下文安装好的 Linux CUDA 服务器环境中执行。Kimi 的 Whisper 编码器依赖 FlashAttention，首次加载声码器还会编译 CUDA 扩展，需要匹配的 CUDA 编译工具链。

```bash
source .venv/bin/activate
python -m pip install setuptools==80.9.0 wheel ninja packaging
python -m pip install --no-build-isolation -r requirements-kimi.txt -c constraints-server.txt
python -m pip check
bash start_voice_call.sh --backend kimi
```

适配器使用已固定版本的官方 Kimi 推理循环，让 `--max-new-tokens` 同时约束语音回答。上下文预算读取模型配置的 `max_position_embeddings`，将参考文字、历史音频和本轮生成一并计入，接近上限时先移除旧轮次，再缩短本轮生成上限。结束判断使用文本模式。取消请求会在主模型每步前后检查，解码阶段会在解码完成后检查。

可以使用本地目录，或在手动调试页运行。

```bash
bash start_voice_call.sh --backend kimi --model-path /models/Kimi-Audio-7B-Instruct
python voice_web_demo.py --backend kimi
python voice_call_server.py --backend kimi --check
```

Kimi/Omni 的 `--check` 验证参数和客服配置，模型与组件在正式启动时加载。首次在线启动按需下载权重。

### Qwen2.5-Omni 环境与启动

Omni 使用支持 `Qwen2_5OmniForConditionalGeneration` 的 Transformers 4.53.2。项目原有 GLM/Kimi 环境使用 4.44.1，因此提供独立依赖文件。音频输入通过 Omni Processor 处理，生成接口返回文字和 24 kHz 音频。[Transformers Qwen2.5-Omni 文档](https://huggingface.co/docs/transformers/v4.53.2/en/model_doc/qwen2_5_omni)

以下命令对应现有服务器使用的 PyTorch 2.6 / CUDA 12.4。

```bash
python3.11 -m venv .venv-omni
.venv-omni/bin/python -m pip install --upgrade pip
.venv-omni/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv-omni/bin/python -m pip install fairseq2==0.4.5 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
.venv-omni/bin/python -m pip install -r requirements-omni.txt
.venv-omni/bin/python -m pip check
VOICE_PYTHON=.venv-omni/bin/python bash start_voice_call.sh --backend omni
```

`--omni-speaker Chelsie|Ethan` 选择音色，默认为 Chelsie。Omni 的 `--max-new-tokens` 约束 Thinker 文字生成。手动调试页可运行 `.venv-omni/bin/python voice_web_demo.py --backend omni`。

GLM 的独立组件路径由 `--tokenizer-path`、`--flow-path`、`--flow-steps` 控制。Kimi/Omni 从自身模型加载音频组件，指定这些 GLM 专用参数时会在启动前报错。Kimi/Omni 当前支持 BF16 加载。

### 适配器与验证

`voice_backend.py` 定义共用的 `VoiceBackendBase`、`VoiceReply` 和 RAG 流程。`voice_web_demo.py` 保留 GLM 的专属加载与流式生成，并通过 `create_backend()` 选择实现。`voice_audio_models.py` 提供 Kimi-Audio 和 Qwen2.5-Omni 适配器。扩展其他模型时实现对应的加载、推理和结束判断接口，再加入后端选择即可。

回归测试使用 CPU 张量和模拟运行时，覆盖后端默认配置、组件隔离、Kimi 双流输入与解码、生成预算、取消后的重试、Omni 文字切片、RAG 记录和 24 kHz 通话协议。

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
启动脚本会仅在服务进程内设置 `HF_HUB_OFFLINE=0`、`TRANSFORMERS_OFFLINE=0`、`HF_DATASETS_OFFLINE=0`。
缺少 Decoder 时自动下载 `config.yaml`、`flow.pt`、`hift.pt`；主模型和 Tokenizer 在加载时按需下载。
这解决了下载日志中的 `OfflineModeIsEnabled`。网络仍需能够访问 Hugging Face；如果继续遇到网络错误，保留完整终端日志以便诊断。

通过下面的 SSH 转发访问，麦克风使用 Mac 上的浏览器：

```bash
ssh -N -L 7860:127.0.0.1:7860 root@192.168.10.10
```

打开 http://localhost:7860 后会进入 `/duplex` 全双工通话。点击「开始通话」，接通后直接说话，由模型判断何时说完。客服播放语音时也可以直接插话。
原有按钮通话保留在 `/manual`，也可以点击页脚的「按钮通话」进入。该模式点「开始说话」录音，说完点「说完了，发送」，服务端合并录音并开始推理。客服思考或播放时也可以点「开始说话」，打断这一轮并立即录音。
全双工通话先在服务端用 Silero VAD 找到人声，裁去首尾杂音，再调整人声音量。结束判断、知识库检索和回答使用处理后的录音。展开「查看本轮录音」可以播放实际送入模型的片段，查看这一轮的语音 token，并按需解码试听。
启用知识库后，模型每轮只参考排名第一的条目。「本轮检索结果」会显示这条结果的排名、分数和完整文本。开启 `--compare-retrieval` 时，页面也会显示另一种检索方式的第一名，并标明哪条用于生成回答。
同一按钮在通话中变为「挂断通话」。点击后立即关闭麦克风和通话连接，停止当前回复并取消在途生成，随后在网页中播放雷士结束语。
页面关闭或连接中断也会清理本地麦克风；服务端不再发送该通话的回复。

收到问题后，页面显示处理状态，正式回复的音频准备好后直接播放。默认关闭「正在为您查询」语音提示；需要恢复时，可在 `call_config.json` 中将 `query_prompt` 设为 `true` 并重启服务。服务器日志的 `first audio in` 记录服务端从开始处理请求到发送正式回复首段音频的用时。
语音检测使用浏览器本地的自适应能量阈值，开启回声消除和降噪。客服播放音频时提高起录门槛，并要求声音持续达到约 240 毫秒，减少外放回声触发下一轮回答的概率。
一台服务实例同时接待一路通话，其他页面会显示客服正在通话。

首次运行会从 `call_config.example.json` 创建 `call_config.json`；后者用于服务器端配置，不出现在用户页面。
可以设置客服名称、开场白、中文/英文、客服话术和 UTF-8 知识库文件路径。
服务启动时用同一后端和音色设置合成开场白与雷士结束语，音频保存在 `answer_data/calls/greetings/`（自定义 `--output-dir` 时为该目录下的 `greetings/`）。接通时播放缓存的开场白，挂断后播放缓存的结束语。修改 `greeting` 并重启后会生成新的开场白音频，设为空字符串可关闭开场白。音频准备失败或回复只有文字时，页面显示相应文字。
新配置默认从 `resources/customer_faq.md` 加载在线客服 FAQ，并启用 SONAR 跨模态检索。
已有配置需在 `call_config.json` 中将 `knowledge_file` 改为 `resources/customer_faq.md` 才会切换。留空时进行自由客服对话。
将 `auto_listen` 设为 `true` 可在客服回复间隙启用自动收音。浏览器检测到说话后会先补发本地保留的起始音频，检测到说完后自动提交本轮。回复期间仍由「开始说话」按钮触发打断。
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

### 全双工入口

同一服务的 `http://localhost:7860/duplex` 提供全双工通话，首页 `/` 自动跳转到此入口；`/manual` 提供原有的按钮通话。
接通后直接说话，客服思考或播放回复时也可以直接插话。浏览器检测到说话起点后持续上传音频，按录音长度发起模型判断。模型认为还没说完时继续收音。模型确认说完后，服务端用本轮收到的完整音频生成回复。
全双工沿用服务的知识库和 `--retrieval-metric` 设置。模型准备好本轮的参考内容后，页面即显示可展开的「本轮检索结果」，其中列出实际使用的 Top 1。插话打断回复后，已经显示的检索结果可查看到下一轮提交。

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
python profile_voice_gpu.py --call-samples --quantization 4bit --attn-implementation auto --json glm-4bit-auto.json
python profile_voice_gpu.py --call-samples --quantization 4bit --attn-implementation sdpa --json glm-4bit-sdpa.json
python profile_voice_gpu.py --call-samples --quantization bf16 --attn-implementation sdpa --json glm-bf16-sdpa.json
```

先看 `auto` 实际选用的注意力实现。如果已选 SDPA，可跳过第二条。对比 4bit 和 BF16 时，指定同一种注意力实现，BF16 需要足够的显存。如果已安装兼容的 FlashAttention，可以保持 4bit，将 `--attn-implementation` 改为 `flash_attention_2` 单独测试注意力的收益。

每组配置只加载一次模型。脚本先取每条录音的重复测量中位数，再取录音间中位数。首块以 25 个语音 token 齐备为准，未达到门槛的样本会标出，比较时使用两组均成功的录音。`--json` 保存逐样本、逐轮数据，`--trace-dir` 可另录 CUDA trace，其中 `glm_generate` 标记 GLM 的生成范围。选定更快的组合后，将对应参数传给 `bash start_voice_call.sh`。

## 知识库

默认关闭知识库，可直接自由对话。需要检索时，勾选“使用知识库回答”，然后粘贴内容或上传 UTF-8 的 TXT/Markdown。
文件优先于粘贴文本；知识库限制为 1 MB / 20 万字。
FAQ 文件以 `## 问题：` 开始每条问答，随后另起一行写 `答案：`，条目之间空一行。每条不超过 600 字符时，可以在答案前逐行添加其他常见问法。
SONAR 按问题和常见问法检索，命中后把完整问答交给模型。多个问法命中同一条 FAQ 时只返回一次。其他文本仍按 600 字符分块，选择最近的四个片段。
每次请求只选择一种语言的编码器，切换语言时重新载入对应编码器。

录音、回答及参考内容保存在 `answer_data/web/<随机编号>/`。点击“开始新对话”清空页面上下文，磁盘记录仍保留。

## 验证范围

本地可验证依赖解析、网页构建、会话和提示词逻辑。实际权重下载、双 GPU 推理、量化质量、中文语音和 SONAR 检索需要在服务器验证。
