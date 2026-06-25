"""
run.py — DeepSeek-Coder-V2-Lite + Latent 驱逐算法验证脚本
--------------------------------------------------------------
功能：
  1. 加载模型（自动分配 GPU/CPU）
  2. 测量 KV 缓存实际被压缩了多少（prefill 阶段，按层统计）
  3. 对比开启 / 关闭驱逐后的模型输出，人工判断质量是否下降

用法：
    python run.py
"""

import os
import sys
import glob
import json
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════
# 配置区 — 按需修改
# ═══════════════════════════════════════════════════════════
# 权重自动下载
HF_REPO_ID    = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"  # HuggingFace 仓库 ID
USE_HF_MIRROR = True   # 国内设 True 走 hf-mirror.com 镜像加速
AUTO_DOWNLOAD = True   # 检测到权重缺失时是否自动下载

# 阈值驱逐：信息量低于 threshold 的 latent 才被丢弃，丢弃数量随内容自适应，
# 不强制固定比例。threshold 越高 → 驱逐越激进（保留越少）。典型范围 0.2 ~ 0.4。
#
# 验证阶段策略（重要）：仅在 Prefill 阶段做一次性/分块驱逐；Decode 阶段纯追加，
# 不做逐 token 的评分/切片，避免显存不连续重排拖垮 tokens/s。先把质量指标测出来，
# 在线滑窗驱逐留作后续工程加速。该策略由 prefill_only=True 强制保证。
EVICTION_THRESHOLD = 0.3   # 信息量阈值
EVICTION_WINDOW    = 4     # 邻居冗余检测窗口半径
EVICTION_PREFILL_ONLY = True  # 仅 Prefill 驱逐，Decode 纯追加（验证阶段推荐）
MAX_NEW_TOKENS     = 200   # 生成最多多少个新 token（质量验证时用）
DTYPE              = torch.bfloat16  # 推理精度，A100/H100 用 bfloat16，其他可改 float16

# 质量验证用的测试 prompt（可随意替换）
TEST_PROMPTS = [
    "写一个 Python 函数，使用递归计算斐波那契数列的第 n 项，并加上注释。",
    "用一句话解释什么是快速排序，再给出它的平均时间复杂度。",
    "1 + 1 等于多少？",  # 故意放一个极简问题，检验模型有没有胡说
]

# ═══════════════════════════════════════════════════════════
# 0. 环境自检 + 权重自动下载
# ═══════════════════════════════════════════════════════════
def check_environment():
    """检查 Python / torch / CUDA / 显存 / 依赖，打印一份环境体检报告。"""
    print("\n" + "═" * 60)
    print("  阶段零：环境自检")
    print("═" * 60)

    ok = True

    # Python 版本
    py = sys.version.split()[0]
    print(f"  Python              : {py}")
    if sys.version_info < (3, 8):
        print("    ⚠  建议 Python >= 3.8")
        ok = False

    # torch / CUDA
    print(f"  PyTorch             : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA 可用           : {cuda_ok}")
    if cuda_ok:
        n = torch.cuda.device_count()
        total_mem = 0.0
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            mem_gb = p.total_memory / 1024**3
            total_mem += mem_gb
            print(f"    GPU {i}: {p.name}  ({mem_gb:.1f} GB)")
        print(f"  GPU 总显存          : {total_mem:.1f} GB")
        if total_mem < 40:
            print("    ⚠  总显存 < 40 GB，加载可能 OOM（考虑量化或更大显存的卡）")
    else:
        print("    ⚠  未检测到 GPU，将在 CPU 上运行（极慢，仅供调试）")

    # bf16 支持性
    if cuda_ok and DTYPE == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("    ⚠  当前 GPU 不支持 bfloat16，请将配置区 DTYPE 改为 torch.float16")
        ok = False

    # 关键依赖
    try:
        import transformers
        print(f"  transformers        : {transformers.__version__}")
    except ImportError:
        print("    ⚠  缺少 transformers：pip install transformers==4.39.3")
        ok = False
    try:
        import accelerate  # noqa: F401
        print("  accelerate          : 已安装")
    except ImportError:
        print("    ⚠  缺少 accelerate（device_map=auto 需要）：pip install accelerate")
        ok = False
    try:
        import flash_attn  # noqa: F401  # type: ignore
        print("  flash-attn          : 已安装（启用加速）")
    except ImportError:
        print("  flash-attn          : 未安装（可选，会自动退回 eager，仅速度稍慢）")

    print(f"  体检结果           : {'✓ 通过' if ok else '✗ 有需处理的项（见上方 ⚠）'}")
    return ok


def _expected_shards():
    """从 model.safetensors.index.json 读出应有的权重分片文件名集合。"""
    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return set()
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return set(index.get("weight_map", {}).values())


def weights_present():
    """检查所有权重分片是否已存在于 MODEL_DIR。"""
    expected = _expected_shards()
    if not expected:
        # 没有 index 文件时，退而检查是否有任意 .safetensors
        return len(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))) > 0
    missing = [s for s in expected if not os.path.isfile(os.path.join(MODEL_DIR, s))]
    return len(missing) == 0


def ensure_weights():
    """若权重缺失且 AUTO_DOWNLOAD=True，从 HuggingFace 自动下载到 MODEL_DIR。"""
    if weights_present():
        print("\n[ 权重文件已就绪，跳过下载 ]")
        return

    print("\n[ 未检测到完整的权重分片 ]")
    if not AUTO_DOWNLOAD:
        print("  AUTO_DOWNLOAD=False，请手动下载权重后重试。")
        sys.exit(1)

    # 镜像加速（国内）
    if USE_HF_MIRROR and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("  已启用镜像：HF_ENDPOINT=https://hf-mirror.com")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  缺少 huggingface_hub，正在安装 ...")
        os.system(f"{sys.executable} -m pip install -U huggingface_hub")
        from huggingface_hub import snapshot_download

    print(f"  从 {HF_REPO_ID} 下载权重到 {MODEL_DIR}（约 31 GB，请耐心等待）...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=MODEL_DIR,
        allow_patterns=["*.safetensors"],  # 代码/配置本地已有，只拉权重
        resume_download=True,              # 支持断点续传
    )

    if not weights_present():
        print("  ✗ 下载后仍有分片缺失，请检查网络后重试。")
        sys.exit(1)
    print("  ✓ 权重下载完成")


# ═══════════════════════════════════════════════════════════
# 1. 加载
# ═══════════════════════════════════════════════════════════
def load_model():
    print("[ 加载 tokenizer ... ]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[ 加载模型（device_map=auto，会自动分配多卡/CPU offload）... ]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()

    # 打印设备分布（多卡场景下有用）
    if hasattr(model, "hf_device_map"):
        unique_devices = set(model.hf_device_map.values())
        print(f"[ 模型加载完成，分布于: {unique_devices} ]")
    else:
        print("[ 模型加载完成（单卡）]")

    return tokenizer, model


# ═══════════════════════════════════════════════════════════
# 2. 缓存压缩统计
# ═══════════════════════════════════════════════════════════
def measure_prefill_cache(model, tokenizer, text: str):
    """
    只做一次 prefill forward（不生成），返回各层实际缓存的 token 数。
    这样可以精确测量驱逐算法把缓存压缩了多少。
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(next(model.parameters()).device)
    input_len = input_ids.shape[1]

    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    # 读取各层实际写入了多少 token
    cached_lens = [
        cache.key_cache[i].shape[2]
        for i in range(len(cache.key_cache))
    ]
    return input_len, cached_lens


def print_cache_stats(label: str, input_len: int, cached_lens: list):
    avg   = sum(cached_lens) / len(cached_lens)
    ratio = avg / input_len if input_len > 0 else 1.0
    saved = 1.0 - ratio

    sep = "─" * 58
    print(f"\n┌{sep}┐")
    print(f"│  {label:<54}│")
    print(f"├{sep}┤")
    print(f"│  输入 tokens（prefill 长度）  : {input_len:<28}│")
    print(f"│  各层平均缓存 tokens          : {avg:<28.1f}│")
    print(f"│  缓存保留率                   : {ratio:<28.1%}│")
    print(f"│  节省显存比例                 : {saved:<28.1%}│")
    print(f"│  层数 / min / max             : {len(cached_lens)} / {min(cached_lens)} / {max(cached_lens):<18}│")
    print(f"└{sep}┘")


def compare_compression(model, tokenizer, prompt: str):
    """对同一 prompt 测量 无驱逐 / 两档阈值 的缓存大小（驱逐量由内容自适应）。"""
    print(f'\n测量 prompt（前 80 字）："{prompt[:80]}..."')

    model.configure_latent_eviction(enabled=False)
    input_len, lens_base = measure_prefill_cache(model, tokenizer, prompt)
    print_cache_stats("无驱逐（基准）", input_len, lens_base)

    model.configure_latent_eviction(enabled=True, threshold=0.2, window=EVICTION_WINDOW)
    _, lens_lo = measure_prefill_cache(model, tokenizer, prompt)
    print_cache_stats("保守驱逐（threshold=0.2）", input_len, lens_lo)

    model.configure_latent_eviction(enabled=True, threshold=0.3, window=EVICTION_WINDOW)
    _, lens_hi = measure_prefill_cache(model, tokenizer, prompt)
    print_cache_stats("标准驱逐（threshold=0.3）", input_len, lens_hi)

    model.configure_latent_eviction(enabled=False)


# ═══════════════════════════════════════════════════════════
# 3. 生成文本
# ═══════════════════════════════════════════════════════════
def build_chat_prompt(tokenizer, user_text: str) -> str:
    """使用模型自带的 chat_template 格式化 prompt。"""
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 如果 tokenizer 没有 chat_template，退回到手动拼接
        bos = tokenizer.bos_token or ""
        return f"{bos}User: {user_text}\nAssistant:"


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    formatted = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs.input_ids.to(next(model.parameters()).device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # 贪婪解码，保证对比可重复
            temperature=1.0,
            use_cache=True,
        )

    generated_ids = out[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════
# 4. 质量对比
# ═══════════════════════════════════════════════════════════
def quality_check(model, tokenizer):
    print("\n" + "═" * 60)
    print("  质量验证：关闭驱逐  vs  开启驱逐")
    print("═" * 60)

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n【Prompt {idx}/{len(TEST_PROMPTS)}】")
        print(f"  {prompt}")

        # 无驱逐
        model.configure_latent_eviction(enabled=False)
        text_base = generate_text(model, tokenizer, prompt)

        # 开启驱逐
        model.configure_latent_eviction(
            enabled=True, threshold=EVICTION_THRESHOLD, window=EVICTION_WINDOW,
            prefill_only=EVICTION_PREFILL_ONLY,
        )
        text_evict = generate_text(model, tokenizer, prompt)

        # 展示
        wrap = lambda s: "\n    ".join(textwrap.wrap(s, width=70))
        print(f"\n  ▸ [无驱逐]")
        print(f"    {wrap(text_base)}")
        print(f"\n  ▸ [驱逐 threshold={EVICTION_THRESHOLD}, window={EVICTION_WINDOW}]")
        print(f"    {wrap(text_evict)}")

        # 简单字符串相似度：相同字符比例（粗略指标）
        match = sum(a == b for a, b in zip(text_base, text_evict))
        sim = match / max(len(text_base), 1)
        print(f"\n  → 输出字符相似度（粗略）: {sim:.1%}")
        print("  " + "─" * 56)

    model.configure_latent_eviction(enabled=False)


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════
COMPRESSION_PROMPT = (
    "以下是一段关于 Transformer 架构的技术介绍：\n"
    "Transformer 模型由 Vaswani 等人于 2017 年提出，采用自注意力机制替代了传统的 RNN 结构。"
    "它通过多头注意力（Multi-Head Attention）和前馈网络（Feed-Forward Network）堆叠构成，"
    "在自然语言处理、代码生成、图像识别等多个领域取得了巨大成功。\n\n"
    "请根据以上内容，回答：Transformer 最核心的创新点是什么？"
)


if __name__ == "__main__":
    # ── 阶段零：环境自检 + 权重就绪 ──
    check_environment()
    ensure_weights()

    # ── 加载 ──
    tokenizer, model = load_model()

    # ── 阶段一：压缩量统计 ──
    print("\n" + "═" * 60)
    print("  阶段一：KV 缓存压缩量统计（prefill 阶段）")
    print("═" * 60)
    compare_compression(model, tokenizer, COMPRESSION_PROMPT)

    # ── 阶段二：质量对比 ──
    print("\n" + "═" * 60)
    print("  阶段二：输出质量对比")
    print("═" * 60)
    quality_check(model, tokenizer)

    print("\n✅ 全部完成")
