import math
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, List

def apply_info(model, dominant_num, contextual_num):
    model._info = {
        "dominant":dominant_num,
        "contextual":contextual_num,
        "num_to_keep": (dominant_num + contextual_num + 1) * 1280 / 5120
    }
    for module in model.modules():
        print('module^^^^', module)
        # if isinstance(module, CLIPEncoderLayer):
        #     module._info = model._info

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def Qwen2VLAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    # past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    raw_key_states = key_states.clone() # --update-- 
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Fix precision issues in Qwen2-VL float16 inference
    # Replace inf values with zeros in attention weights to prevent NaN propagation
    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, raw_key_states.mean(1)

def Qwen2VLVisionBlock_forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
    attn_output, attn_weights, metric = self.attn(
        self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
    )
    hidden_states = hidden_states + attn_output
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    self.metric = metric
    return hidden_states, attn_weights

def VisionSdpaAttention_forward(
    self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    # print('&&&&&&&&&', k.shape)
    # 计算 metric
    raw_key_states = k.clone()  # 获取键的原始状态
    metric = raw_key_states.mean(-1)
    metric = metric.transpose(0, 1) # torch.Size([3456, 16])
    # 计算点积注意力权重
    attention_logits = torch.matmul(q, k.transpose(-2, -1))  # 点积 Q 和 K
    head_dim = q.size(-1)  # 每个头的维度
    attention_logits = attention_logits / head_dim**0.5  # 缩放

    # 应用注意力掩码
    if attention_mask is not None:
        attention_logits = attention_logits.masked_fill(attention_mask, float('-inf'))  # 设置为负无穷

    # 计算注意力权重
    attn_weights = torch.nn.functional.softmax(attention_logits, dim=-1)

    # 修改点：使用 scaled_dot_product_attention 返回 attn_weights
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    # attn_output = torch.matmul(attn_weights, v)  # 使用注意力权重计算输出
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights, metric

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Doc utilities: Utilities related to documentation
"""

import functools
import re
import types

def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator
    
def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator

PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""

TF_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.

"""

def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]

def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = output_docstring.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = "\n".join(lines[(i + 1) :])
        params_docstring = _convert_output_args_doc(params_docstring)
    else:
        raise ValueError(
            f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. Make sure it has "
            "docstring and contain either `Args` or `Parameters`."
        )

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    result = intro + params_docstring

    # Apply minimum indent if necessary
    if min_indent is not None:
        lines = result.split("\n")
        # Find the indent of the first nonempty line
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # If too small, add indentation to all nonempty lines
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line) for line in lines]
            result = "\n".join(lines)

    return result

def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator
