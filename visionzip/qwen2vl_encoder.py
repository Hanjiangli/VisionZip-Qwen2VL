import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from .utils_qwen2vl import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import QWEN2_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, Qwen2VLPreTrainedModel, Qwen2VLCausalLMOutputWithPast
from transformers.generation import GenerationMixin

class Qwen2VLForConditionalGeneration_VisionZip(Qwen2VLPreTrainedModel, GenerationMixin):
    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                # --update--
                # --------------更新input_ids---------------------
                mask = (input_ids == self.config.image_token_id)
                # num_to_keep = (self.visual.dominant + self.visual.contextual + 1) * 1280 // 5120
                num_to_keep, _ = image_embeds.shape
                indices = torch.nonzero(mask)
                if indices.size(0) > num_to_keep:
                    indices_to_remove = indices[num_to_keep:]
                    # Create a mask for positions to remove
                    remove_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    for index in indices_to_remove:
                        remove_mask[index[0], index[1]] = False
                    # Apply the mask to input_ids and position_ids
                    input_ids = input_ids[remove_mask].reshape(input_ids.shape[0], -1)
                    # Correctly apply the mask for position ids across all heads
                    position_ids = position_ids[remove_mask.unsqueeze(0).expand(position_ids.shape[0], -1,-1)].reshape(position_ids.shape[0],position_ids.shape[1],-1)
                # --------------以上为修改部分---------------------
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # image_mask = image_mask.to(image_embeds.device) # 确保两者在同一device上
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                # 主要是为了让inputs_embeds.device不报错
                inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

# --update--
def adjust_to_multiple_of_4(number):
    """
    调整一个数字,使其能被4整除。如果不能整除,则返回最接近的能被4整除的数。
    Args:
        number: 要调整的数字 (int 或 float).
    Returns:
        调整后的数字 (int).
    """
    if isinstance(number, float):
         number=int(number)
    if number % 4 == 0:
        return number  # 如果能整除，直接返回
    lower_multiple = (number // 4) * 4  # 小于等于number的最近倍数
    upper_multiple = lower_multiple + 4   # 大于number的最近倍数
    if number - lower_multiple < upper_multiple - number:
        return lower_multiple  # 返回更接近的较小倍数
    else:
        return upper_multiple # 返回更接近的较大倍数
    
class Qwen2VisionTransformerPretrainedModel_VisionZip(nn.Module):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Initialize variables to store attention weights and hidden states
        attn_weights_list = []
        hidden_states_list = []  
        self.gradient_checkpointing = False
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_out = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
                # 假设 blk 输出包括 hidden_states 和 attention_weights
                hidden_states, attn_weights = hidden_out

                # Store the attention weights and hidden states
                attn_weights_list.append(attn_weights)
                hidden_states_list.append(hidden_states)

        # --update--
        # Process attention weights and hidden states to get dominant and contextual tokens
        last_attn_weights = attn_weights_list[-2]  # 获取倒数第二层的注意力权重
        last_attn_weights = last_attn_weights.unsqueeze(0)
        last_hidden_states = hidden_states_list[-2]  # 获取倒数第二层的隐藏状态
        last_hidden_states = last_hidden_states.unsqueeze(0)
        # print('last_hidden_states.shape',last_hidden_states.shape)
        metric = self.blocks[-2].metric # 获取metric
        metric = metric.unsqueeze(0)
        # the key parameters, The goal is for the ratio of param self.dominant to param self.contextual to be 5.4:1
        _, total_token_num, dim = last_hidden_states.shape
        retain_token = adjust_to_multiple_of_4(self.retain_token_ratio * total_token_num) # 5120/1280=4，这么设置与qwen2vl的VisionMlp模块参数有关
        self.contextual = int(retain_token // 6.4)
        self.dominant = int(retain_token - self.contextual - 1)
        # 这里的self.dominant和self.contextual是直接按照比例去分配，参考论文table9将比例设置为5.4:1
        # Dominant Visual Tokens
        cls_idx = 0
        # print(last_attn_weights.shape)
        cls_attention = last_attn_weights[:, :, cls_idx, cls_idx + 1:]
        cls_attention_sum = cls_attention.sum(dim=1)
        topk_indices = cls_attention_sum.topk(self.dominant, dim=1).indices + 1
        all_indices = torch.cat([torch.zeros((last_hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)

        mask = torch.ones_like(last_hidden_states[:, :, 0], dtype=torch.bool, device=last_hidden_states.device).scatter_(1, all_indices, False)
        dominant_tokens = last_hidden_states.masked_select(~mask.unsqueeze(-1)).view(last_hidden_states.shape[0], self.dominant + 1, last_hidden_states.shape[2])

        # Filter and normalize metrics (if available)
        metric_filtered = metric[mask].view(last_hidden_states.shape[0], last_hidden_states.shape[1] - (self.dominant + 1), metric.shape[2])
        hidden_states_filtered = last_hidden_states.masked_select(mask.unsqueeze(-1)).view(last_hidden_states.shape[0], last_hidden_states.shape[1] - (self.dominant + 1), last_hidden_states.shape[2])
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

        # Contextual Visual Tokens
        step = max(1, metric_normalized.shape[1] // self.contextual)
        target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:self.contextual]
        target_tokens = metric_normalized[:, target_indices, :]

        tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], self.contextual, dtype=last_hidden_states.dtype, device=metric_normalized.device)
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        target_hidden = last_hidden_states[:, target_indices, :]
        contextual_tokens = target_hidden + aggregated_hidden

        # Merge dominant and contextual tokens
        final_hidden_states = torch.cat([dominant_tokens, contextual_tokens], dim=1)
        return self.merger(final_hidden_states)
        # return final_hidden_states
