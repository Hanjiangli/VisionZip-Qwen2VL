

import time
import os
import torch
from visionzip import visionzip_qwen2vl
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# print("CUDA Device Count:", torch.cuda.device_count())
# print("Current Device:", torch.cuda.current_device())

# 调用 visionzip_qwen2vl 函数进行设置，调整 retain_token_ratio
# 函数无返回值，会对qwen2vl的部分forward函数进行修改
# visionzip_qwen2vl(retain_token_ratio=0.125)

# 加载预训练模型并设置设备
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# 加载处理器（processor），负责文本和图像的预处理
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 设置输入消息，包括图像和文本内容
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": './image/tennis.jpg'},  # 图像路径
            {"type": "text", "text": "请输出这张图片包含的内容"}  # 请求描述图像内容的文本
        ]
    }
]

# 准备推理所需的输入
# 生成聊天模板文本并进行处理
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 处理图像和视频输入数据
image_inputs, video_inputs = process_vision_info(messages)

# 合并处理后的文本、图像和视频输入数据，准备进入模型
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")  # 将数据移动到GPU

# 开始推理，记录开始时间
start_time = time.time()

# 调用模型生成输出
generated_ids = model.generate(**inputs, max_new_tokens=520)

# 去掉生成文本中多余的部分（trim）
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 解码生成的ID为文本
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 打印推理时间和输出文本
print(f'inference & batch_decode time cost is: {time.time() - start_time:.2f}s')
print(output_text)
