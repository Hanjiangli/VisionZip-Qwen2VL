from .utils_qwen2vl import VisionSdpaAttention_forward, Qwen2VLVisionBlock_forward
from .qwen2vl_encoder import Qwen2VisionTransformerPretrainedModel_VisionZip, Qwen2VLForConditionalGeneration_VisionZip
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock, VisionSdpaAttention, Qwen2VLForConditionalGeneration, Qwen2VisionTransformerPretrainedModel

def visionzip_qwen2vl(retain_token_ratio):
    """
    该函数通过修改模型的 forward 方法来适配 VisionZip,
    其中传入的 retain_token_ratio 用于调整模型的 token 保留比例。
    """
    # 修改 VisionBlock 和 Attention 的 forward 方法
    Qwen2VLVisionBlock.forward = Qwen2VLVisionBlock_forward
    VisionSdpaAttention.forward = VisionSdpaAttention_forward

    # 定义一个内部函数来为 forward 方法传递 retain_token_ratio 参数
    def create_forward_with_params(forward_func, retain_token_ratio):
        """
        为给定的 forward 函数包装一个闭包, 使其支持 retain_token_ratio 参数。
        """
        original_forward = forward_func

        def modified_forward(self, *args, **kwargs):
            # 在每次调用 forward 时, 传入 retain_token_ratio 参数
            self.retain_token_ratio = retain_token_ratio
            return original_forward(self, *args, **kwargs)

        return modified_forward

    # 修改 Qwen2VisionTransformerPretrainedModel 和 Qwen2VLForConditionalGeneration 的 forward 方法
    Qwen2VisionTransformerPretrainedModel.forward = create_forward_with_params(
        Qwen2VisionTransformerPretrainedModel_VisionZip.forward, retain_token_ratio
    )
    Qwen2VLForConditionalGeneration.forward = Qwen2VLForConditionalGeneration_VisionZip.forward
