import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vlm_manager import VLMManager
from models.radar_branch import get_model as get_radar_model

class LearnableTimeSeriesToImage(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)

    def forward(self, x_enc):
        B, T, C, H, W = x_enc.shape
        x_enc = x_enc.reshape(B * H * W, T, C)
        time_steps = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B * H * W, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, C, 1)

        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < T:
            pad = torch.zeros(B * H * W, T - x_fft_mag.shape[1], C, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)
        x_enc = x_enc.unsqueeze(-1)
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)

        x_enc = x_enc.permute(0, 2, 3, 1)
        x_enc = x_enc.reshape(B * H * W, C * 4, T) 
        x_enc = self.conv1d(x_enc)
        x_enc = x_enc.reshape(B, H, W, self.hidden_dim, T)

        x_enc = x_enc.permute(0, 4, 3, 1, 2)
        x_enc = x_enc.reshape(B * T, self.hidden_dim, H, W)
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(self.conv2d_2(x_enc))

        x_enc = x_enc.reshape(B * T, self.output_channels, H, W)

        return x_enc

class PreMM(nn.Module):
    def __init__(self, config, device):
        super(PreMM, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config, device)
        self.device = device
        self.criterion = nn.MSELoss()

        self._build_model(config)

    def _build_model(self, config):

        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3,
            hidden_dim=48,
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.img_size,
            periodicity=config.periodicity
        )

        self.radar_model = get_radar_model(
            img_channels=config.img_channel,
            T_in=config.frames_in,
            T_out=config.frames_out,
            input_shape=(
                config.img_size,
                config.img_size
            ),
            scale=config.radar_scale,
            patch_size=4,
            backbone_stride=2,
            window_size=4,
            spec_num=config.spec_num,
            pha_weight=config.pha_weight,
            amp_weight=config.amp_weight,
            use_checkpoint=config.radar_checkpoint,
            dropout=config.dropout
        )

        self.mm_residual_scale = nn.Parameter(torch.tensor(0.1))

        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.radar_projection = nn.Conv2d(
            in_channels=self.radar_model.hidden_dim,
            out_channels=config.d_model,
            kernel_size=1
        )
        self.pred_time_embedding = nn.Parameter(torch.randn(1, config.frames_out, config.d_model) * 0.02)

        self.vision_projection = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        self.multimodal_trans = nn.Linear(config.frames_in, config.frames_out)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(config.d_model)

        self.multimodal_decoder = nn.Sequential(
            nn.Conv2d(config.d_model, config.fusion_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(config.fusion_dim, config.fusion_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(config.fusion_dim, config.fusion_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.fusion_dim, config.img_channel, kernel_size=3, padding=1)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(config.img_channel * 2, config.fusion_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.fusion_dim, 2, kernel_size=1)
        )

        self.fusion_layer = nn.Conv2d(config.img_channel * 2, config.img_channel, kernel_size=3, padding=1)


    def _normalize_input(self, x):
        means = x.mean(dim=[1, 3, 4], keepdim=True).detach()
        x = x - means
        var = torch.var(x, dim=[1, 3, 4], keepdim=True, unbiased=False)
        stdev = torch.sqrt(var + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        B, OT, C, H, W = y.shape
        y = y * (stdev.repeat(1, OT, 1, H, W))
        y = y + (means.repeat(1, OT, 1, H, W))
        return y

    @staticmethod
    def _normalize_images(images):
        B, C, H, W = images.shape
        min_vals = images.reshape(B, -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(B, -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        images = (images - min_vals) / scale
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        images = images.reshape(B, C, H, W)

        return images


    def vision_augmented_learner(self, x_enc):
        images = self.learnable_image_module(x_enc)
        images = self._normalize_images(images)
        return images

    def text_augmented_learner(self, x_enc, description, frames_out, frames_in, top_k=5):
        B, T, C, H, W = x_enc.shape
        prompts = []

        for b in range(B):
            min_value = torch.min(x_enc[b]).item()
            max_value = torch.max(x_enc[b]).item()
            median_value = torch.median(x_enc[b]).item() 
            trend = x_enc[b].diff(dim=0).sum().item()
            trend_direction = "upward" if trend > 0 else "downward"

            prompt_parts = [
                "The radar observation sequences are converted into image sequences using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {frames_out} frames using the past {frames_in} frames.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_manager.max_input_text_length] if len(
                prompt) > self.vlm_manager.max_input_text_length else prompt
            prompts.append(prompt)

        return prompts

    def radar_prediction(self, x_enc, vision_embeddings, text_embeddings):
        B, T, C, H, W = x_enc.shape
        x_enc = x_enc.to(self.device)

        model_results, radar_feature_dict = self.radar_model(
            x_enc,
            return_features=True
        )

        OT = model_results.shape[1]

        radar_features = radar_feature_dict[
            "radar_features"
        ]

        B_, OT_, D, h, w = radar_features.shape

        assert B_ == B
        assert OT_ == OT

        radar_features = radar_features.reshape(B * OT, D, h, w)

        radar_features = self.radar_projection(radar_features)
        d_model = radar_features.shape[1]

        radar_tokens = radar_features.flatten(2) 
        radar_tokens = radar_tokens.transpose(1, 2)
        radar_tokens = radar_tokens.reshape(B, OT, h * w, d_model)

        time_embedding = self.pred_time_embedding[:, :OT]
        radar_tokens = radar_tokens+ time_embedding.unsqueeze(2)
        text_embeddings = text_embeddings.repeat(T, 1)
        vision_token = self.vision_projection(vision_embeddings)
        text_token = self.text_projection(text_embeddings)
        joint_embedding = torch.cat([vision_embeddings, text_embeddings], dim=-1)
        joint_token = self.multimodal_enhancement(joint_embedding)
        multimodal_tokens = torch.stack([vision_token, text_token, joint_token],dim=1)
        multimodal_tokens = self.multimodal_trans(multimodal_tokens.reshape(B, T, 3, d_model).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        radar_tokens = radar_tokens.reshape(B * OT, h * w, d_model)

        multimodal_tokens = multimodal_tokens.reshape(B * OT, 3, d_model)

        radar_tokens_norm = F.normalize(radar_tokens, p=2, dim=-1, eps=1e-6)
        multimodal_tokens_norm = F.normalize(multimodal_tokens, p=2, dim=-1, eps=1e-6)
        cross_features, attention_weights = self.cross_attention(
            query=radar_tokens_norm,
            key=multimodal_tokens_norm,
            value=multimodal_tokens
        )

        cross_features = self.cross_norm(radar_tokens + cross_features)
        cross_features = cross_features.transpose(1, 2) 
        cross_features = cross_features.reshape(B * OT, d_model, h, w)

        multimodal_results = self.multimodal_decoder(cross_features)
        if multimodal_results.shape[-2:] != (H, W):
            multimodal_results = F.interpolate(
                multimodal_results,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        multimodal_results = multimodal_results.reshape(B, OT, C, H, W)
        gate_input = torch.cat([model_results, multimodal_results], dim=2)
        gate_input = gate_input.reshape(B * OT, 2 * C, H, W)
        gate_weights = self.spatial_gate(gate_input)
        gate_weights = torch.softmax(gate_weights, dim=1)
        radar_gate = gate_weights[:, 0:1] 
        multimodal_gate = gate_weights[:, 1:2] 

        model_results_2d = model_results.reshape(B * OT, C, H, W)
        multimodal_results_2d = multimodal_results.reshape(B * OT, C, H, W)
        fused_results = (radar_gate * model_results_2d + multimodal_gate * multimodal_results_2d) 

        fusion_input = torch.cat([model_results_2d, fused_results],dim=1) 
        fusion_residual = self.fusion_layer(fusion_input)
        predictions = (model_results_2d+ self.mm_residual_scale * fusion_residual)

        predictions = predictions.reshape(B, OT, C, H, W)
        return predictions, model_results, radar_feature_dict

    def forward(self, x_enc, y, cmp_fft_loss=False):
        B, T, C, H, W = x_enc.shape

        x_enc, means, stdev = self._normalize_input(x_enc)

        images = self.vision_augmented_learner(x_enc)
        prompts = self.text_augmented_learner(x_enc, self.config.content, self.config.frames_out, self.config.frames_in)

        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(B, images, prompts)

        predictions, radar_base, radar_aux = self.radar_prediction(x_enc, vision_embeddings, text_embeddings)

        predictions = self._denormalize_output(predictions, means, stdev)
        radar_base = self._denormalize_output(radar_base, means, stdev)
        return predictions, radar_base, radar_aux

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        pred, radar_base, radar_aux = self(frames_in, frames_gt, compute_loss)
        if compute_loss:

            final_loss = self.criterion(
                pred,
                frames_gt
            )

            radar_loss_dict = self.radar_model.compute_loss(
                radar_base,
                frames_gt,
                radar_aux
            )

            total_loss = final_loss + 0.2 * radar_loss_dict['total_loss']
            return pred, total_loss
        else:
            return pred, None

def get_model(config, device):
    model = PreMM(config, device)
    return model
