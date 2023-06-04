import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from m3ae.modules import objectives, m3ae_utils
from m3ae.modules import prediction_heads
from m3ae.modules.language_encoders.bert_model import BertCrossLayer
from m3ae.modules.m3ae_utils import init_weights
from m3ae.modules.vision_encoders import swin_transformer as swin
from m3ae.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding
import torch.nn.functional as F


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

# 尝试1：在clip抽取特征后，将特征穿过UNET网络
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(256, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        # x = self.up3(x3, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class M3AETransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # == Begin: 1. Build Models ==
        self.is_clip = ('swin' not in config['vit'])
        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        elif 'bert' in config['tokenizer']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            raise ValueError

        resolution_after = config['image_size']
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        if self.is_clip:
            self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vision_encoder = getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
            self.vision_pooler = nn.AdaptiveAvgPool1d(1)
        if 'roberta' in config['tokenizer']:
            self.language_encoder = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.language_encoder = BertModel.from_pretrained(config['tokenizer'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        # == End  : 1. Build Models ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mlm"] > 0:
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        if config["loss_names"]["mim"] > 0:
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)
        # == End  : 2. Build Pre-Training Heads ==

        # == Begin: 3. Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=resolution_after,
                                                     patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)
        # == End  : 3. Load Models ==

        # == 4. Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"]
            self.vqa_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_head.apply(init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            ms = self.hparams.config["melinda_label_size"][self.hparams.config["label_column_name"]]
            self.cls_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.cls_head.apply(init_weights)
        

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.irtr_head = nn.Linear(hs * 2, 1)
            self.irtr_head.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.irtr_head.bias.data = self.itm_head.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_head.parameters():
                p.requires_grad = False

        # todo: 增加分割下游任务
        if self.hparams.config["loss_names"]["seg"] > 0:
            ss = self.hparams.config["octa_image_size"]["height"] * self.hparams.config["octa_image_size"]["width"] # * self.hparams.config["octa_image_size"]["channel"]
            self.seg_head_linear = nn.Sequential(
                nn.Linear(72192, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ss),
            )
            # self.seg_head_conv = nn.Sequential(
            #     nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            #     nn.BatchNorm2d(1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            #     nn.BatchNorm2d(1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(2),
            # )
            self.seg_head_conv = UNet(n_channels=1, n_classes=2, bilinear=False)

        m3ae_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  4. Build Heads For Downstream Tasks ==

        # == Begin: 5. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 5. Load Models For Testing ==

    # infer中调用

# 该方法名为random_masking，接受两个参数：输入张量 x和遮盖比率 mask_ratio。
# 其中，x表示模型的输入数据，通常是一个形状为 [batch_size, sequence_length, embedding_dim] 的三维张量，mask_ratio是一个浮点数，表示要遮盖的比例，取值范围为 [0, 1]。
# 该方法的作用是将 x 中的一部分内容随机遮盖，返回遮盖后的结果，以及对应的二进制遮盖掩码。

# 具体地，该方法首先将输入张量的第一个元素（通常为特殊标记或类别标记）单独取出，然后对剩余部分进行遮盖。
# 遮盖的方式是首先生成一个形状与 x 相同的噪声张量 noise，其中每个元素的取值在 [0, 1] 之间。然后将 noise 按行升序排列，
# 并记录每个元素在排列后的位置。排列后，取前 L*(1-mask_ratio) 个位置对应的元素作为保留的内容，其余的元素则被遮盖。

# 接下来，该方法生成一个二进制遮盖掩码 mask，其中每个元素为0或1，0表示对应位置的元素被保留，1表示被遮盖。生成 mask 的方式是先生成一个全1的张量，
# 然后将前 L*(1-mask_ratio) 个位置上的元素设为0。最后，使用记录好的位置信息将 mask 进行逆排序，即可得到最终的遮盖掩码。
# 最后，该方法将遮盖后的内容与第一个元素拼接起来，形成最终的输出张量。同时返回二进制遮盖掩码和逆排序后的位置信息。
    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        # 这行代码的作用是根据生成的 ids_keep 遮盖掉 x 中的一部分元素，生成一个遮盖后的新的张量 x_masked。
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        # print(x_masked.shape)
        # print(mask.shape)
        # x_masked.reshape(1,400,400)
        # import torchvision
        # torchvision.utils.save_image(x_masked, "2.bmp")
        # torchvision.utils.save_image(mask, "3.bmp")


        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # 在forward中，每一步需要先执行infer
    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key][0]
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # print(img.shape)
        # import torchvision
        # torchvision.utils.save_image(img, "1.bmp")
        device = text_ids.device
        # == End  : Fetch the inputs ==

        # == Begin: Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==

        activations = []
        # print(mask_image)
        # == Begin: Image Encoding ==
        if mask_image:
            # == Begin: Image Masking ==
            # Mask: length -> length * mask_ratio
            # Perform position embedding inside the masking function
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
            print(uni_modal_image_feats.shape)
            # import torchvision
            # torchvision.utils.save_image(uni_modal_image_feats.reshape(1, 400 ,400), "2.bmp")

            # 第二个参数是mask率，第一个返回值是mask后的结果，第二个返回值是mask，第三个是mask恢复
            uni_modal_image_feats, mim_masks, mim_ids_restore = self.random_masking(uni_modal_image_feats,
                                                                                    self.hparams.config["mim_prob"])
                                                                            
            uni_modal_image_feats = self.vision_encoder.forward_trans(uni_modal_image_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
            # == End  : Image Masking ==
        else:
            # 加：（经过vision encoder，出来特征和各个层级的特征）
            uni_modal_image_feats, activations = self.vision_encoder(img)
        # input_image_embed_size -> hidden size
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                 device=device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 device)
        # == End  : Image Encoding ==

        # == Begin: Assign Type Embeddings ==
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None
        x, y = uni_modal_text_feats, uni_modal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                  self.multi_modal_vision_layers)):
            # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
            if mask_image and self.hparams.config["mim_layer"] == layer_idx:
                ret[f"multi_modal_text_feats_{layer_idx}"], ret[f"multi_modal_image_feats_{layer_idx}"] = x, y
            # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
            # == Begin: Co-Attention ==
            x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
            # == End: Co-Attention ==
            # == Begin: For visualization: Return the attention weights ==
            if output_attentions:
                ret["attentions"]["text2image_attns"].append(x1[1:])
                ret["attentions"]["image2text_attns"].append(y1[1:])
            # == End  : For visualization: Return the attention weights ==
        # == End  : Multi-Modal Fusion ==

        # == Begin: == Output Multi-Modal Features ==
        # 1. 将模型放到其他分类任务上
        # 2. 增加分割任务
        multi_modal_text_feats, multi_modal_image_feats = x, y
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        if self.is_clip:
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        else:
            avg_image_feats = self.vision_pooler(multi_modal_image_feats.transpose(1, 2)).view(
                multi_modal_image_feats.size(0), 1, -1)
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(avg_image_feats)
        # 1 * 128 * 768
        # print(multi_modal_text_feats.shape)
        # 1 * 626 * 768
        # print(multi_modal_image_feats.shape)
        multi_modal_seg_feats = torch.cat((multi_modal_text_feats, multi_modal_image_feats), dim=1)

        # 1, 1536
        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        # print(multi_modal_cls_feats.shape)
        # == End  : == Output Multi-Modal Features ==

        ret.update({
            "images": img,
            "patched_images": self.patchify(img),
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
            "multi_modal_seg_feats": multi_modal_seg_feats
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Pre-Training: Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Segmentation
        if "seg" in self.current_tasks:
            ret.update(objectives.compute_seg(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)
