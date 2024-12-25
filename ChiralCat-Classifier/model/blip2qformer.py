import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import numpy as np
# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
# from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.blip2 import Blip2Base
from model.dist_funs import pl_concat_all_gather
# from pytorch_lightning.utilities import distributed

# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     # if use distributed training
#     if not is_dist_avail_and_initialized():
#         return tensor

#     tensors_gather = [
#         torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     print('running here')
#     return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        alpha = [0.07, 0.07, 1.0, 1.0, 0.8]
        alpha = np.array(alpha)
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, device):
        # Convert targets to one-hot encoding
        self.alpha = self.alpha.to(device)
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Compute the log probability
        logpt = F.log_softmax(inputs, dim=1)
        
        # Get the probability for the true class
        pt = torch.exp(logpt)

        # Compute focal loss
        if self.alpha is not None:
            at = self.alpha.to(inputs.device)[targets.argmax(dim=1)]
            at = at.view(-1, 1)
        else:
            at = torch.ones(targets.size(0), 1, device=inputs.device)
        
        focal_loss = -at * (1 - pt) ** self.gamma * logpt
        loss = focal_loss * targets
        loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        args=None,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        self.args = args
        self.tokenizer = self.init_tokenizer()
        if args.use_3d:
            self.graph_encoder, self.ln_graph, self.dictionary = self.init_unimol_encoder(args)
        else:
            self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)

        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.classifier = nn.Sequential(
                nn.Linear(self.Qformer.config.hidden_size, embed_dim),
                nn.Dropout(0.29),
                nn.LeakyReLU(),
                nn.Linear(embed_dim, 5)
                        )
        self.floss = FocalLoss()
        self.temperature = temperature

    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(dim=-1) # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(dim=-2) # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss
        
    def forward_old(self, batch):
        graph, text, _ = batch
        batch_node, batch_mask = self.graph_encoder(*graph)
        batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]
        batch_node = self.ln_graph(batch_node)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        sim_g2t, sim_t2g, loss_gtc = self.contrast(graph_feats, text_feats, return_sim=True)
        
        g_emb = batch_node
        g_mask = batch_mask
        text_ids = text.input_ids.clone()
        mask = text.attention_mask
        with torch.no_grad():
            weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
            weights_t2g.fill_diagonal_(0)
            weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
            weights_g2t.fill_diagonal_(0)

        # select a negative graph for each text
        graph_embeds_neg = []
        graph_mask_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2g[b], 1).item()
            graph_embeds_neg.append(g_emb[neg_idx])
            graph_mask_neg.append(g_mask[neg_idx])
        
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
        graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_g2t[b], 1).item()
            text_ids_neg.append(text_ids[neg_idx])
            text_atts_neg.append(mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_ids, text_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [mask, mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text.input_ids.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        graph_embeds_all = torch.cat([g_emb, graph_embeds_neg, g_emb], dim=0)  # pos, neg, pos
        graph_atts_all = torch.cat([g_mask, graph_mask_neg, g_mask], dim=0)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=graph_embeds_all,
            encoder_attention_mask=graph_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[: batch_size, :query_tokens_itm.size(1), :] # keep query tokens only
        vl_embeddings = torch.max(vl_embeddings, dim=1)[0]
        logits = self.classifier(vl_embeddings)

        return logits




    def forward(self, batch):
        ## for 3d forward
        device = self.device
        graph_batch, text_batch, cls_labels = batch
        batch_node, batch_mask = self.graph_encoder(*graph_batch)
        if not self.tune_gnn:
            batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]
        batch_node = self.ln_graph(batch_node)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text_batch.input_ids, attention_mask=text_batch.attention_mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)

        ###============== Chirality Classification ===================###
        ## not aggregate global tensor because of their different shapes
        g_emb_world = batch_node
        g_mask_world = batch_mask
        text_ids_world = text_batch.input_ids
        text_mask_world = text_batch.attention_mask
        with torch.no_grad():
            weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
            weights_t2g.fill_diagonal_(0)
            weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
            weights_g2t.fill_diagonal_(0)

        # select a negative graph for each text
        graph_embeds_neg = []
        graph_mask_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2g[b], 1).item()
            graph_embeds_neg.append(g_emb_world[neg_idx])
            graph_mask_neg.append(g_mask_world[neg_idx])
        
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
        graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_g2t[b], 1).item()
            text_ids_neg.append(text_ids_world[neg_idx])
            text_atts_neg.append(text_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_batch.input_ids, text_batch.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_batch.attention_mask, text_batch.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
        graph_atts_all = torch.cat([batch_mask, graph_mask_neg, batch_mask], dim=0)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=graph_embeds_all,
            encoder_attention_mask=graph_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[ : batch_size, : query_tokens_itm.size(1), :] # keep query tokens only
        vl_embeddings = torch.max(vl_embeddings, dim=1)[0]
        logits = self.classifier(vl_embeddings)
        cls_labels = torch.tensor(cls_labels).to(device)
        loss_chirality = self.floss(logits, cls_labels, device)
        return loss_chirality
        

    def graph_forward(self, graph):
        if self.args.use_3d:
            batch_node, batch_mask = self.graph_encoder(*graph)
        else:
            batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = self.ln_graph(batch_node)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, batch_mask

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_gtm(self, batch_node, batch_mask, text_ids, text_atts):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        text_ids shape = [B, N]
        text_atts shape = [B, N]
        '''
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            batch_node.device
        ) # shape = [B, Nq]
        attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
        output_gtm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            return_dict=True,
        )
        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        return gtm_logit

