import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration

from isa_data.isa import TASK_NAMES, EXTRA_COLUMNS


class RouterMoEFFN(nn.Module):

    def __init__(self, original_ffn, num_experts, top_k, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)

        self.layer_norm = copy.deepcopy(original_ffn.layer_norm)
        self.dropout = copy.deepcopy(original_ffn.dropout)
        self.experts = nn.ModuleList(
            [copy.deepcopy(original_ffn.DenseReluDense) for _ in range(self.num_experts)]
        )

        self._pending_gate_probs = None

    def set_routing_context(self, gate_probs):
        self._pending_gate_probs = gate_probs

    def forward(self, hidden_states):
        if self._pending_gate_probs is None:
            raise RuntimeError(
                "RouterMoEFFN forward called before set_routing_context; "
                "the parent FFNMoET5ForConditionalGeneration must run the "
                "router prior to invoking the backbone."
            )

        forwarded_states = self.layer_norm(hidden_states)
        gate_probs_fp32 = self._pending_gate_probs.to(
            device=hidden_states.device, dtype=torch.float32
        )

        topk_scores, topk_indices = torch.topk(gate_probs_fp32, k=self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        sparse_gate = torch.zeros_like(gate_probs_fp32)
        sparse_gate.scatter_(1, topk_indices, topk_scores)

        moe_output = torch.zeros_like(forwarded_states)
        for expert_idx, expert in enumerate(self.experts):
            sample_mask = sparse_gate[:, expert_idx] > 0
            if not torch.any(sample_mask):
                continue
            sample_indices = torch.nonzero(sample_mask, as_tuple=False).squeeze(-1)
            expert_out = expert(forwarded_states[sample_indices])
            weights = sparse_gate[sample_indices, expert_idx].to(expert_out.dtype).view(-1, 1, 1)
            moe_output.index_add_(
                0, sample_indices, (weights * expert_out).to(moe_output.dtype)
            )

        return hidden_states + self.dropout(moe_output)


class FFNMoET5ForConditionalGeneration(nn.Module):

    def __init__(
        self,
        model_name,
        num_experts=8,
        encoder_moe_layers=None,
        decoder_moe_layers=None,
        task_embedding_dim=64,
        router_hidden_dim=1024,
        top_k=2,
    ):
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.base_model.config
        self.generation_config = self.base_model.generation_config
        self.main_input_name = self.base_model.main_input_name

        self.num_experts = int(num_experts)
        self.num_tasks = len(TASK_NAMES)
        self.encoder_moe_layers = list(encoder_moe_layers or [])
        self.decoder_moe_layers = list(decoder_moe_layers or [])
        self._has_decoder_moe = len(self.decoder_moe_layers) > 0

        self._moe_layers: list[RouterMoEFFN] = []

        self.task_embedding = nn.Embedding(self.num_tasks, int(task_embedding_dim))
        self.task_gates = nn.ModuleList()

        for layer_idx in self.encoder_moe_layers:
            self._install_moe_layer(
                block=self.base_model.encoder.block[layer_idx],
                layer_name=f"encoder.{layer_idx}",
                top_k=top_k,
                task_embedding_dim=task_embedding_dim,
                router_hidden_dim=router_hidden_dim,
            )
        for layer_idx in self.decoder_moe_layers:
            self._install_moe_layer(
                block=self.base_model.decoder.block[layer_idx],
                layer_name=f"decoder.{layer_idx}",
                top_k=top_k,
                task_embedding_dim=task_embedding_dim,
                router_hidden_dim=router_hidden_dim,
            )

    def _install_moe_layer(self, block, layer_name, top_k, task_embedding_dim, router_hidden_dim):
        original_ffn = block.layer[-1]
        moe_ffn = RouterMoEFFN(
            original_ffn=original_ffn,
            num_experts=self.num_experts,
            top_k=top_k,
            layer_name=layer_name,
        )
        block.layer[-1] = moe_ffn
        self._moe_layers.append(moe_ffn)
        self.task_gates.append(nn.Sequential(
            nn.LayerNorm(int(task_embedding_dim)),
            nn.Linear(int(task_embedding_dim), int(router_hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(router_hidden_dim), self.num_experts),
        ))

    def _set_routing_context(self, task_ids):
        task_emb = self.task_embedding(task_ids)
        gate_probs_list = []
        for gate, moe_layer in zip(self.task_gates, self._moe_layers, strict=True):
            logits = gate(task_emb)
            probs_fp32 = F.softmax(logits.float(), dim=-1)
            moe_layer.set_routing_context(probs_fp32)
            gate_probs_list.append(probs_fp32)
        return gate_probs_list

    def _compute_task_separation_loss(self, task_ids, gate_probs_per_layer):
        zero = gate_probs_per_layer[0].new_zeros(())
        unique_tasks = torch.unique(task_ids)
        if unique_tasks.numel() < 2:
            return zero

        layer_losses = []
        for gate_probs in gate_probs_per_layer:
            task_means = torch.stack(
                [gate_probs[task_ids == task].mean(dim=0) for task in unique_tasks],
                dim=0,
            )
            task_means = F.normalize(task_means, p=2, dim=-1)
            sim = task_means @ task_means.transpose(0, 1)
            off_diag = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
            layer_losses.append(sim.masked_select(off_diag).mean())
        return torch.stack(layer_losses).mean()

    @staticmethod
    def _strip_dataset_columns(kwargs):
        for key in EXTRA_COLUMNS:
            kwargs.pop(key, None)

    def forward(self, input_ids, attention_mask, labels, task_id, **kwargs):
        self._strip_dataset_columns(kwargs)
        kwargs.pop("num_items_in_batch", None)

        gate_probs_list = self._set_routing_context(task_id)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        self._last_task_separation_loss = self._compute_task_separation_loss(
            task_id, gate_probs_list
        )
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, task_id, **kwargs):
        self._strip_dataset_columns(kwargs)

        num_beams = kwargs.get("num_beams") or self.generation_config.num_beams or 1
        if self._has_decoder_moe and num_beams > 1:
            raise ValueError(
                "Decoder MoE layers are incompatible with beam search "
                f"(num_beams={num_beams}). Use greedy / sampling decoding "
                "or disable --decoder_moe_layers."
            )

        self._set_routing_context(task_id)
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
