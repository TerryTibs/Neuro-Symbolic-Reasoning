import ast
import sys


def upgrade(in_path="sacrsn_v59_gemini.py", out_path="sacrsn_v60_gemini.py"):
    with open(in_path, "r", encoding="utf-8") as f:
        src = f.read()

    replacements = [
        # 1. Header comment
        (
            "# SACRSN v59 (Dual-Clock Polar + Self-Evolving Holographic Macro-Packets)",
            "# SACRSN v60 (Dual-Clock Polar + Tuned Macro-Packets + Regularized Complex ACT/Stack)",
        ),
        (
            "#   3. Compositional Relational VQ (R_t) + ComplEx Binding + Pure Symbolic SymLM + All v55 Fixes",
            "#   3. v60 Upgrades: 5x Faster Eval Gating, Complex/Label-Smooth Regularization,\n"
            "#      Deeper Macro-Packet Mining, Stronger Pop-Clear Stack Balance, Clean Edge Sampling",
        ),
        # 2. CONFIG updates
        (
            '    "max_macro_phrases": 256,\n'
            '    "macro_min_freq": 6,\n'
            '    "macro_max_act": 2.40,\n'
            '    "macro_min_noop": 0.45,\n'
            '    "macro_min_rel_prob": 0.08,',
            '    "max_macro_phrases": 256,\n'
            '    "macro_scan_cap": 262144,\n'
            '    "macro_min_freq": 6,\n'
            '    "macro_max_act": 2.40,\n'
            '    "macro_min_noop": 0.30,\n'
            '    "macro_min_rel_prob": 0.08,',
        ),
        (
            '    "max_recursion_depth": 4,\n'
            '    "decoder_dropout": 0.10,\n\n'
            '    # Reasoning (PonderNet ACT) & Memory Controls\n'
            '    "act_threshold": 0.85,\n'
            '    "ponder_prior_halt": 0.45,\n'
            '    "ponder_penalty": 0.030,\n'
            '    "use_stack": True,\n'
            '    "stack_temp": 0.75,\n'
            '    "stack_balance_weight": 0.08,',
            '    "max_recursion_depth": 4,\n'
            '    "decoder_dropout": 0.15,\n'
            '    "cell_dropout": 0.10,\n'
            '    "emb_dropout": 0.08,\n'
            '    "label_smoothing": 0.05,\n\n'
            '    # Reasoning (PonderNet ACT) & Memory Controls\n'
            '    "act_threshold": 0.85,\n'
            '    "ponder_prior_halt": 0.40,\n'
            '    "ponder_penalty": 0.030,\n'
            '    "ponder_min_penalty": 0.008,\n'
            '    "use_stack": True,\n'
            '    "stack_temp": 0.75,\n'
            '    "stack_clear_decay": 0.80,\n'
            '    "stack_pop_balance_mult": 3.0,\n'
            '    "stack_balance_weight": 0.08,',
        ),
        # 3. auto_configure_from_text updates
        (
            '        adj_wd = 1e-5 if n_steps < 4000 else 3e-4\n'
            '        adj_drop = 0.0 if n_steps < 4000 else 0.10',
            '        adj_wd = 1e-5 if n_steps < 4000 else 8e-4\n'
            '        adj_drop = 0.0 if n_steps < 4000 else 0.15\n'
            '        adj_cell_drop = 0.0 if n_steps < 4000 else 0.10\n'
            '        adj_emb_drop = 0.0 if n_steps < 4000 else 0.08\n'
            '        adj_lbl_smooth = 0.0 if n_steps < 4000 else 0.05\n'
            '        adj_max_macro = max(256, min(2048, used_stems // 2))',
        ),
        (
            '        config["weight_decay"] = adj_wd\n'
            '        config["decoder_dropout"] = adj_drop',
            '        config["weight_decay"] = adj_wd\n'
            '        config["decoder_dropout"] = adj_drop\n'
            '        config["cell_dropout"] = adj_cell_drop\n'
            '        config["emb_dropout"] = adj_emb_drop\n'
            '        config["label_smoothing"] = adj_lbl_smooth\n'
            '        config["max_macro_phrases"] = adj_max_macro',
        ),
        (
            '    print("=== Auto-Configuration Summary (SACRSN v59 Dual-Clock Polar + Macro-Packets) ===")',
            '    print("=== Auto-Configuration Summary (SACRSN v60 Dual-Clock Polar + Macro-Packets) ===")',
        ),
        (
            '        f"WD={config[\'weight_decay\']:.1e} | Drop={config[\'decoder_dropout\']:.2f} | ClipMax={config[\'grad_clip_max\']:.1f} | "',
            '        f"WD={config[\'weight_decay\']:.1e} | Drop={config[\'decoder_dropout\']:.2f}/{config.get(\'cell_dropout\', 0.0):.2f}/{config.get(\'emb_dropout\', 0.0):.2f} | LS={config.get(\'label_smoothing\', 0.0):.2f} | ClipMax={config[\'grad_clip_max\']:.1f} | "',
        ),
        # 4. complex_dropout + ComplexFFN + ComplexAttention
        (
            'class ComplexLayerNorm(nn.Module):',
            'def complex_dropout(z_r, z_i, p, training):\n'
            '    """Phase-preserving complex dropout (shared mask across real & imag channels)."""\n'
            '    if (not training) or p <= 0.0:\n'
            '        return z_r, z_i\n'
            '    keep = 1.0 - float(p)\n'
            '    mask = (torch.rand(z_r.shape, device=z_r.device, dtype=torch.float32) < keep).to(z_r.dtype) / keep\n'
            '    return z_r * mask, z_i * mask\n\n\n'
            'class ComplexLayerNorm(nn.Module):',
        ),
        (
            'class ComplexFFN(nn.Module):\n'
            '    """Residual 2x complex MLP expansion block with variance-preserving scaling."""\n'
            '    def __init__(self, dim, mult=2):\n'
            '        super().__init__()\n'
            '        hidden = dim * mult',
            'class ComplexFFN(nn.Module):\n'
            '    """Residual 2x complex MLP expansion block with variance-preserving scaling."""\n'
            '    def __init__(self, dim, mult=2, dropout=None):\n'
            '        super().__init__()\n'
            '        hidden = dim * mult\n'
            '        self.dropout = float(CONFIG.get("cell_dropout", 0.0)) if dropout is None else float(dropout)',
        ),
        (
            '        o_r = self.fc2_r(a_r) - self.fc2_i(a_i)\n'
            '        o_i = self.fc2_r(a_i) + self.fc2_i(a_r)\n'
            '        return z_r + 0.5 * o_r, z_i + 0.5 * o_i',
            '        o_r = self.fc2_r(a_r) - self.fc2_i(a_i)\n'
            '        o_i = self.fc2_r(a_i) + self.fc2_i(a_r)\n'
            '        o_r, o_i = complex_dropout(o_r, o_i, self.dropout, self.training)\n'
            '        return z_r + 0.5 * o_r, z_i + 0.5 * o_i',
        ),
        # 5. DifferentiableStack pop clearing
        (
            '        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)\n'
            '        retain_mask = (1.0 - write_mask.unsqueeze(2)) * (1.0 - 0.35 * clear_mask.unsqueeze(2)) * 0.996\n'
            '        new_memory = write_val + (memory * retain_mask)',
            '        clear_decay = float(CONFIG.get("stack_clear_decay", 0.80))\n'
            '        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)\n'
            '        retain_mask = (1.0 - write_mask.unsqueeze(2)) * (1.0 - clear_decay * clear_mask.unsqueeze(2)) * 0.992\n'
            '        new_memory = write_val + (memory * retain_mask)',
        ),
        # 6. ComplexAttention dropout
        (
            'class ComplexAttention(nn.Module):\n'
            '    def __init__(self, dim):\n'
            '        super().__init__()\n'
            '        self.dim = dim',
            'class ComplexAttention(nn.Module):\n'
            '    def __init__(self, dim, dropout=None):\n'
            '        super().__init__()\n'
            '        self.dim = dim\n'
            '        self.dropout = float(CONFIG.get("cell_dropout", 0.0)) if dropout is None else float(dropout)',
        ),
        (
            '        out_r = torch.matmul(attn_weights, v_r_h).squeeze(2).reshape(B, self.dim)\n'
            '        out_i = torch.matmul(attn_weights, v_i_h).squeeze(2).reshape(B, self.dim)\n\n'
            '        return z_r + 0.25 * out_r, z_i + 0.25 * out_i',
            '        out_r = torch.matmul(attn_weights, v_r_h).squeeze(2).reshape(B, self.dim)\n'
            '        out_i = torch.matmul(attn_weights, v_i_h).squeeze(2).reshape(B, self.dim)\n'
            '        out_r, out_i = complex_dropout(out_r, out_i, self.dropout, self.training)\n\n'
            '        return z_r + 0.25 * out_r, z_i + 0.25 * out_i',
        ),
        # 7. UberCRSN slot_ctx_gate + embedding dropout
        (
            '        self.slot_ctx_gate = nn.Parameter(torch.tensor(-2.0))',
            '        self.slot_ctx_gate = nn.Parameter(torch.tensor(-0.5))',
        ),
        (
            '        batch_size = pkt_stems.size(0)\n'
            '        z_r, z_i = self.embed_packet(pkt_stems, pkt_morphs, pkt_lens, ablate_morph=ablate_morph)',
            '        batch_size = pkt_stems.size(0)\n'
            '        z_r, z_i = self.embed_packet(pkt_stems, pkt_morphs, pkt_lens, ablate_morph=ablate_morph)\n'
            '        z_r, z_i = complex_dropout(z_r, z_i, float(CONFIG.get("emb_dropout", 0.0)), self.training)',
        ),
        # 8. discover_macro_trie deeper corpus scan & adaptive threshold
        (
            'def discover_macro_trie(model, atomic_dict, max_phrases=256, max_pkt_len=4):',
            'def discover_macro_trie(model, atomic_dict, max_phrases=None, max_pkt_len=4):',
        ),
        (
            '    n_scan = min(n_total - 1, 49152)\n'
            '    b_scan = 128 if n_scan >= 512 else 1\n'
            '    s_scan = n_scan // b_scan\n'
            '    usable = b_scan * s_scan',
            '    if max_phrases is None:\n'
            '        max_phrases = int(CONFIG.get("max_macro_phrases", 256))\n'
            '    scan_cap = int(CONFIG.get("macro_scan_cap", 262144))\n'
            '    n_scan = min(n_total - 1, scan_cap)\n'
            '    b_scan = (256 if (HIGH_VRAM and n_scan >= 2048) else 128) if n_scan >= 512 else 1\n'
            '    s_scan = n_scan // b_scan\n'
            '    usable = b_scan * s_scan',
        ),
        (
            '    min_freq = CONFIG.get("macro_min_freq", 6)\n'
            '    max_act = CONFIG.get("macro_max_act", 2.40)\n'
            '    min_noop = CONFIG.get("macro_min_noop", 0.45)\n'
            '    min_rel = CONFIG.get("macro_min_rel_prob", 0.08)',
            '    base_min_freq = int(CONFIG.get("macro_min_freq", 6))\n'
            '    min_freq = max(base_min_freq, n_scan // 16384)\n'
            '    max_act = float(CONFIG.get("macro_max_act", 2.40))\n'
            '    cfg_min_noop = float(CONFIG.get("macro_min_noop", 0.30))\n'
            '    mean_corpus_noop = float(noops_mat.mean().item())\n'
            '    min_noop = min(cfg_min_noop, max(0.18, 0.85 * mean_corpus_noop))\n'
            '    min_rel = float(CONFIG.get("macro_min_rel_prob", 0.08))',
        ),
        (
            '    return trie, preview_strs',
            '    return trie, (preview_strs, len(top_selected), n_scan)',
        ),
        # 9. compute_packet_loss_and_bpc exact NLL BPC + optional train label smoothing
        (
            'def compute_packet_loss_and_bpc(\n'
            '    stem_logits,\n'
            '    morph_logits,\n'
            '    edge_logits,\n'
            '    len_logits,\n'
            '    tgt_stems,\n'
            '    tgt_morphs,\n'
            '    tgt_edges,\n'
            '    tgt_lens,\n'
            '    tgt_chars,\n'
            '    macro_active=False,\n'
            '):\n'
            '    B, S_max, V_s = stem_logits.shape\n'
            '    ts = tgt_stems[:, :S_max]\n'
            '    tm = tgt_morphs[:, :S_max]\n'
            '    te = tgt_edges[:, :S_max]\n'
            '    mask = (ts > 0)\n'
            '    n_slots = mask.sum().clamp(min=1)\n\n'
            '    ce_s = F.cross_entropy(stem_logits.reshape(-1, V_s).float(), ts.reshape(-1), reduction="none").view(B, S_max)\n'
            '    ce_m = F.cross_entropy(morph_logits.reshape(-1, morph_logits.size(-1)).float(), tm.reshape(-1), reduction="none").view(B, S_max)\n'
            '    ce_e = F.cross_entropy(edge_logits.reshape(-1, edge_logits.size(-1)).float(), te.reshape(-1), reduction="none").view(B, S_max)\n\n'
            '    loss_stem = (ce_s * mask).sum() / n_slots',
            'def compute_packet_loss_and_bpc(\n'
            '    stem_logits,\n'
            '    morph_logits,\n'
            '    edge_logits,\n'
            '    len_logits,\n'
            '    tgt_stems,\n'
            '    tgt_morphs,\n'
            '    tgt_edges,\n'
            '    tgt_lens,\n'
            '    tgt_chars,\n'
            '    macro_active=False,\n'
            '    label_smoothing=0.0,\n'
            '):\n'
            '    B, S_max, V_s = stem_logits.shape\n'
            '    ts = tgt_stems[:, :S_max]\n'
            '    tm = tgt_morphs[:, :S_max]\n'
            '    te = tgt_edges[:, :S_max]\n'
            '    mask = (ts > 0)\n'
            '    n_slots = mask.sum().clamp(min=1)\n\n'
            '    log_probs_s = F.log_softmax(stem_logits.reshape(-1, V_s).float(), dim=-1)\n'
            '    ce_s = F.nll_loss(log_probs_s, ts.reshape(-1), reduction="none").view(B, S_max)\n'
            '    if label_smoothing > 0.0:\n'
            '        smooth_s = -log_probs_s.mean(dim=-1).view(B, S_max)\n'
            '        ce_s_opt = (1.0 - label_smoothing) * ce_s + label_smoothing * smooth_s\n'
            '    else:\n'
            '        ce_s_opt = ce_s\n\n'
            '    ce_m = F.cross_entropy(morph_logits.reshape(-1, morph_logits.size(-1)).float(), tm.reshape(-1), reduction="none").view(B, S_max)\n'
            '    ce_e = F.cross_entropy(edge_logits.reshape(-1, edge_logits.size(-1)).float(), te.reshape(-1), reduction="none").view(B, S_max)\n\n'
            '    loss_stem = (ce_s_opt * mask).sum() / n_slots',
        ),
        # 10. sample_edge_token + generate_sample_preview cleaner sampling
        (
            '@torch.no_grad()\n'
            'def generate_sample_preview(model, macro_active=False, n_steps_gen=None):',
            'def sample_edge_token(ej_logits, stem_id, temperature=0.35, top_k=6, top_p=0.85):\n'
            '    """Samples boundary edge cleanly: forbids <PAD>/<EDGE_UNK> and avoids gluing multi-char words."""\n'
            '    scores = ej_logits.squeeze(0).float().clone()\n'
            '    if scores.dim() > 1:\n'
            '        scores = scores[0]\n'
            '    scores[tokenizer.edge_pad_id] = -float("inf")\n'
            '    if tokenizer.edge_unk_id < scores.size(-1):\n'
            '        scores[tokenizer.edge_unk_id] = -float("inf")\n'
            '    stem_str = tokenizer.id_to_stem.get(int(stem_id), "")\n'
            '    if int(stem_id) >= 4 and len(stem_str) >= 2 and not stem_str.isdigit():\n'
            '        if tokenizer.edge_empty_id < scores.size(-1):\n'
            '            scores[tokenizer.edge_empty_id] -= 4.0\n'
            '        if tokenizer.edge_cont_id < scores.size(-1):\n'
            '            scores[tokenizer.edge_cont_id] -= 2.5\n'
            '    return sample_next_token(\n'
            '        scores.unsqueeze(0), None, temperature=temperature, top_k=top_k, top_p=top_p,\n'
            '        rep_penalty=1.0, forbid_pad=False\n'
            '    )\n\n\n'
            '@torch.no_grad()\n'
            'def generate_sample_preview(model, macro_active=False, n_steps_gen=None):',
        ),
        (
            '                sj = sample_next_token(sj_logits, recent_stems, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.15)\n'
            '                mj = sample_next_token(mj_logits, None, temperature=0.55, top_k=12, top_p=0.90, rep_penalty=1.0)\n'
            '                ej = sample_next_token(ej_logits, None, temperature=0.60, top_k=16, top_p=0.90, rep_penalty=1.0)',
            '                sj = sample_next_token(sj_logits, recent_stems, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.15)\n'
            '                mj = sample_next_token(mj_logits, None, temperature=0.40, top_k=8, top_p=0.85, rep_penalty=1.0)\n'
            '                ej = sample_edge_token(ej_logits, sj, temperature=0.35, top_k=6, top_p=0.85)',
        ),
        # 11. train() updates: banner, macro unpack/logging, label_smoothing, stack balance, ponder penalty, eval gating
        (
            '    print(f"--- Training SACRSN v59 (BF16_AMP={USE_AMP} | FusedAdamW={fused_used} | TBPTT={CONFIG[\'seq_len\']}) ---")',
            '    print(f"--- Training SACRSN v60 (BF16_AMP={USE_AMP} | FusedAdamW={fused_used} | TBPTT={CONFIG[\'seq_len\']}) ---")',
        ),
        (
            '                    macro_trie, preview_phrases = discover_macro_trie(\n'
            '                        model,\n'
            '                        data_atomic,\n'
            '                        max_phrases=CONFIG.get("max_macro_phrases", 256),\n'
            '                        max_pkt_len=CONFIG.get("max_pkt_len", 4),\n'
            '                    )\n'
            '                    if macro_trie:\n'
            '                        train_pkts = pack_atomic_to_packets(data_atomic, macro_trie=macro_trie, max_pkt_len=CONFIG["max_pkt_len"])\n'
            '                        if val_atomic_slices:\n'
            '                            val_pkts_macro = [pack_atomic_to_packets(va, macro_trie=macro_trie, max_pkt_len=CONFIG["max_pkt_len"]) for va in val_atomic_slices]\n'
            '                            val_batches = build_val_batches(val_pkts_macro)\n'
            '                        macro_active = True\n'
            '                        mean_u_per_pkt = float(train_pkts["lens"].float().mean().item())\n'
            '                        best_metric = float("inf")\n'
            '                        best_epoch = -1\n'
            '                        best_state = None\n'
            '                        ema_bpc = None\n'
            '                        sample_p = ", ".join(f"\'{p}\'(n={c},ACT={a:.2f})" for p, c, a, _ in preview_phrases[:4])\n'
            '                        print(\n'
            '                            f"[Phase 3 Macro-Packets @ Ep {epoch:04d}] Crystallized mastered phrases -> "\n'
            '                            f"{mean_u_per_pkt:.2f} units/pkt | Top: [{sample_p}]"\n'
            '                        )',
            '                    macro_trie, macro_meta = discover_macro_trie(\n'
            '                        model,\n'
            '                        data_atomic,\n'
            '                        max_phrases=CONFIG.get("max_macro_phrases", 256),\n'
            '                        max_pkt_len=CONFIG.get("max_pkt_len", 4),\n'
            '                    )\n'
            '                    if macro_trie:\n'
            '                        preview_phrases, n_macro_sel, n_macro_scan = macro_meta\n'
            '                        train_pkts = pack_atomic_to_packets(data_atomic, macro_trie=macro_trie, max_pkt_len=CONFIG["max_pkt_len"])\n'
            '                        if val_atomic_slices:\n'
            '                            val_pkts_macro = [pack_atomic_to_packets(va, macro_trie=macro_trie, max_pkt_len=CONFIG["max_pkt_len"]) for va in val_atomic_slices]\n'
            '                            val_batches = build_val_batches(val_pkts_macro)\n'
            '                        macro_active = True\n'
            '                        mean_u_per_pkt = float(train_pkts["lens"].float().mean().item())\n'
            '                        pct_multi_pkt = float((train_pkts["lens"] > 1).float().mean().item() * 100.0)\n'
            '                        best_metric = float("inf")\n'
            '                        best_epoch = -1\n'
            '                        best_state = None\n'
            '                        ema_bpc = None\n'
            '                        sample_p = ", ".join(f"\'{p}\'(n={c},ACT={a:.2f})" for p, c, a, _ in preview_phrases[:4])\n'
            '                        print(\n'
            '                            f"[Phase 3 Macro-Packets @ Ep {epoch:04d}] Mined {n_macro_sel} phrases (scan={n_macro_scan:,}) -> "\n'
            '                            f"{mean_u_per_pkt:.2f} units/pkt ({pct_multi_pkt:.1f}% multi) | Top: [{sample_p}]"\n'
            '                        )',
        ),
        (
            '                    loss_pred, nll_step, ch_step, s_ok, m_ok, e_ok, n_sl = compute_packet_loss_and_bpc(\n'
            '                        stem_logits, morph_logits, edge_logits, len_logits,\n'
            '                        tgt_s, tgt_m, tgt_e, tgt_l, tgt_c,\n'
            '                        macro_active=macro_active,\n'
            '                    )',
            '                    loss_pred, nll_step, ch_step, s_ok, m_ok, e_ok, n_sl = compute_packet_loss_and_bpc(\n'
            '                        stem_logits, morph_logits, edge_logits, len_logits,\n'
            '                        tgt_s, tgt_m, tgt_e, tgt_l, tgt_c,\n'
            '                        macro_active=macro_active,\n'
            '                        label_smoothing=float(CONFIG.get("label_smoothing", 0.0)),\n'
            '                    )',
        ),
        (
            '                    loss_stk_bal = CONFIG["stack_balance_weight"] * (\n'
            '                        (avg_stk_ctrl[0] - avg_stk_ctrl[1]) ** 2\n'
            '                        + 1.0 * (F.relu(avg_stk_ctrl[2] - 0.72) ** 2)\n'
            '                    )',
            '                    pop_bal_mult = float(CONFIG.get("stack_pop_balance_mult", 3.0))\n'
            '                    loss_stk_bal = CONFIG["stack_balance_weight"] * (\n'
            '                        pop_bal_mult * ((avg_stk_ctrl[0] - avg_stk_ctrl[1]) ** 2)\n'
            '                        + 1.0 * (F.relu(avg_stk_ctrl[2] - 0.72) ** 2)\n'
            '                    )',
        ),
        (
            '                    if act_std < 0.25 and avg_act_steps <= 2.4:\n'
            '                        live_ponder_penalty = max(0.002, live_ponder_penalty * 0.94)\n'
            '                    elif avg_act_steps > act_high:\n'
            '                        live_ponder_penalty = min(0.15, live_ponder_penalty * 1.05)\n'
            '                    elif avg_act_steps < act_low:\n'
            '                        live_ponder_penalty = max(0.002, live_ponder_penalty * 0.94)',
            '                    ponder_floor = float(CONFIG.get("ponder_min_penalty", 0.008))\n'
            '                    if act_std > 1.15:\n'
            '                        live_ponder_penalty = min(0.08, live_ponder_penalty * 1.02)\n'
            '                    elif act_std < 0.25 and avg_act_steps <= 2.4:\n'
            '                        live_ponder_penalty = max(ponder_floor, live_ponder_penalty * 0.97)\n'
            '                    elif avg_act_steps > act_high:\n'
            '                        live_ponder_penalty = min(0.15, live_ponder_penalty * 1.05)\n'
            '                    elif avg_act_steps < act_low:\n'
            '                        live_ponder_penalty = max(ponder_floor, live_ponder_penalty * 0.97)',
        ),
        (
            '            if is_ready:\n'
            '                t_va_0 = time.time()\n'
            '                val_loss, val_bpc, val_acc, val_m_acc, val_e_acc = evaluate_val(\n'
            '                    model, val_batches, macro_active=macro_active\n'
            '                )\n'
            '                va_time_since_log += time.time() - t_va_0\n'
            '                monitor_metric = val_bpc if val_bpc is not None else ema_bpc\n'
            '                if monitor_metric < best_metric - 1e-3:\n'
            '                    best_metric = monitor_metric\n'
            '                    best_epoch = epoch\n'
            '                    history["best_epoch"] = best_epoch\n'
            '                    t_io_0 = time.time()\n'
            '                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n'
            '                    dirty_best_disk = True\n'
            '                    io_time_since_log += time.time() - t_io_0\n'
            '                    is_new_best = True\n'
            '                    improved_since_log = True\n'
            '                    improved_since_sched = True\n\n'
            '                if epoch >= lr_warmup_epochs and is_eval_epoch:',
            '            if is_ready and is_eval_epoch:\n'
            '                t_va_0 = time.time()\n'
            '                val_loss, val_bpc, val_acc, val_m_acc, val_e_acc = evaluate_val(\n'
            '                    model, val_batches, macro_active=macro_active\n'
            '                )\n'
            '                va_time_since_log += time.time() - t_va_0\n'
            '                monitor_metric = val_bpc if val_bpc is not None else ema_bpc\n'
            '                if monitor_metric < best_metric - 1e-3:\n'
            '                    best_metric = monitor_metric\n'
            '                    best_epoch = epoch\n'
            '                    history["best_epoch"] = best_epoch\n'
            '                    t_io_0 = time.time()\n'
            '                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n'
            '                    dirty_best_disk = True\n'
            '                    io_time_since_log += time.time() - t_io_0\n'
            '                    is_new_best = True\n'
            '                    improved_since_log = True\n'
            '                    improved_since_sched = True\n\n'
            '                if epoch >= lr_warmup_epochs:',
        ),
        # 12. plot_training_dynamics title + visualize_all clean edge sampling
        (
            '    fig.suptitle("SACRSN v59 — Dual-Clock Polar + Macro-Packet Training Dashboard", fontsize=14, fontweight="bold")',
            '    fig.suptitle("SACRSN v60 — Dual-Clock Polar + Macro-Packet Training Dashboard", fontsize=14, fontweight="bold")',
        ),
        (
            '                sj = sample_next_token(model.decoder(feat_j), recent_stems, temperature=0.80, top_k=40, top_p=0.92, rep_penalty=1.10)\n'
            '                mj = sample_next_token(model.morph_decoder(feat_j), None, temperature=0.60, top_k=12, top_p=0.90, rep_penalty=1.0)\n'
            '                ej = sample_next_token(model.edge_decoder(feat_j), None, temperature=0.60, top_k=16, top_p=0.90, rep_penalty=1.0)',
            '                sj = sample_next_token(model.decoder(feat_j), recent_stems, temperature=0.80, top_k=40, top_p=0.92, rep_penalty=1.10)\n'
            '                mj = sample_next_token(model.morph_decoder(feat_j), None, temperature=0.40, top_k=8, top_p=0.85, rep_penalty=1.0)\n'
            '                ej = sample_edge_token(model.edge_decoder(feat_j), sj, temperature=0.35, top_k=6, top_p=0.85)',
        ),
        # 13. dream_mode cleaner morph/edge sampling
        (
            '        m_idx = sample_next_token(morph_logits, None, temperature=0.5, top_k=8, top_p=0.90, rep_penalty=1.0)\n'
            '        e_idx = sample_next_token(edge_logits, None, temperature=0.55, top_k=12, top_p=0.90, rep_penalty=1.0)',
            '        m_idx = sample_next_token(morph_logits, None, temperature=0.40, top_k=8, top_p=0.85, rep_penalty=1.0)\n'
            '        e_idx = sample_edge_token(edge_logits, s_idx, temperature=0.35, top_k=6, top_p=0.85)',
        ),
        # 14. run_pddl_probe disable dropout during probe
        (
            '        CONFIG["embedding_dim"] = 64\n'
            '        CONFIG["act_threshold"] = 0.85',
            '        CONFIG["embedding_dim"] = 64\n'
            '        CONFIG["act_threshold"] = 0.85\n'
            '        CONFIG["decoder_dropout"] = 0.0\n'
            '        CONFIG["cell_dropout"] = 0.0\n'
            '        CONFIG["emb_dropout"] = 0.0\n'
            '        CONFIG["label_smoothing"] = 0.0',
        ),
    ]

    out = src
    # These three display banners in the supplied source are still labelled v56,
    # even though the file header and the actual architecture are v59. Accept
    # either banner version, but still require exactly one match.
    legacy_banner_patches = {6, 23, 29}

    for idx, (old, new) in enumerate(replacements):
        matched_old = old
        cnt = out.count(matched_old)

        if cnt == 0 and idx in legacy_banner_patches:
            legacy_old = old.replace("v59", "v56")
            legacy_count = out.count(legacy_old)
            if legacy_count == 1:
                matched_old = legacy_old
                cnt = legacy_count

        if cnt != 1:
            raise RuntimeError(
                f"Patch #{idx} matched {cnt} times (expected exactly 1). "
                "The input file may be the wrong version or differ from the "
                "source this updater was designed for. No output was written."
            )

        out = out.replace(matched_old, new, 1)

    ast.parse(out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Upgraded {in_path} -> {out_path} ({len(out.splitlines())} lines, AST verified).")


if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "sacrsn_v59_gemini.py"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "sacrsn_v60_gemini.py"
    upgrade(in_p, out_p)