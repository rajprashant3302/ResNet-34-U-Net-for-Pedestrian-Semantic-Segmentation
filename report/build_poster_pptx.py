import sys, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_bw_poster():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.500)
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Pure Black & White / Grayscale Palette
    c_black = RGBColor(0, 0, 0)
    c_dark_gray = RGBColor(40, 40, 40)
    c_mid_gray = RGBColor(100, 100, 100)
    c_light_gray = RGBColor(230, 230, 230)
    c_zebra_gray = RGBColor(245, 245, 245)
    c_white = RGBColor(255, 255, 255)
    c_card_border = RGBColor(0, 0, 0)

    # 1. Slide Background (Pure White)
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.500))
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = c_white
    bg_shape.line.fill.background()

    # 2. Header Banner (Solid Black with White Text)
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.08), Inches(0.08), Inches(13.173), Inches(1.10))
    header.fill.solid()
    header.fill.fore_color.rgb = c_black
    header.line.color.rgb = c_black
    header.line.width = Pt(1.5)

    # Add IIT Patna Logo (Left side) - NO IMPACT LAB LOGO
    # Use B&W logo version for clean monochrome look
    logo_path = "report/iitp_logo_bw.png"
    if not os.path.exists(logo_path):
        logo_path = "report/iitp_logo.png"
    if os.path.exists(logo_path):
        slide.shapes.add_picture(logo_path, Inches(0.18), Inches(0.12), height=Inches(1.02))

    # Header Text Box
    tb_header = slide.shapes.add_textbox(Inches(1.40), Inches(0.10), Inches(11.75), Inches(1.05))
    tf = tb_header.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0

    p0 = tf.paragraphs[0]
    p0.text = "An Empirical Investigation of Controlled Geometric Data Augmentation in U-Net Architectures for Pedestrian Semantic Segmentation"
    p0.font.bold = True
    p0.font.size = Pt(16.5)
    p0.font.color.rgb = c_white
    p0.alignment = PP_ALIGN.LEFT

    p1 = tf.add_paragraph()
    p1.text = "Prashant Raj  |  Department of Computer Science & Engineering, Indian Institute of Technology Patna"
    p1.font.bold = True
    p1.font.size = Pt(11)
    p1.font.color.rgb = c_light_gray
    p1.alignment = PP_ALIGN.LEFT

    p2 = tf.add_paragraph()
    p2.text = "Academic Module: Machine Learning Laboratory  |  Project Track: Problem 4 (U-Net Segmentation)  |  Session 2026–2027"
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = RGBColor(200, 200, 200)
    p2.alignment = PP_ALIGN.LEFT

    # Helper function for Section Cards in Black & White
    def add_card(left, top, width, height, title):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = c_white
        card.line.color.rgb = c_card_border
        card.line.width = Pt(1.2)

        # Title bar: Solid Black Title Header Tab
        tab = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left + 0.01), Inches(top + 0.01), Inches(width - 0.02), Inches(0.28))
        tab.fill.solid()
        tab.fill.fore_color.rgb = c_black
        tab.line.fill.background()

        t_frame = tab.text_frame
        t_frame.margin_top = t_frame.margin_bottom = t_frame.margin_left = t_frame.margin_right = 0
        p = t_frame.paragraphs[0]
        p.text = "  " + title.upper()
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = c_white
        p.alignment = PP_ALIGN.LEFT
        return card

    # =========================================================================
    # COLUMN 1: LEFT (Width = 3.75 in, Left = 0.08 in)
    # =========================================================================
    col1_left = 0.08
    col1_w = 3.75

    # 1.1 Abstract Card
    add_card(col1_left, 1.22, col1_w, 1.68, "Abstract")
    tb_abs = slide.shapes.add_textbox(Inches(col1_left + 0.10), Inches(1.54), Inches(col1_w - 0.20), Inches(1.30))
    tf_abs = tb_abs.text_frame
    tf_abs.word_wrap = True
    tf_abs.margin_top = tf_abs.margin_bottom = tf_abs.margin_left = tf_abs.margin_right = 0
    p = tf_abs.paragraphs[0]
    p.text = "Dense visual perception of pedestrians via semantic segmentation is essential for autonomous vehicles and mobile robotics. We empirically investigate whether controlled geometric data augmentations (horizontal reflection and scale zooming) yield statistically significant benefits when fine-tuning an asymmetric ResNet-34 U-Net on the compact Penn-Fudan Pedestrian Database. Under frozen random seeds and a leak-free stratified partition, paired statistical hypothesis testing demonstrates that transfer-learned encoders already embody strong spatial invariances, challenging heuristic augmentation assumptions while highlighting post-training calibration as a superior performance lever."
    p.font.size = Pt(8.5)
    p.font.color.rgb = c_black

    # 1.2 Introduction Card
    add_card(col1_left, 2.94, col1_w, 1.95, "Introduction & Motivation")
    tb_intro = slide.shapes.add_textbox(Inches(col1_left + 0.10), Inches(3.26), Inches(col1_w - 0.20), Inches(1.58))
    tf_intro = tb_intro.text_frame
    tf_intro.word_wrap = True
    tf_intro.margin_top = tf_intro.margin_bottom = tf_intro.margin_left = tf_intro.margin_right = 0
    bullets_intro = [
        "Pixel-Level Visual Perception: Modern autonomous navigation requires fine-grained boundary delineation beyond coarse bounding boxes for reliable collision avoidance.",
        "The U-Net Paradigm: Lateral skip connections resolve spatial downsampling bottlenecks by routing high-resolution features directly to the synthesis decoder.",
        "Data Scarcity & Augmentation Dilemma: Academic pedestrian benchmarks often contain few samples (170 in Penn-Fudan). Practitioners universally prescribe geometric transforms, but their true marginal utility under deep transfer learning is rarely validated with paired statistical tests."
    ]
    for i, b in enumerate(bullets_intro):
        p = tf_intro.paragraphs[0] if i == 0 else tf_intro.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(8.5)
        p.font.color.rgb = c_black
        p.space_after = Pt(2)

    # 1.3 Problem Statement & Hypotheses
    add_card(col1_left, 4.93, col1_w, 1.72, "Problem Formulation & Hypotheses")
    tb_prob = slide.shapes.add_textbox(Inches(col1_left + 0.10), Inches(5.24), Inches(col1_w - 0.20), Inches(1.36))
    tf_prob = tb_prob.text_frame
    tf_prob.word_wrap = True
    tf_prob.margin_top = tf_prob.margin_bottom = tf_prob.margin_left = tf_prob.margin_right = 0
    bullets_prob = [
        "Mathematical Objective: Learn posterior mapping f_θ: I ∈ R^(256×256×3) → P_hat ∈ [0, 1]^(256×256) predicting pedestrian occupancy under severe spatial imbalance (17.28% foreground vs. 82.72% background).",
        "Null Hypothesis (H₀): Controlled geometric augmentation (horizontal flip p=0.5, zoom ±20%) produces no statistically significant difference in test IoU over unaugmented training (ΔIoU = 0).",
        "Alternative Hypothesis (H₁): Geometric transforms act as inductive regularizers yielding statistically significant IoU improvements (ΔIoU > 0)."
    ]
    for i, b in enumerate(bullets_prob):
        p = tf_prob.paragraphs[0] if i == 0 else tf_prob.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(8.3)
        p.font.color.rgb = c_black
        p.space_after = Pt(2)

    # =========================================================================
    # COLUMN 2: CENTER (Width = 5.75 in, Left = 3.90 in)
    # =========================================================================
    col2_left = 3.90
    col2_w = 5.75

    # 2.1 Key Contributions Card
    add_card(col2_left, 1.22, col2_w, 1.35, "Key Technical Contributions")
    tb_contrib = slide.shapes.add_textbox(Inches(col2_left + 0.10), Inches(1.52), Inches(col2_w - 0.20), Inches(1.00))
    tf_contrib = tb_contrib.text_frame
    tf_contrib.word_wrap = True
    tf_contrib.margin_top = tf_contrib.margin_bottom = tf_contrib.margin_left = tf_contrib.margin_right = 0
    contribs = [
        "1. Leak-Free Stratified Partition: 70/15/15 split jointly balanced across camera provenance (Penn vs. Fudan) and pedestrian area tertiles, ensuring equal 17.3% foreground balance.",
        "2. ResNet-34 U-Net with Full-Res Bypass: Auxiliary skip connection ('up0') conveying raw 256×256 RGB spatial gradients directly to the terminal 16-channel decoder head.",
        "3. Compound BCE + Soft-Dice Loss: Resolves severe foreground gradient starvation caused by 82.72% dominant background pixels, providing stable optimization.",
        "4. Paired Hypothesis Refutation: Non-parametric bootstrap (95% CI) and Wilcoxon signed-rank test rigorously refute the heuristic augmentation benefit under transfer learning."
    ]
    for i, c in enumerate(contribs):
        p = tf_contrib.paragraphs[0] if i == 0 else tf_contrib.add_paragraph()
        p.text = c
        p.font.size = Pt(8.5)
        p.font.color.rgb = c_black
        p.space_after = Pt(1.5)

    # 2.2 Proposed Methodology & Architecture
    add_card(col2_left, 2.61, col2_w, 2.30, "Proposed ResNet-34 U-Net Architecture")
    arch_img = "report/unet_architecture_bw.png"
    if not os.path.exists(arch_img):
        arch_img = "report/unet_architecture.png"
    if os.path.exists(arch_img):
        slide.shapes.add_picture(arch_img, Inches(col2_left + 0.12), Inches(2.90), width=Inches(5.51), height=Inches(1.70))
    tb_arch_cap = slide.shapes.add_textbox(Inches(col2_left + 0.10), Inches(4.63), Inches(col2_w - 0.20), Inches(0.24))
    tf_ac = tb_arch_cap.text_frame
    tf_ac.margin_top = tf_ac.margin_bottom = tf_ac.margin_left = tf_ac.margin_right = 0
    p = tf_ac.paragraphs[0]
    p.text = "Figure 1: End-to-end ResNet-34 U-Net (24.44M params) with auxiliary full-resolution RGB bypass ('up0') and 5-stage bilinear decoder."
    p.font.size = Pt(8)
    p.font.color.rgb = c_mid_gray
    p.alignment = PP_ALIGN.CENTER

    # 2.3 Algorithm Card
    add_card(col2_left, 4.95, col2_w, 1.70, "Algorithm 1: End-to-End Training & Paired Significance Testing")
    tb_alg = slide.shapes.add_textbox(Inches(col2_left + 0.10), Inches(5.24), Inches(col2_w - 0.20), Inches(1.36))
    tf_alg = tb_alg.text_frame
    tf_alg.word_wrap = True
    tf_alg.margin_top = tf_alg.margin_bottom = tf_alg.margin_left = tf_alg.margin_right = 0
    alg_lines = [
        "Require: Dataset D = 170 photographs; Regimes A ∈ {Baseline (zero aug), Proposed (flip + zoom)}; Seeds S.",
        "1: Stratified Partition: Split D → D_train (118), D_val (26), D_test (26) conditioned on provenance × scale tertiles.",
        "2: For each condition A and frozen seed s ∈ S:",
        "3:    Initialize ResNet34-UNet(θ); AdamW (Dec LR = 3e-4, Enc LR = 1.5e-4, Weight Decay = 1e-4, AMP FP16).",
        "4:    For epoch = 1 to 60: θ ← train_epoch(θ, D_train, A, CosineAnnealing(step)); Eval IoU_val ← eval(θ, D_val).",
        "5:    Checkpoint optimal weights θ* based strictly on unaugmented validation IoU (patience = 25 epochs).",
        "6:    Load θ*; Evaluate held-out test split D_test: compute per-image IoU, Precision, Recall, F1, Boundary-F1 (τ_tol=2px).",
        "7: Execute paired non-parametric bootstrap (B=1000 resamples, 95% CI) and Wilcoxon signed-rank test on ΔIoU."
    ]
    for i, al in enumerate(alg_lines):
        p = tf_alg.paragraphs[0] if i == 0 else tf_alg.add_paragraph()
        p.text = al
        p.font.size = Pt(8.2)
        p.font.name = "Consolas"
        p.font.color.rgb = c_black
        p.space_after = Pt(1.5)

    # =========================================================================
    # COLUMN 3: RIGHT (Width = 3.60 in, Left = 9.72 in)
    # =========================================================================
    col3_left = 9.72
    col3_w = 3.53

    # 3.1 Quantitative Results Card
    add_card(col3_left, 1.22, col3_w, 2.70, "Quantitative Results & Judging Parameters")
    
    # Native Table inside Card 3.1 in Black & White
    rows, cols = 6, 6
    tbl_left = Inches(col3_left + 0.06)
    tbl_top = Inches(1.52)
    tbl_w = Inches(col3_w - 0.12)
    tbl_h = Inches(1.50)
    table_shape = slide.shapes.add_table(rows, cols, tbl_left, tbl_top, tbl_w, tbl_h)
    tbl = table_shape.table

    tbl.columns[0].width = Inches(1.10)
    tbl.columns[1].width = Inches(0.48)
    tbl.columns[2].width = Inches(0.48)
    tbl.columns[3].width = Inches(0.48)
    tbl.columns[4].width = Inches(0.48)
    tbl.columns[5].width = Inches(0.48)

    headers = ["Model / Variant", "Prec", "Rec", "F1", "IoU", "B-F1"]
    data = [
        ["Baseline (No Aug)", "0.8578", "0.9396", "0.8920", "0.8110", "0.7985"],
        ["  + Flip TTA", "0.8597", "0.9464", "0.8967", "0.8183", "0.8077"],
        ["  + Flip TTA + Val Thr", "0.8795", "0.9290", "0.8992", "0.8222", "0.8155"],
        ["Proposed (Flip+Zoom)", "0.8600", "0.9336", "0.8918", "0.8089", "0.7916"],
        ["  + Flip TTA + Val Thr", "0.8822", "0.9223", "0.8980", "0.8192", "0.8102"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = c_black
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(7.5)
        p.font.color.rgb = c_white
        p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    for r_idx, row_data in enumerate(data):
        for c_idx, val in enumerate(row_data):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx in [2, 4]:
                cell.fill.fore_color.rgb = RGBColor(235, 235, 235) # accent gray
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = c_zebra_gray
            else:
                cell.fill.fore_color.rgb = c_white
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(7)
            if r_idx in [0, 2] and c_idx in [4, 5]: # Best scores bold
                p.font.bold = True
                p.font.color.rgb = c_black
            else:
                p.font.color.rgb = c_black
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    # Statistical significance text below table
    tb_stats = slide.shapes.add_textbox(Inches(col3_left + 0.08), Inches(3.08), Inches(col3_w - 0.16), Inches(0.80))
    tf_st = tb_stats.text_frame
    tf_st.word_wrap = True
    tf_st.margin_top = tf_st.margin_bottom = tf_st.margin_left = tf_st.margin_right = 0
    p = tf_st.paragraphs[0]
    p.text = "Statistical Significance & Judging Metrics:"
    p.font.bold = True
    p.font.size = Pt(8.5)
    p.font.color.rgb = c_black

    p_st = tf_st.add_paragraph()
    p_st.text = "• Mean Difference: ΔIoU = -0.0021  (Baseline won 15/26 test scenes)\n• 95% Bootstrap CI: [-0.0125, +0.0099] (Straddles zero)\n• Wilcoxon Signed-Rank Test: p = 0.247 > 0.05 → Fail to Reject H₀\n• Overall Pixel Accuracy: 96.2% | Inference Speed: 32 FPS (AMP FP16)"
    p_st.font.size = Pt(7.8)
    p_st.font.color.rgb = c_black
    p_st.space_after = Pt(1)

    # 3.2 Qualitative Delineation
    add_card(col3_left, 3.96, col3_w, 1.45, "Qualitative Test Visualizations")
    qual_img = "report/qualitative_bw.png"
    if not os.path.exists(qual_img):
        qual_img = "results/figures/qualitative.png"
    if os.path.exists(qual_img):
        slide.shapes.add_picture(qual_img, Inches(col3_left + 0.10), Inches(4.24), width=Inches(col3_w - 0.20), height=Inches(0.98))
    tb_q_cap = slide.shapes.add_textbox(Inches(col3_left + 0.08), Inches(5.24), Inches(col3_w - 0.16), Inches(0.18))
    tf_qc = tb_q_cap.text_frame
    tf_qc.margin_top = tf_qc.margin_bottom = tf_qc.margin_left = tf_qc.margin_right = 0
    p = tf_qc.paragraphs[0]
    p.text = "High-contrast test predictions: Ground-truth contours vs. predicted segmentation masks"
    p.font.size = Pt(7)
    p.font.color.rgb = c_mid_gray
    p.alignment = PP_ALIGN.CENTER

    # 3.3 Conclusion Card
    add_card(col3_left, 5.46, col3_w, 1.18, "Conclusions & Practical Takeaways")
    tb_conc = slide.shapes.add_textbox(Inches(col3_left + 0.10), Inches(5.72), Inches(col3_w - 0.20), Inches(0.88))
    tf_conc = tb_conc.text_frame
    tf_conc.word_wrap = True
    tf_conc.margin_top = tf_conc.margin_bottom = tf_conc.margin_left = tf_conc.margin_right = 0
    conc_points = [
        "1. Heuristic Augmentation Refuted: Stochastic geometric transforms add zero statistically significant benefit over unaugmented ImageNet transfer learning on small datasets.",
        "2. Full-Res Bypass Value: Auxiliary skip connection ('up0') preserves fine contours, achieving 79.85% Boundary-F1.",
        "3. Calibration Superiority: Flip TTA + validation threshold calibration elevates test IoU from 81.10% to 82.22% (Bound-F1: 81.55%)."
    ]
    for i, cp in enumerate(conc_points):
        p = tf_conc.paragraphs[0] if i == 0 else tf_conc.add_paragraph()
        p.text = cp
        p.font.size = Pt(7.5)
        p.font.color.rgb = c_black
        p.space_after = Pt(1)

    # =========================================================================
    # FOOTER BAR (Width = 13.173 in, Top = 6.70 in, Height = 0.72 in)
    # =========================================================================
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.08), Inches(6.70), Inches(13.173), Inches(0.72))
    footer.fill.solid()
    footer.fill.fore_color.rgb = c_black
    footer.line.color.rgb = c_black
    footer.line.width = Pt(1)

    tb_foot = slide.shapes.add_textbox(Inches(0.18), Inches(6.72), Inches(12.973), Inches(0.68))
    tf_f = tb_foot.text_frame
    tf_f.word_wrap = True
    tf_f.margin_top = tf_f.margin_bottom = tf_f.margin_left = tf_f.margin_right = 0

    p_f0 = tf_f.paragraphs[0]
    p_f0.text = "Key Literature References & Academic Module Acknowledgments:"
    p_f0.font.bold = True
    p_f0.font.size = Pt(8.5)
    p_f0.font.color.rgb = c_white

    p_f1 = tf_f.add_paragraph()
    p_f1.text = "[1] Ronneberger et al., 'U-Net: Convolutional Networks for Biomedical Image Segmentation,' MICCAI 2015.   [2] He et al., 'Deep Residual Learning for Image Recognition,' IEEE CVPR 2016.\n[3] Wang et al., 'Object Detection Combining Recognition and Segmentation,' ACCV 2007.   [4] Milletari et al., 'V-Net,' IEEE 3DV 2016.   [5] Shorten & Khoshgoftaar, 'A Survey on Image Data Augmentation,' J. Big Data 2019."
    p_f1.font.size = Pt(7.5)
    p_f1.font.color.rgb = RGBColor(220, 220, 220)

    out_pptx = "report/project_poster.pptx"
    prs.save(out_pptx)
    print(f"Successfully generated black-and-white {out_pptx} ({os.path.getsize(out_pptx)} bytes)")

if __name__ == "__main__":
    create_bw_poster()
