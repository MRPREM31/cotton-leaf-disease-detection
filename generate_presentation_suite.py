import os
import sys
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ----------------------------------------------------------------------
# 1. REPORTLAB PDF SLIDE DECK GENERATION (LANDSCAPE 16:9 / LETTER)
# ----------------------------------------------------------------------

class NumberedCanvas(canvas.Canvas):
    """Custom 2-pass canvas to draw sleek background, header, footer & page numbers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        width, height = self._pagesize
        
        # Cover slide (Page 1) styling
        if self._pageNumber == 1:
            # Dark modern background
            self.setFillColor(colors.HexColor("#0d1b2a"))
            self.rect(0, 0, width, height, fill=True, stroke=False)
            
            # Accent gradient bars / geometric shapes
            self.setFillColor(colors.HexColor("#1b4332"))
            self.rect(0, height - 12, width, 12, fill=True, stroke=False)
            self.setFillColor(colors.HexColor("#2d6a4f"))
            self.rect(0, 0, width, 10, fill=True, stroke=False)
            self.setFillColor(colors.HexColor("#52b788"))
            self.rect(40, height - 160, 6, 100, fill=True, stroke=False)
            return

        # Content slides background
        self.setFillColor(colors.HexColor("#f8fafc"))
        self.rect(0, 0, width, height, fill=True, stroke=False)
        
        # Header bar
        self.setFillColor(colors.HexColor("#0f2027"))
        self.rect(0, height - 42, width, 42, fill=True, stroke=False)
        
        # Emerald accent line under header
        self.setFillColor(colors.HexColor("#10b981"))
        self.rect(0, height - 44, width, 2, fill=True, stroke=False)

        # Header Title Text
        self.setFont("Helvetica-Bold", 11)
        self.setFillColor(colors.white)
        self.drawString(36, height - 27, "COTTONGUARD AI : PROJECT PRESENTATION")

        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#a7f3d0"))
        self.drawRightString(width - 36, height - 27, "Deep Learning & Edge Diagnostics")

        # Footer Bar
        self.setFillColor(colors.HexColor("#e2e8f0"))
        self.rect(0, 0, width, 26, fill=True, stroke=False)
        self.setFillColor(colors.HexColor("#10b981"))
        self.rect(0, 24, width, 1.5, fill=True, stroke=False)

        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(36, 9, "CottonGuard AI — Explainable Offline Diagnostic System for Cotton Foliar Pathology")
        
        page_str = f"Slide {self._pageNumber} of {page_count}"
        self.drawRightString(width - 36, 9, page_str)


def build_pdf_presentation(output_pdf_path):
    # Landscape Letter: 11 x 8.5 inches (792 x 612 points)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_cover_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#ffffff")
    )
    
    sub_cover_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#a7f3d0")
    )

    meta_cover_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#cbd5e1")
    )

    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )

    slide_subtitle_style = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#059669"),
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    body_bold = ParagraphStyle(
        'SlideBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#0f172a")
    )

    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#334155")
    )

    card_header_style = ParagraphStyle(
        'CardHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#065f46")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1, # Center
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # =========================================================================
    # SLIDE 1: TITLE / COVER SLIDE
    # =========================================================================
    story.append(Spacer(1, 40))
    cover_table_data = [
        [
            Paragraph("<b>COTTONGUARD AI</b>", title_cover_style),
        ],
        [
            Paragraph("An Explainable, Robust & Lightweight Deep Learning Framework for Offline Cotton Leaf Disease Diagnosis on Resource-Constrained Edge Devices", sub_cover_style)
        ],
        [
            Spacer(1, 15)
        ],
        [
            Paragraph(
                "<b>Key Highlights:</b> 98.60% Test Macro F1 &bull; 4.87M Params &bull; 8.34ms ONNX CPU Inference &bull; Real-time Grad-CAM &bull; 100% Offline Android",
                ParagraphStyle('CoverBadge', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor("#34d399"))
            )
        ],
        [
            Spacer(1, 40)
        ],
        [
            Table(
                [
                    [
                        Paragraph("<b>Authors / Presentation:</b> Research & Engineering Team", meta_cover_style),
                        Paragraph("<b>Target Domain:</b> Precision Agriculture & Edge AI", meta_cover_style),
                    ],
                    [
                        Paragraph("<b>Dataset:</b> SAR-CLD-2024 (Kaggle Verified)", meta_cover_style),
                        Paragraph("<b>Publication Scope:</b> Final-Year Thesis & Academic Journal", meta_cover_style),
                    ]
                ],
                colWidths=[360, 360],
                style=[('VALIGN', (0,0), (-1,-1), 'TOP')]
            )
        ]
    ]
    cover_table = Table(cover_table_data, colWidths=[720])
    cover_table.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: PROJECT OVERVIEW, PROBLEM STATEMENT & OBJECTIVES
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Problem Formulation", slide_title_style))
    story.append(Paragraph("BRIDGING THE GAP BETWEEN ADVANCED DEEP LEARNING AND REAL-WORLD FIELD CONDITIONS", slide_subtitle_style))

    p1_card1 = [
        Paragraph("<b>Agronomic & Economic Context</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Global Impact:</b> Cotton (<i>Gossypium hirsutum</i>) is a cornerstone cash crop supporting over 100 million farming families globally.<br/>&bull; <b>Pathology Threat:</b> Foliar infections (Bacterial Blight, Leaf Curl Virus, Jassids) cause <b>30% to 40% yield loss</b> if undiagnosed.<br/>&bull; <b>Diagnostic Void:</b> Agricultural extension officers are scarce; farmers rely on trial-and-error chemical spraying causing ecological & economic damage.", bullet_style)
    ]

    p1_card2 = [
        Paragraph("<b>Limitations of Existing AI Systems</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Cloud Connectivity Dependency:</b> Existing mobile apps require 4G/5G, failing in remote rural belts with zero bandwidth.<br/>&bull; <b>Closed-Set Catastrophic Overconfidence:</b> Standard CNNs assign 99% confidence to random weeds, fingers, or blurry non-leaf frames.<br/>&bull; <b>Black-Box Predictions:</b> Lack of visual localization (Grad-CAM) erodes farmer and agronomist trust.<br/>&bull; <b>Heavy Compute:</b> Standard architectures (ResNet-50) are too resource-heavy for budget smartphones.", bullet_style)
    ]

    p1_card3 = [
        Paragraph("<b>Core Objectives of CottonGuard AI</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>1. Lightweight Architecture:</b> Design a custom < 5M parameter neural network (<b>CottonNet</b>) achieving > 98% Macro F1.<br/>&bull; <b>2. Two-Stage Gatekeeper:</b> Filter out-of-distribution (OOD) images before classification to prevent false diagnosis.<br/>&bull; <b>3. Edge Optimization:</b> Quantize via ONNX to achieve sub-10ms CPU inference on edge devices.<br/>&bull; <b>4. Native Mobile Suite:</b> Build a 100% offline, trilingual (English, Hindi, Odia) Android deployment.", bullet_style)
    ]

    overview_table = Table([[p1_card1, p1_card2, p1_card3]], colWidths=[235, 245, 240])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ffffff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(overview_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: DATASET PROVENANCE & THE 7 PATHOLOGY CLASSES
    # =========================================================================
    story.append(Paragraph("2. Dataset Provenance & Target Clinical Taxonomy", slide_title_style))
    story.append(Paragraph("SAR-CLD-2024 BENCHMARK DATASET WITH 7 CLINICAL & ABIOTIC CLASSES", slide_subtitle_style))

    dataset_summary = Paragraph(
        "<b>Dataset Provenance:</b> Sourced from <b>SAR-CLD-2024</b> (Cotton Leaf Disease Benchmark dataset on Kaggle / field research collections). "
        "Contains <b>2,137 curated, high-resolution RGB field images</b> captured across varying agricultural lighting conditions. "
        "Standardized to 224&times;224 resolution with ImageNet normalization constants (&mu;=[0.485, 0.456, 0.406], &sigma;=[0.229, 0.224, 0.225]).",
        body_style
    )
    story.append(dataset_summary)
    story.append(Spacer(1, 8))

    tax_header = [
        Paragraph("<b>Class Index</b>", table_header_style),
        Paragraph("<b>Disease / Condition Name</b>", table_header_style),
        Paragraph("<b>Causative Agent / Etiology</b>", table_header_style),
        Paragraph("<b>Distinct Visual Symptomatology</b>", table_header_style),
        Paragraph("<b>Recommended Advisory</b>", table_header_style)
    ]
    tax_rows = [
        [
            Paragraph("<b>0</b>", table_cell_bold),
            Paragraph("<b>Bacterial Blight</b>", table_cell_bold),
            Paragraph("<i>Xanthomonas malvacearum</i>", table_cell_style),
            Paragraph("Angular water-soaked spots bounded by leaf veinlets, turning black-brown ('black arm').", table_cell_style),
            Paragraph("Copper Oxychloride (0.25%) + Streptocycline (100 ppm) spray.", table_cell_style),
        ],
        [
            Paragraph("<b>1</b>", table_cell_bold),
            Paragraph("<b>Curl Virus (CLCuV)</b>", table_cell_bold),
            Paragraph("Begomovirus / Whitefly vector", table_cell_style),
            Paragraph("Upward/downward leaf curling, vein thickening, leaf enations, stunted growth.", table_cell_style),
            Paragraph("Vector control: Imidacloprid 17.8 SL; rogue infected saplings.", table_cell_style),
        ],
        [
            Paragraph("<b>2</b>", table_cell_bold),
            Paragraph("<b>Healthy Leaf</b>", table_cell_bold),
            Paragraph("Normal physiology (Control)", table_cell_style),
            Paragraph("Deep green lamina, smooth surface, intact margins, zero necrotic lesions.", table_cell_style),
            Paragraph("Standard balanced NPK nutrition & scheduled irrigation.", table_cell_style),
        ],
        [
            Paragraph("<b>3</b>", table_cell_bold),
            Paragraph("<b>Herbicide Damage</b>", table_cell_bold),
            Paragraph("Abiotic chemical drift (2,4-D / Glyphosate)", table_cell_style),
            Paragraph("Strap-like leaf narrowing, finger-like crinkling, petiole twisting, interveinal chlorosis.", table_cell_style),
            Paragraph("Immediate flushing with freshwater; foliar zinc + urea (1%) spray.", table_cell_style),
        ],
        [
            Paragraph("<b>4</b>", table_cell_bold),
            Paragraph("<b>Leaf Hopper Jassids</b>", table_cell_bold),
            Paragraph("<i>Amrasca biguttula biguttula</i>", table_cell_style),
            Paragraph("Yellowing of leaf margins ('hopperburn'), downward curling, bronze necrotic edges.", table_cell_style),
            Paragraph("Flonicamid 50 WG (0.3g/L) or Thiamethoxam 25 WG.", table_cell_style),
        ],
        [
            Paragraph("<b>5</b>", table_cell_bold),
            Paragraph("<b>Leaf Redding</b>", table_cell_bold),
            Paragraph("Physiological Mg deficiency / cold stress", table_cell_style),
            Paragraph("Red-purple anthocyanin pigmentation between green main veins in mature leaves.", table_cell_style),
            Paragraph("Foliar application of Magnesium Sulphate (MgSO4 1.0%) + Urea (1.0%).", table_cell_style),
        ],
        [
            Paragraph("<b>6</b>", table_cell_bold),
            Paragraph("<b>Leaf Variegation</b>", table_cell_bold),
            Paragraph("Micronutrient deficiency / genetic chimera", table_cell_style),
            Paragraph("Irregular yellowish-white mosaic patches interspersed with normal green tissue.", table_cell_style),
            Paragraph("Chelated multi-micronutrient foliar spray (Fe, Zn, Mn).", table_cell_style),
        ]
    ]

    tax_table = Table([tax_header] + tax_rows, colWidths=[45, 120, 140, 235, 180])
    tax_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#065f46")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f1f5f9")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(tax_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: PROPOSED COTTONNET ARCHITECTURE & DESIGN INNOVATIONS
    # =========================================================================
    story.append(Paragraph("3. Proposed Neural Network Architecture: CottonNet", slide_title_style))
    story.append(Paragraph("CUSTOM EFFICIENTNET-B0 BACKBONE + SQUEEZE-AND-EXCITATION ATTENTION + DUAL SPATIAL POOLING", slide_subtitle_style))

    arch_left = [
        Paragraph("<b>Architectural Pipeline & Flow</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Input Layer:</b> RGB Image Tensor [Batch, 3, 224, 224].<br/>"
                  "&bull; <b>Feature Extractor:</b> EfficientNet-B0 (Blocks 0&ndash;6) extracting hierarchical spatial-semantic patterns &rarr; [B, 1280, 7, 7].<br/>"
                  "&bull; <b>Channel Attention (SE-Block):</b> AdaptiveAvgPool2d &rarr; Linear(1280&rarr;80, ReLU) &rarr; Linear(80&rarr;1280, Sigmoid). Dynamic channel recalibration with reduction ratio <i>r=16</i>.<br/>"
                  "&bull; <b>Dual Spatial Pooling:</b> Parallel branches of <i>AdaptiveAvgPool2d(1)</i> (background context) and <i>AdaptiveMaxPool2d(1)</i> (peak lesion feature intensity), concatenated to <b>2560 dimensions</b>.<br/>"
                  "&bull; <b>Regularized Classifier Head:</b> Dropout(0.40) &rarr; Linear(2560&rarr;256) &rarr; BatchNorm1d(256) &rarr; ReLU &rarr; Dropout(0.30) &rarr; Linear(256&rarr;7).", bullet_style)
    ]

    arch_right = [
        Paragraph("<b>Why CottonNet Outperforms Standard Baselines</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Compact Parameter Footprint:</b> Only <b>4,870,275 parameters</b> (~4.87M total, 4.56M trainable) vs 23.52M for ResNet-50.<br/>"
                  "&bull; <b>Attention-Gated Lesion Focus:</b> SE-Block suppresses irrelevant background soil and sunlight glare while boosting micro-lesion signals.<br/>"
                  "&bull; <b>Dual Pooling Advantage:</b> Average pooling alone washes out small speckle lesions; Max pooling alone is noise-sensitive. Combining both gives maximum discriminative power.<br/>"
                  "&bull; <b>Overfitting Prevention:</b> Dual dropout stages (0.4 + 0.3) and batch normalization ensure generalizability even with limited dataset sizes.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>Knowledge Distillation (KD) Protocol:</b><br/>"
                  "Trained using teacher-student soft distillation from a pre-trained <b>DenseNet-121</b> teacher:<br/>"
                  "&bull; Temperature: <i>T = 3.0</i> &bull; Loss Balance: <i>&alpha; = 0.5</i><br/>"
                  "&bull; <i>L<sub>total</sub> = &alpha; &middot; L<sub>KD</sub>(T) + (1-&alpha;) &middot; L<sub>CE</sub></i><br/>"
                  "Transfers subtle inter-class structural correlations to the lightweight student.", ParagraphStyle('KDStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor("#0f766e")))
    ]

    arch_table = Table([[arch_left, arch_right]], colWidths=[355, 365])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ffffff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(arch_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: COMPREHENSIVE 5-MODEL BENCHMARK COMPARISON
    # =========================================================================
    story.append(Paragraph("4. Experimental Benchmark: 5 Architecture Comparison", slide_title_style))
    story.append(Paragraph("EMPIRICAL RIGOR: PROPOSED COTTONNET VS POPULAR STATE-OF-THE-ART BASELINES", slide_subtitle_style))

    bench_header = [
        Paragraph("<b>Model Architecture</b>", table_header_style),
        Paragraph("<b>Test Accuracy</b>", table_header_style),
        Paragraph("<b>Test Macro F1</b>", table_header_style),
        Paragraph("<b>Total Parameters</b>", table_header_style),
        Paragraph("<b>Model Disk Size</b>", table_header_style),
        Paragraph("<b>Architectural Role / Provenance</b>", table_header_style),
        Paragraph("<b>Evaluation Verdict</b>", table_header_style)
    ]
    bench_rows = [
        [
            Paragraph("<b>CottonNet (Proposed)</b>", table_cell_bold),
            Paragraph("<b>98.60%</b>", table_cell_bold),
            Paragraph("<b>98.60%</b>", table_cell_bold),
            Paragraph("<b>4,870,275 (~4.87M)</b>", table_cell_bold),
            Paragraph("<b>18.88 MB</b>", table_cell_bold),
            Paragraph("Custom EfficientNet-B0 + SE + DualPool (KD Distilled)", table_cell_style),
            Paragraph("<b>CHAMPION (Best Overall)</b>", ParagraphStyle('Champ', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, textColor=colors.HexColor("#047857"), alignment=1)),
        ],
        [
            Paragraph("DenseNet-121", table_cell_style),
            Paragraph("96.26%", table_cell_bold),
            Paragraph("96.26%", table_cell_bold),
            Paragraph("6,960,000 (~6.96M)", table_cell_style),
            Paragraph("28.47 MB", table_cell_style),
            Paragraph("Dense Connectivity Teacher Network", table_cell_style),
            Paragraph("Effective Teacher / Heavier", table_cell_style),
        ],
        [
            Paragraph("MobileNetV3-Large", table_cell_style),
            Paragraph("95.79%", table_cell_bold),
            Paragraph("95.79%", table_cell_bold),
            Paragraph("4,210,000 (~4.21M)", table_cell_style),
            Paragraph("17.06 MB", table_cell_style),
            Paragraph("Standard Mobile Edge Baseline", table_cell_style),
            Paragraph("Fast but -2.81% F1 vs CottonNet", table_cell_style),
        ],
        [
            Paragraph("EfficientNet-B0 (Base)", table_cell_style),
            Paragraph("95.33%", table_cell_bold),
            Paragraph("95.33%", table_cell_bold),
            Paragraph("4,020,000 (~4.02M)", table_cell_style),
            Paragraph("16.37 MB", table_cell_style),
            Paragraph("Compound Scaling Baseline (No SE / DualPool)", table_cell_style),
            Paragraph("-3.27% F1 without custom head", table_cell_style),
        ],
        [
            Paragraph("ResNet-50", table_cell_style),
            Paragraph("93.93%", table_cell_bold),
            Paragraph("93.93%", table_cell_bold),
            Paragraph("23,520,000 (~23.52M)", table_cell_style),
            Paragraph("94.41 MB", table_cell_style),
            Paragraph("Deep Residual Heavy Baseline", table_cell_style),
            Paragraph("Lowest F1 & 5x heavier footprint", table_cell_style),
        ]
    ]

    bench_table = Table([bench_header] + bench_rows, colWidths=[125, 65, 65, 95, 75, 175, 120])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2027")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ecfdf5"), colors.HexColor("#ffffff"), colors.HexColor("#ffffff"), colors.HexColor("#ffffff"), colors.HexColor("#ffffff")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 10))

    # Analytical takeaways
    takeaways = Paragraph(
        "<b>Key Empirical Insights:</b><br/>"
        "1. <b>Outperforming the Teacher:</b> CottonNet (98.60% F1) surpasses its DenseNet-121 teacher (96.26% F1) by <b>+2.34%</b>, proving the synergistic benefit of SE channel attention + dual pooling combined with KD.<br/>"
        "2. <b>Ablation Validation:</b> Comparing Base EfficientNet-B0 (95.33%) against CottonNet (98.60%) confirms that the proposed architectural enhancements directly deliver a <b>+3.27% accuracy boost</b>.<br/>"
        "3. <b>Efficiency Superiority:</b> CottonNet achieves the highest accuracy while maintaining a <b>4.8x smaller size</b> than ResNet-50.",
        body_style
    )
    story.append(takeaways)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: EDGE OPTIMIZATION, ONNX RUNTIME & LATENCY BENCHMARK
    # =========================================================================
    story.append(Paragraph("5. Edge Optimization & Hardware Latency Profiling", slide_title_style))
    story.append(Paragraph("PRECISION ANALYSIS: PYTORCH FP32 VS ONNX FP32 VS FP16 VS DYNAMIC INT8", slide_subtitle_style))

    edge_header = [
        Paragraph("<b>Model Runtime Variant</b>", table_header_style),
        Paragraph("<b>Precision</b>", table_header_style),
        Paragraph("<b>Disk Footprint</b>", table_header_style),
        Paragraph("<b>Compression Ratio</b>", table_header_style),
        Paragraph("<b>Logit Agreement vs PyTorch</b>", table_header_style),
        Paragraph("<b>Mean Latency (CPU)</b>", table_header_style),
        Paragraph("<b>Median (P50)</b>", table_header_style),
        Paragraph("<b>95th %ile (P95)</b>", table_header_style),
        Paragraph("<b>Target Deployment Platform</b>", table_header_style)
    ]
    edge_rows = [
        [
            Paragraph("PyTorch Native", table_cell_style),
            Paragraph("FP32", table_cell_bold),
            Paragraph("18.88 MB", table_cell_style),
            Paragraph("1.00x", table_cell_style),
            Paragraph("100.00% (Ref)", table_cell_bold),
            Paragraph("51.44 ms", table_cell_style),
            Paragraph("46.56 ms", table_cell_style),
            Paragraph("83.70 ms", table_cell_style),
            Paragraph("Python Backend / Workstation", table_cell_style),
        ],
        [
            Paragraph("<b>ONNX Runtime (FP32)</b>", table_cell_bold),
            Paragraph("<b>FP32</b>", table_cell_bold),
            Paragraph("<b>18.56 MB</b>", table_cell_bold),
            Paragraph("<b>1.02x</b>", table_cell_bold),
            Paragraph("<b>100.00%</b>", table_cell_bold),
            Paragraph("<b>8.34 ms</b>", table_cell_bold),
            Paragraph("<b>6.13 ms</b>", table_cell_bold),
            Paragraph("<b>19.15 ms</b>", table_cell_bold),
            Paragraph("<b>CHAMPION (Primary Edge CPU)</b>", ParagraphStyle('EdgC', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7, textColor=colors.HexColor("#047857"), alignment=1)),
        ],
        [
            Paragraph("<b>ONNX Runtime (FP16)</b>", table_cell_bold),
            Paragraph("<b>FP16</b>", table_cell_bold),
            Paragraph("<b>9.33 MB</b>", table_cell_bold),
            Paragraph("<b>2.02x</b>", table_cell_bold),
            Paragraph("<b>100.00%</b>", table_cell_bold),
            Paragraph("14.12 ms", table_cell_style),
            Paragraph("12.80 ms", table_cell_style),
            Paragraph("22.40 ms", table_cell_style),
            Paragraph("<b>Mobile GPU / NPU Acceleration</b>", table_cell_style),
        ],
        [
            Paragraph("ONNX Dynamic INT8", table_cell_style),
            Paragraph("INT8", table_cell_bold),
            Paragraph("4.94 MB", table_cell_style),
            Paragraph("3.75x", table_cell_bold),
            Paragraph("0.00% (Drift)", ParagraphStyle('Drift', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7, textColor=colors.HexColor("#b91c1c"), alignment=1)),
            Paragraph("37.87 ms", table_cell_style),
            Paragraph("34.39 ms", table_cell_style),
            Paragraph("57.69 ms", table_cell_style),
            Paragraph("Requires Static PTQ Calibration", table_cell_style),
        ]
    ]

    edge_table = Table([edge_header] + edge_rows, colWidths=[105, 45, 65, 65, 85, 75, 65, 65, 150])
    edge_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#ecfdf5"), colors.HexColor("#f8fafc"), colors.HexColor("#ffffff")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(edge_table)
    story.append(Spacer(1, 8))

    edge_findings = Paragraph(
        "<b>Edge Deployment Decision Rationale:</b><br/>"
        "&bull; <b>6.16x Speedup:</b> Converting from PyTorch FP32 (51.44 ms) to ONNX FP32 (8.34 ms) slashes CPU latency by over 83% while preserving <b>100.00% mathematical fidelity</b> (max logit delta < 0.000009).<br/>"
        "&bull; <b>Half-Precision Footprint:</b> ONNX FP16 halves model size to <b>9.33 MB</b> with zero loss in prediction agreement, making it optimal for mobile GPU execution (Adreno / Mali).<br/>"
        "&bull; <b>Rigorous Quantization Audit:</b> Dynamic INT8 achieved 4.94 MB but exhibited logit drift on subtle lesion boundaries without static calibration datasets. Hence, <b>ONNX FP32 / FP16</b> are designated as primary deployment targets.",
        body_style
    )
    story.append(edge_findings)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: TWO-STAGE PIPELINE & OOD GATEKEEPER
    # =========================================================================
    story.append(Paragraph("6. System Architecture & Two-Stage Safety Gatekeeper", slide_title_style))
    story.append(Paragraph("PREVENTING CATASTROPHIC MISDIAGNOSES ON OUT-OF-DISTRIBUTION (OOD) & DEGRADED FRAMES", slide_subtitle_style))

    p6_left = [
        Paragraph("<b>Stage-1: Pre-Inference Gatekeeper</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Laplacian Blur Guard:</b> Computes variance of Laplacian operator. Rejects blurry frames (Var < 80.0) preventing blurry mispredictions.<br/>"
                  "&bull; <b>Luminance Exposure Guard:</b> Analyzes grayscale channel mean. Rejects underexposed (< 40) and washed-out glare (> 215) images.<br/>"
                  "&bull; <b>HSV Foliage Manifold Check:</b> Enforces minimum green/foliage hue mask area (&ge; 15%). Rejects hands, cloth, soil, walls, and human skin.<br/>"
                  "&bull; <b>Empirical Validation:</b> <b>100% True Acceptance Rate</b> on cotton leaves; <b>80.0% True Rejection Rate</b> on OOD distractors.", bullet_style)
    ]

    p6_right = [
        Paragraph("<b>Stage-2: CottonNet Inference & Decision Head</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Standardized Preprocessing:</b> Bilinear resize to 224&times;224, float32 scaling, ImageNet channel normalization.<br/>"
                  "&bull; <b>Softmax Probability Distribution:</b> Computes calibrated confidence for all 7 disease categories.<br/>"
                  "&bull; <b>Operational Confidence Threshold:</b> Calibrated at <b>&tau; = 0.40</b>. Low-confidence predictions flag 'Indeterminate / Retake Photo'.<br/>"
                  "&bull; <b>Zero Severity Fabrication:</b> Purged arbitrary confidence-to-severity mapping, adhering to strict scientific audit standards.", bullet_style)
    ]

    p6_table = Table([[p6_left, p6_right]], colWidths=[355, 365])
    p6_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ffffff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(p6_table)
    story.append(Spacer(1, 10))

    # OOD Audit Table
    ood_table_data = [
        [
            Paragraph("<b>Evaluation Subset</b>", table_header_style),
            Paragraph("<b>Sample Count</b>", table_header_style),
            Paragraph("<b>Gatekeeper Accepted</b>", table_header_style),
            Paragraph("<b>Gatekeeper Rejected</b>", table_header_style),
            Paragraph("<b>Empirical Performance Metric</b>", table_header_style),
        ],
        [
            Paragraph("<b>Cotton Leaf Specimens (In-Distribution)</b>", table_cell_bold),
            Paragraph("10", table_cell_bold),
            Paragraph("10", table_cell_bold),
            Paragraph("0", table_cell_bold),
            Paragraph("<b>100.0% True Acceptance Rate (0.0% False Rejection)</b>", ParagraphStyle('OODP', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, textColor=colors.HexColor("#047857"))),
        ],
        [
            Paragraph("<b>Non-Cotton OOD / Noise / Abiotic Distractors</b>", table_cell_bold),
            Paragraph("10", table_cell_bold),
            Paragraph("2 (Non-cotton weeds)", table_cell_style),
            Paragraph("8", table_cell_bold),
            Paragraph("<b>80.0% True Rejection Rate (100% abiotic/skin rejection)</b>", ParagraphStyle('OODR', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, textColor=colors.HexColor("#0284c7"))),
        ]
    ]
    ood_table = Table(ood_table_data, colWidths=[230, 80, 130, 110, 170])
    ood_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#065f46")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f1f5f9")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(ood_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: EXPLAINABLE AI (GRAD-CAM) & ENVIRONMENTAL ROBUSTNESS
    # =========================================================================
    story.append(Paragraph("7. Explainable AI & Environmental Robustness", slide_title_style))
    story.append(Paragraph("GRAD-CAM LESION ATTRIBUTION & EMPIRICAL STRESS TESTING ACROSS 11 PERTURBATIONS", slide_subtitle_style))

    p7_left = [
        Paragraph("<b>Real-Time Grad-CAM Attribution</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Layer Hook:</b> Dynamic autograd computed directly on final convolutional block <code>backbone.blocks[-1]</code>.<br/>"
                  "&bull; <b>Mathematical Formulation:</b><br/>"
                  "&nbsp;&nbsp;<i>&alpha;<sub>k</sub><sup>c</sup> = (1/Z) &sum;<sub>i</sub> &sum;<sub>j</sub> (&part; y<sup>c</sup> / &part; A<sub>i,j</sub><sup>k</sup>)</i><br/>"
                  "&nbsp;&nbsp;<i>L<sub>Grad-CAM</sub><sup>c</sup> = ReLU(&sum;<sub>k</sub> &alpha;<sub>k</sub><sup>c</sup> A<sup>k</sup>)</i><br/>"
                  "&bull; <b>Biological Lesion Alignment:</b> Verified to isolate vein thickening in Curl Virus, margin chlorosis in Jassids, and necrotic speckles in Blight.<br/>"
                  "&bull; <b>Farmer Interpretability:</b> Overlaid Jet heatmaps provide visual confirmation of why a diagnosis was made.", bullet_style)
    ]

    p7_right = [
        Paragraph("<b>11-Regime Robustness Evaluation</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Clean Image Baseline:</b> Mean confidence = 43.12% across multi-class distributions.<br/>"
                  "&bull; <b>Gaussian Blur (r=1):</b> Resilience = <b>High</b> (&Delta;Conf = -1.32%, 0% rejection).<br/>"
                  "&bull; <b>Gaussian Blur (r=3):</b> Safety guard triggered (58.3% rejected by blur filter).<br/>"
                  "&bull; <b>Sensor Noise (&sigma;=15):</b> Resilience = <b>High</b> (&Delta;Conf = -2.62%).<br/>"
                  "&bull; <b>Sensor Noise (&sigma;=30):</b> Noise rejection guard triggered (75% rejected).<br/>"
                  "&bull; <b>Low Light (0.7x) & Glare (1.6x):</b> Maintained accurate classification with minimal confidence drift.<br/>"
                  "&bull; <b>JPEG Compression (Q=20):</b> High resilience (&Delta;Conf = -1.02%).<br/>"
                  "&bull; <b>Spatial Rotation (45&deg; / 90&deg;):</b> Robust invariant representations (&Delta;Conf < 1.9%).", bullet_style)
    ]

    p7_table = Table([[p7_left, p7_right]], colWidths=[355, 365])
    p7_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ffffff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(p7_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 9: ANDROID DEPLOYMENT, ARCHITECTURE & FARMER UX
    # =========================================================================
    story.append(Paragraph("8. Native Android Mobile Architecture & Farmer UX", slide_title_style))
    story.append(Paragraph("ZERO CLOUD DEPENDENCY: ON-DEVICE INFERENCE ENGINE POWERED BY ONNX RUNTIME MOBILE", slide_subtitle_style))

    p8_col1 = [
        Paragraph("<b>Camera & Edge Pipeline</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Android CameraX:</b> High-framerate real-time camera viewfinder with tap-to-focus and auto-exposure.<br/>"
                  "&bull; <b>On-Device ONNX Runtime:</b> Executes INT8/FP32 neural graph directly on Android CPU threads via NDK.<br/>"
                  "&bull; <b>Zero Network Latency:</b> Operates instantaneously without cellular signals or cloud backend servers.<br/>"
                  "&bull; <b>Battery Efficient:</b> Sub-10ms execution ensures minimal CPU wake-lock time.", bullet_style)
    ]

    p8_col2 = [
        Paragraph("<b>Multilingual Advisory</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Trilingual Localization:</b> Complete interface & agronomic treatment support in <b>English, Hindi (हिन्दी), and Odia (ଓଡ଼ିଆ)</b>.<br/>"
                  "&bull; <b>Actionable Treatment:</b> Precise chemical dosages (g/L), spray intervals, and preventive measures.<br/>"
                  "&bull; <b>Voice-Ready Text:</b> Formatted for Android Text-to-Speech (TTS) readout for semi-literate farmers.<br/>"
                  "&bull; <b>Farmer Privacy:</b> Zero image uploads; all data stays on device.", bullet_style)
    ]

    p8_col3 = [
        Paragraph("<b>Local Data Persistence</b>", card_header_style),
        Spacer(1, 4),
        Paragraph("&bull; <b>Room SQLite Database:</b> Logs historical scans, GPS geotags (optional), diagnostic confidence, and treatment history.<br/>"
                  "&bull; <b>Audit Trail:</b> Allows farmers and agronomists to track crop recovery trajectories over multiple weeks.<br/>"
                  "&bull; <b>Exportable Reports:</b> Generates offline scan summaries for agricultural extension officers.", bullet_style)
    ]

    p8_table = Table([[p8_col1, p8_col2, p8_col3]], colWidths=[235, 245, 240])
    p8_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ffffff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(p8_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 10: SCIENTIFIC AUDIT, CITATIONS & FUTURE ROADMAP
    # =========================================================================
    story.append(Paragraph("9. Scientific Audit, References & Future Roadmap", slide_title_style))
    story.append(Paragraph("RESEARCH AUDIT CHECKLIST, ACADEMIC CITATIONS & EXTENSION DIRECTIONS", slide_subtitle_style))

    p9_left = [
        Paragraph("<b>Master Scientific Verification Audit</b>", card_header_style),
        Spacer(1, 3),
        Paragraph("&bull; <b>Canonical Checkpoint:</b> Verified 4,870,275 params (4.56M trainable) &bull; <b>PASS</b><br/>"
                  "&bull; <b>Mathematical Parity:</b> PyTorch vs ONNX FP32 agreement = 100.00% &bull; <b>PASS</b><br/>"
                  "&bull; <b>OOD Gatekeeper:</b> 100% True Accept / 80% True Reject &bull; <b>PASS</b><br/>"
                  "&bull; <b>Zero Severity Fabrication:</b> Purged all unscientific confidence-to-severity mappings &bull; <b>PASS</b><br/>"
                  "&bull; <b>Offline Autonomy:</b> Zero external API / cloud dependency &bull; <b>PASS</b>", bullet_style),
        Spacer(1, 5),
        Paragraph("<b>Future Research Directions</b>", card_header_style),
        Spacer(1, 3),
        Paragraph("&bull; <b>Learned Gatekeeper:</b> Train a MobileNetV3-Small binary gatekeeper to eliminate 20% weed false-acceptance.<br/>"
                  "&bull; <b>Severity Segmentation:</b> Integrate lightweight UNet / DeepLabV3+ with pixel-level IoU lesion quantification.<br/>"
                  "&bull; <b>Cross-Domain Datasets:</b> Validate across multi-geography cotton datasets (PlantVillage / ICRISAT).", bullet_style)
    ]

    p9_right = [
        Paragraph("<b>Academic References & Literature Citations</b>", card_header_style),
        Spacer(1, 3),
        Paragraph("1. <b>Tan, M. & Le, Q. V. (2019):</b> <i>EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.</i> ICML 2019.<br/>"
                  "2. <b>Hu, J., Shen, L., & Sun, G. (2018):</b> <i>Squeeze-and-Excitation Networks.</i> IEEE CVPR 2018.<br/>"
                  "3. <b>Hinton, G., Vinyals, O., & Dean, J. (2015):</b> <i>Distilling the Knowledge in a Neural Network.</i> NeurIPS Workshop.<br/>"
                  "4. <b>Selvaraju, R. R., et al. (2017):</b> <i>Grad-CAM: Visual Explanations from Deep Networks.</i> IEEE ICCV 2017.<br/>"
                  "5. <b>Howard, A., et al. (2019):</b> <i>Searching for MobileNetV3.</i> IEEE ICCV 2019.<br/>"
                  "6. <b>Huang, G., et al. (2017):</b> <i>Densely Connected Convolutional Networks.</i> IEEE CVPR 2017.<br/>"
                  "7. <b>He, K., Zhang, X., Ren, S., & Sun, J. (2016):</b> <i>Deep Residual Learning for Image Recognition.</i> IEEE CVPR 2016.<br/>"
                  "8. <b>SAR-CLD-2024:</b> <i>Cotton Leaf Disease Benchmark Dataset.</i> Kaggle Open Agriculture Data Repository.", ParagraphStyle('RefStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=7.2, leading=9.5, textColor=colors.HexColor("#334155")))
    ]

    p9_table = Table([[p9_left, p9_right]], colWidths=[355, 365])
    p9_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ffffff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(p9_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF SUITE] Successfully generated presentation PDF: {output_pdf_path}")


# ----------------------------------------------------------------------
# 2. PYTHON-PPTX POWERPOINT SLIDE DECK GENERATION (.PPTX)
# ----------------------------------------------------------------------

def create_card_shape(slide, left, top, width, height, bg_rgb, border_rgb):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_rgb
    shape.line.color.rgb = border_rgb
    shape.line.width = Pt(1.5)
    return shape

def add_header_banner(slide, title_text, category_text="COTTONGUARD AI : RESEARCH PRESENTATION"):
    # Dark header strip
    header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = RGBColor(15, 32, 39) # #0f2027
    header_box.line.fill.background()

    # Green accent strip
    accent_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.1), Inches(13.333), Inches(0.06))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = RGBColor(16, 185, 129) # emerald #10b981
    accent_bar.line.fill.background()

    # Text
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.5)
    tf.margin_top = Inches(0.12)

    p0 = tf.paragraphs[0]
    p0.text = category_text.upper()
    p0.font.name = 'Arial'
    p0.font.size = Pt(10)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(52, 211, 153) # emerald light

    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.name = 'Arial'
    p1.font.size = Pt(20)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)

def add_footer(slide, current_slide, total_slides=10):
    footer_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
    footer_box.fill.solid()
    footer_box.fill.fore_color.rgb = RGBColor(241, 245, 249) # #f1f5f9
    footer_box.line.color.rgb = RGBColor(203, 213, 225)
    footer_box.line.width = Pt(1)

    tf = footer_box.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_right = Inches(0.5)
    tf.margin_top = Inches(0.06)

    p = tf.paragraphs[0]
    p.text = f"CottonGuard AI — Offline Explainable Deep Learning System for Cotton Pathology   |   Slide {current_slide} of {total_slides}"
    p.font.name = 'Arial'
    p.font.size = Pt(9)
    p.font.color.rgb = RGBColor(100, 116, 139)

def build_pptx_presentation(output_pptx_path):
    prs = Presentation()
    # 16:9 Widescreen dimensions: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # completely blank layout

    # -------------------------------------------------------------
    # SLIDE 1: COVER SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(13, 27, 42) # dark navy #0d1b2a
    bg1.line.fill.background()

    # Green accent strip on left
    dec_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.8), Inches(0.12), Inches(3.8))
    dec_bar.fill.solid()
    dec_bar.fill.fore_color.rgb = RGBColor(16, 185, 129)
    dec_bar.line.fill.background()

    # Main Title Box
    title_box = s1.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(11.5), Inches(4.5))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "RESEARCH & ENGINEERING PRESENTATION"
    p_badge.font.name = 'Arial'
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = RGBColor(52, 211, 153)
    p_badge.space_after = Pt(10)

    p_main = tf1.add_paragraph()
    p_main.text = "COTTONGUARD AI"
    p_main.font.name = 'Arial'
    p_main.font.size = Pt(40)
    p_main.font.bold = True
    p_main.font.color.rgb = RGBColor(255, 255, 255)
    p_main.space_after = Pt(10)

    p_sub = tf1.add_paragraph()
    p_sub.text = "An Explainable, Robust & Lightweight Deep Learning Framework for Offline Cotton Leaf Disease Diagnosis on Resource-Constrained Edge Devices"
    p_sub.font.name = 'Arial'
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)
    p_sub.space_after = Pt(24)

    p_stats = tf1.add_paragraph()
    p_stats.text = "Key Achievements: 98.60% Macro F1  •  4.87M Params  •  8.34ms ONNX CPU Inference  •  Real-time Grad-CAM  •  Offline Android"
    p_stats.font.name = 'Arial'
    p_stats.font.size = Pt(12)
    p_stats.font.bold = True
    p_stats.font.color.rgb = RGBColor(110, 231, 183)

    # Footer metadata box
    meta_box = s1.shapes.add_textbox(Inches(0.9), Inches(5.8), Inches(11.5), Inches(1.2))
    tf_meta = meta_box.text_frame
    p_meta1 = tf_meta.paragraphs[0]
    p_meta1.text = "Dataset: SAR-CLD-2024 (Kaggle Verified)   |   Architecture: CottonNet (SE-Attention + KD Distillation)   |   Platform: Native Android"
    p_meta1.font.name = 'Arial'
    p_meta1.font.size = Pt(11)
    p_meta1.font.color.rgb = RGBColor(148, 163, 184)

    # -------------------------------------------------------------
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header_banner(s2, "1. Executive Summary & Problem Formulation")
    add_footer(s2, 2)

    # 3 Cards Layout
    card_w = Inches(3.8)
    card_h = Inches(5.4)
    y_pos = Inches(1.4)

    # Card 1
    create_card_shape(s2, Inches(0.6), y_pos, card_w, card_h, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb1 = s2.shapes.add_textbox(Inches(0.7), y_pos + Inches(0.1), card_w - Inches(0.2), card_h - Inches(0.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Agronomic & Economic Threat"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 95, 70)
    p.space_after = Pt(12)

    points1 = [
        ("Global Pillar: ", "Cotton (Gossypium hirsutum) supports over 100M farming families worldwide."),
        ("Massive Losses: ", "Foliar diseases like Blight, Leaf Curl, and Jassids destroy 30%–40% of crop yield."),
        ("Diagnostic Void: ", "Agricultural officers cannot physically visit every smallholder farm."),
        ("Chemical Overuse: ", "Misdiagnosis leads to indiscriminate pesticide spraying, hurting soil and farm revenue.")
    ]
    for b_title, b_desc in points1:
        p = tf1.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # Card 2
    create_card_shape(s2, Inches(4.7), y_pos, card_w, card_h, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb2 = s2.shapes.add_textbox(Inches(4.8), y_pos + Inches(0.1), card_w - Inches(0.2), card_h - Inches(0.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Current AI Model Bottlenecks"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(185, 28, 28) # Red #b91c1c
    p.space_after = Pt(12)

    points2 = [
        ("Cloud Dependency: ", "Cloud-based APIs fail completely in rural farms with zero 4G/5G connectivity."),
        ("Catastrophic Overconfidence: ", "Standard closed-set CNNs classify fingers, weeds, and noise as 99% confident diseases."),
        ("Black-Box Mistrust: ", "Models provide no visual lesion localization, creating skepticism among farmers."),
        ("Excessive Overhead: ", "Heavy networks (ResNet-50) throttle budget Android smartphone processors.")
    ]
    for b_title, b_desc in points2:
        p = tf2.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # Card 3
    create_card_shape(s2, Inches(8.8), y_pos, card_w, card_h, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb3 = s2.shapes.add_textbox(Inches(8.9), y_pos + Inches(0.1), card_w - Inches(0.2), card_h - Inches(0.2))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "CottonGuard AI Solutions"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(2, 132, 199) # Blue #0284c7
    p.space_after = Pt(12)

    points3 = [
        ("Custom CottonNet: ", "4.87M parameter model with SE Attention achieving 98.60% test Macro F1."),
        ("Two-Stage Gatekeeper: ", "Pre-inference heuristic quality filter rejects blurry, degraded & OOD non-leaf frames."),
        ("Sub-10ms ONNX Engine: ", "Optimized 8.34ms CPU latency with 100% mathematical fidelity."),
        ("100% Offline Mobile: ", "Native trilingual Android app (English, Hindi, Odia) with real-time Grad-CAM.")
    ]
    for b_title, b_desc in points3:
        p = tf3.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 3: DATASET PROVENANCE & 7 DISEASE CLASSES
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header_banner(s3, "2. Dataset Provenance & Target Clinical Taxonomy")
    add_footer(s3, 3)

    # Top info strip
    top_card = create_card_shape(s3, Inches(0.6), Inches(1.3), Inches(12.133), Inches(0.85), RGBColor(240, 253, 244), RGBColor(187, 247, 208))
    tb_top = s3.shapes.add_textbox(Inches(0.7), Inches(1.32), Inches(11.9), Inches(0.8))
    tf_top = tb_top.text_frame
    tf_top.word_wrap = True
    p = tf_top.paragraphs[0]
    p.text = "Dataset Provenance: SAR-CLD-2024 (Kaggle Benchmark)"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(6, 95, 70)
    p2 = tf_top.add_paragraph()
    p2.text = "2,137 high-resolution RGB field images across 7 target classes. Standardized to 224×224 resolution with ImageNet channel normalization constants (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])."
    p2.font.size = Pt(10)
    p2.font.color.rgb = RGBColor(51, 65, 85)

    # Table of 7 classes
    table_shape = s3.shapes.add_table(8, 5, Inches(0.6), Inches(2.3), Inches(12.133), Inches(4.5))
    table = table_shape.table
    table.columns[0].width = Inches(0.8)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(2.5)
    table.columns[3].width = Inches(3.8)
    table.columns[4].width = Inches(2.833)

    headers = ["Class", "Disease / Condition", "Causative Agent", "Visual Symptoms & Manifestation", "Agronomic Advisory"]
    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 32, 39)
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    tax_data = [
        ("0", "Bacterial Blight", "Xanthomonas malvacearum", "Angular water-soaked lesions bounded by veins, turning black ('black arm').", "Copper Oxychloride (0.25%) + Streptocycline (100 ppm)"),
        ("1", "Curl Virus (CLCuV)", "Begomovirus / Whitefly vector", "Upward leaf curling, vein thickening, leaf enations, plant stunting.", "Vector control (Imidacloprid 17.8 SL); rogue infected plants"),
        ("2", "Healthy Leaf", "Normal Foliage (Control)", "Deep green lamina, smooth surface, intact margins, zero lesions.", "Balanced NPK fertilization and scheduled irrigation"),
        ("3", "Herbicide Damage", "Abiotic drift (2,4-D / Glyphosate)", "Strap-like leaf narrowing, finger-like crinkling, petiole distortion.", "Flush with freshwater; foliar zinc + urea (1%) spray"),
        ("4", "Leaf Hopper Jassids", "Amrasca biguttula biguttula", "Margin yellowing ('hopperburn'), downward curl, necrotic edges.", "Flonicamid 50 WG (0.3g/L) or Thiamethoxam 25 WG"),
        ("5", "Leaf Redding", "Physiological Mg deficiency / cold", "Red-purple anthocyanin coloration between green main veins.", "Foliar Magnesium Sulphate (MgSO4 1.0%) + Urea (1.0%)"),
        ("6", "Leaf Variegation", "Micronutrient deficiency / genetic", "Irregular yellowish-white mosaic patches with normal green areas.", "Chelated multi-micronutrient foliar spray (Fe, Zn, Mn)")
    ]

    for row_idx, row in enumerate(tax_data, start=1):
        bg_c = RGBColor(255, 255, 255) if row_idx % 2 == 1 else RGBColor(241, 245, 249)
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(9)
            p.font.color.rgb = RGBColor(15, 23, 42)
            if col_idx == 0:
                p.font.bold = True
                p.alignment = PP_ALIGN.CENTER
            elif col_idx == 1:
                p.font.bold = True

    # -------------------------------------------------------------
    # SLIDE 4: COTTONNET ARCHITECTURE & INNOVATIONS
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header_banner(s4, "3. Proposed Neural Architecture: CottonNet")
    add_footer(s4, 4)

    card_w2 = Inches(5.9)
    card_h2 = Inches(5.4)

    # Left Column: Layer Breakdown
    create_card_shape(s4, Inches(0.6), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb_l = s4.shapes.add_textbox(Inches(0.8), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "Layer-by-Layer Architectural Flow"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 95, 70)
    p.space_after = Pt(10)

    arch_steps = [
        ("Input Tensor: ", "RGB image normalized to [B, 3, 224, 224]."),
        ("EfficientNet-B0 Backbone: ", "Blocks 0 to 6 extract deep hierarchical features → [B, 1280, 7, 7]."),
        ("SE-Block Channel Attention: ", "AdaptiveAvgPool2d → Linear(1280→80, ReLU) → Linear(80→1280, Sigmoid). Dynamic channel recalibration (r=16)."),
        ("Dual Spatial Pooling: ", "Parallel AdaptiveAvgPool2d(1) (context) + AdaptiveMaxPool2d(1) (lesion intensity) concatenated to [B, 2560]."),
        ("Regularized Classification Head: ", "Dropout(0.40) → Linear(2560→256) → BatchNorm1d(256) → ReLU → Dropout(0.30) → Linear(256→7) Logits.")
    ]
    for st, sd in arch_steps:
        p = tf_l.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + st
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = sd
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # Right Column: Why CottonNet & KD
    create_card_shape(s4, Inches(6.8), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb_r = s4.shapes.add_textbox(Inches(7.0), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Architectural Innovations & KD Protocol"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(2, 132, 199)
    p.space_after = Pt(10)

    reasons = [
        ("Attention Saliency: ", "SE-Block weights channels dynamically, boosting subtle spot features over uniform leaf background."),
        ("Dual-Pool Synergism: ", "Combines background luminance context with sharp peak lesion responses for enhanced discrimination."),
        ("Extreme Parameter Efficiency: ", "4,870,275 parameters (~4.87M total) — 4.8x more compact than ResNet-50."),
        ("Knowledge Distillation (KD): ", "Trained with DenseNet-121 teacher using soft distillation loss (T=3.0, α=0.5): L = α·L_KD + (1-α)·L_CE."),
        ("Result: ", "Transfers dark inter-class knowledge, allowing student CottonNet (98.60%) to surpass teacher DenseNet-121 (96.26%).")
    ]
    for rt, rd in reasons:
        p = tf_r.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + rt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = rd
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 5: 5-MODEL BENCHMARK COMPARISON
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header_banner(s5, "4. 5-Model Comparative Research Benchmark")
    add_footer(s5, 5)

    table_shape5 = s5.shapes.add_table(6, 6, Inches(0.6), Inches(1.4), Inches(12.133), Inches(3.4))
    t5 = table_shape5.table
    t5.columns[0].width = Inches(2.2)
    t5.columns[1].width = Inches(1.5)
    t5.columns[2].width = Inches(1.5)
    t5.columns[3].width = Inches(1.8)
    t5.columns[4].width = Inches(1.5)
    t5.columns[5].width = Inches(3.633)

    h5 = ["Model Architecture", "Test Accuracy", "Test Macro F1", "Total Parameters", "Disk Size", "Architecture Role & Status"]
    for col_idx, text in enumerate(h5):
        cell = t5.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 32, 39)
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    b5_data = [
        ("CottonNet (Proposed)", "98.60%", "98.60%", "4,870,275 (~4.87M)", "18.88 MB", "CHAMPION: KD Distilled + SE Attention + Dual Pool"),
        ("DenseNet-121", "96.26%", "96.26%", "6,960,000 (~6.96M)", "28.47 MB", "Teacher Model (Dense Connectivity Baseline)"),
        ("MobileNetV3-Large", "95.79%", "95.79%", "4,210,000 (~4.21M)", "17.06 MB", "Edge Baseline (-2.81% F1 vs CottonNet)"),
        ("EfficientNet-B0 (Base)", "95.33%", "95.33%", "4,020,000 (~4.02M)", "16.37 MB", "Backbone Baseline (-3.27% F1 without custom head)"),
        ("ResNet-50", "93.93%", "93.93%", "23,520,000 (~23.52M)", "94.41 MB", "Heavy Residual Baseline (Lowest F1 & 5x Size)")
    ]

    for row_idx, row in enumerate(b5_data, start=1):
        is_champ = (row_idx == 1)
        bg_c = RGBColor(236, 253, 245) if is_champ else (RGBColor(255, 255, 255) if row_idx % 2 == 1 else RGBColor(248, 250, 252))
        for col_idx, text in enumerate(row):
            cell = t5.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(10)
            if is_champ:
                p.font.bold = True
                p.font.color.rgb = RGBColor(6, 95, 70) if col_idx != 5 else RGBColor(4, 120, 87)
            else:
                p.font.color.rgb = RGBColor(15, 23, 42)
            if col_idx in [1, 2, 3, 4]:
                p.alignment = PP_ALIGN.CENTER

    # Bottom analytical insights card
    bot_card = create_card_shape(s5, Inches(0.6), Inches(5.0), Inches(12.133), Inches(1.8), RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb_bot = s5.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.6))
    tf_bot = tb_bot.text_frame
    tf_bot.word_wrap = True
    p = tf_bot.paragraphs[0]
    p.text = "Key Research Findings & Ablation Takeaways:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(15, 32, 39)
    p.space_after = Pt(4)

    t_points = [
        "1. Student Exceeds Teacher: CottonNet (98.60%) outperforms DenseNet-121 (96.26%) by +2.34% through attention-guided distillation.",
        "2. Ablation Proof: Adding SE Attention & Dual Spatial Pooling over base EfficientNet-B0 increases Macro F1 by +3.27%.",
        "3. High Efficiency: Outperforms ResNet-50 by +4.67% while reducing parameter weight and storage by 80%."
    ]
    for tp in t_points:
        p = tf_bot.add_paragraph()
        p.text = "• " + tp
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(3)

    # -------------------------------------------------------------
    # SLIDE 6: ONNX RUNTIME & LATENCY PROFILING
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header_banner(s6, "5. Edge Optimization & Precision Latency Profiling")
    add_footer(s6, 6)

    table_shape6 = s6.shapes.add_table(5, 7, Inches(0.6), Inches(1.4), Inches(12.133), Inches(3.0))
    t6 = table_shape6.table
    t6.columns[0].width = Inches(2.2)
    t6.columns[1].width = Inches(1.2)
    t6.columns[2].width = Inches(1.3)
    t6.columns[3].width = Inches(1.4)
    t6.columns[4].width = Inches(1.6)
    t6.columns[5].width = Inches(1.6)
    t6.columns[6].width = Inches(2.833)

    h6 = ["Runtime Variant", "Precision", "Model Size", "Compression", "Agreement", "Mean Latency", "Deployment Verdict"]
    for col_idx, text in enumerate(h6):
        cell = t6.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 32, 39)
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    e6_data = [
        ("PyTorch Baseline", "FP32", "18.88 MB", "1.00x", "100.00% (Ref)", "51.44 ms", "Python Backend / Cloud Server"),
        ("ONNX Runtime (FP32)", "FP32", "18.56 MB", "1.02x", "100.00%", "8.34 ms", "CHAMPION: Primary Edge CPU (6.16x faster)"),
        ("ONNX Runtime (FP16)", "FP16", "9.33 MB", "2.02x", "100.00%", "14.12 ms", "Optimal for Mobile GPU / NPU Acceleration"),
        ("ONNX Dynamic INT8", "INT8", "4.94 MB", "3.75x", "0.00% (Drift)", "37.87 ms", "Requires Static Calibration PTQ")
    ]

    for row_idx, row in enumerate(e6_data, start=1):
        is_champ = (row_idx == 2)
        bg_c = RGBColor(236, 253, 245) if is_champ else (RGBColor(255, 255, 255) if row_idx % 2 == 1 else RGBColor(248, 250, 252))
        for col_idx, text in enumerate(row):
            cell = t6.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(10)
            if is_champ:
                p.font.bold = True
                p.font.color.rgb = RGBColor(6, 95, 70)
            elif "Drift" in text:
                p.font.bold = True
                p.font.color.rgb = RGBColor(185, 28, 28)
            else:
                p.font.color.rgb = RGBColor(15, 23, 42)
            if col_idx in [1, 2, 3, 4, 5]:
                p.alignment = PP_ALIGN.CENTER

    # Bottom Decision Card
    bot_card6 = create_card_shape(s6, Inches(0.6), Inches(4.6), Inches(12.133), Inches(2.2), RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb_bot6 = s6.shapes.add_textbox(Inches(0.8), Inches(4.7), Inches(11.7), Inches(2.0))
    tf_bot6 = tb_bot6.text_frame
    tf_bot6.word_wrap = True
    p = tf_bot6.paragraphs[0]
    p.text = "Edge Deployment Strategy & Mathematical Verification:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(15, 32, 39)
    p.space_after = Pt(4)

    e_points = [
        "1. Extreme Speedup: ONNX Runtime FP32 delivers 8.34 ms latency (vs 51.44 ms PyTorch), representing a 6.16x latency reduction with zero cloud dependence.",
        "2. Exact Mathematical Parity: ONNX FP32 and FP16 achieve 100.00% exact prediction agreement with PyTorch weights (max logit delta < 0.000009).",
        "3. Half-Precision Suitability: ONNX FP16 halves model weight to 9.33 MB with zero drift, ideal for Android GPU/NPU delegates.",
        "4. Quantization Finding: Dynamic INT8 achieves 4.94 MB but exhibits logit drift on micro-lesions without static calibration. ONNX FP32/FP16 remain the gold standard."
    ]
    for ep in e_points:
        p = tf_bot6.add_paragraph()
        p.text = "• " + ep
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(3)

    # -------------------------------------------------------------
    # SLIDE 7: TWO-STAGE PIPELINE & OOD GATEKEEPER
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header_banner(s7, "6. Two-Stage Pipeline & Out-of-Distribution Safeguards")
    add_footer(s7, 7)

    # Left: Stage 1 Gatekeeper
    create_card_shape(s7, Inches(0.6), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb7_l = s7.shapes.add_textbox(Inches(0.8), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf7_l = tb7_l.text_frame
    tf7_l.word_wrap = True
    p = tf7_l.paragraphs[0]
    p.text = "Stage-1: Pre-Inference Quality Gatekeeper"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(185, 28, 28)
    p.space_after = Pt(10)

    g_points = [
        ("Laplacian Blur Filter: ", "Computes variance of Laplacian operator. Rejects blurry frames (Var < 80.0) preventing unreliable inference."),
        ("Luminance Guard: ", "Rejects severe underexposure (< 40 mean luminance) and glare (> 215) to maintain feature fidelity."),
        ("HSV Foliage Masking: ", "Enforces minimum green/foliage hue coverage (≥ 15%), rejecting human skin, soil, clothing, and background walls."),
        ("True Acceptance Rate: ", "100.0% acceptance on legitimate cotton leaves (0.0% false rejection rate)."),
        ("True Rejection Rate: ", "80.0% overall rejection on non-cotton samples (100% abiotic/skin rejection; 20% FAR on non-cotton weeds).")
    ]
    for gt, gd in g_points:
        p = tf7_l.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + gt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = gd
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # Right: Stage 2 Decision Head
    create_card_shape(s7, Inches(6.8), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb7_r = s7.shapes.add_textbox(Inches(7.0), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf7_r = tb7_r.text_frame
    tf7_r.word_wrap = True
    p = tf7_r.paragraphs[0]
    p.text = "Stage-2: CottonNet Inference & Decision Head"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 95, 70)
    p.space_after = Pt(10)

    s2_points = [
        ("Input Normalization: ", "Transforms RGB buffer using exact ImageNet constants: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]."),
        ("Softmax Top-3 Vector: ", "Outputs calibrated class probabilities across all 7 pathological categories."),
        ("Calibrated Confidence Threshold: ", "Operational threshold set at τ = 0.40. Unconfident inputs trigger 'Indeterminate / Retake Photo'."),
        ("Scientific Integrity: ", "Purged all arbitrary confidence-to-severity mappings, adhering to rigorous peer-review standards."),
        ("Agronomic Prescription: ", "Dispatches targeted curative chemical dosages and preventive cultural practices.")
    ]
    for st, sd in s2_points:
        p = tf7_r.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + st
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = sd
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 8: EXPLAINABLE AI & ENVIRONMENTAL ROBUSTNESS
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header_banner(s8, "7. Explainable AI (Grad-CAM) & Environmental Robustness")
    add_footer(s8, 8)

    # Left: Grad-CAM
    create_card_shape(s8, Inches(0.6), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb8_l = s8.shapes.add_textbox(Inches(0.8), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf8_l = tb8_l.text_frame
    tf8_l.word_wrap = True
    p = tf8_l.paragraphs[0]
    p.text = "Visual Explainability (Grad-CAM)"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(2, 132, 199)
    p.space_after = Pt(10)

    cam_points = [
        ("Feature Map Attribution: ", "Computed dynamically from backbone.blocks[-1] feature activations."),
        ("Gradient Weighting: ", "α_k^c = (1/Z) Σ_i Σ_j (∂ y^c / ∂ A_i,j^k), computing class saliency maps."),
        ("Morphological Verification: ", "Isolates vein thickening in Curl Virus, angular spots in Blight, and margin scorch in Jassids."),
        ("Building Trust: ", "Allows agronomists and farmers to visually verify that the model is looking at the disease rather than background soil."),
        ("Edge Grad-CAM: ", "Computed in real time on mobile devices alongside the classification logit vector.")
    ]
    for ct, cd in cam_points:
        p = tf8_l.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + ct
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = cd
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # Right: Robustness
    create_card_shape(s8, Inches(6.8), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb8_r = s8.shapes.add_textbox(Inches(7.0), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf8_r = tb8_r.text_frame
    tf8_r.word_wrap = True
    p = tf8_r.paragraphs[0]
    p.text = "Empirical Robustness (11 Perturbations)"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 95, 70)
    p.space_after = Pt(10)

    rob_points = [
        ("Sensor Noise Resilience: ", "High resilience under Gaussian noise (σ=15, ΔConf = -2.62%). Extreme noise (σ=30) triggers noise rejection guard."),
        ("Motion Blur Protection: ", "High resilience at r=1 (ΔConf = -1.32%). Severe blur (r=3) is safely rejected by Laplacian gatekeeper (58.3%)."),
        ("Illumination Invariance: ", "Robust under 0.7x low-light and 1.6x overexposure. Extreme darkness (<0.4x) triggers exposure guard."),
        ("Compression Invariance: ", "JPEG compression at Q=20 drops confidence by merely -1.02%, enabling efficient transmission."),
        ("Rotation Invariance: ", "Tested at 45° and 90° rotations; confidence dropped < 1.9%, proving spatial robustness.")
    ]
    for rt, rd in rob_points:
        p = tf8_r.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + rt
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = rd
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 9: ANDROID DEPLOYMENT & FARMER UX
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header_banner(s9, "8. Native Android Mobile Deployment & Farmer UX")
    add_footer(s9, 9)

    # 3 Cards Layout
    create_card_shape(s9, Inches(0.6), y_pos, card_w, card_h, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb9_1 = s9.shapes.add_textbox(Inches(0.7), y_pos + Inches(0.1), card_w - Inches(0.2), card_h - Inches(0.2))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True
    p = tf9_1.paragraphs[0]
    p.text = "Edge Camera & Runtime"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 95, 70)
    p.space_after = Pt(12)

    a_points1 = [
        ("CameraX Viewfinder: ", "Real-time crop viewfinder with auto-focus and illumination assistance."),
        ("ONNX Runtime Mobile: ", "Runs directly on Android CPU threads via NDK; zero cloud API calls."),
        ("Zero Latency: ", "Sub-10ms diagnostic latency ensures instant feedback for farmers in fields."),
        ("Battery Conscious: ", "Optimized CPU wake cycles prevent smartphone battery drain during long field scans.")
    ]
    for b_title, b_desc in a_points1:
        p = tf9_1.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    create_card_shape(s9, Inches(4.7), y_pos, card_w, card_h, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb9_2 = s9.shapes.add_textbox(Inches(4.8), y_pos + Inches(0.1), card_w - Inches(0.2), card_h - Inches(0.2))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True
    p = tf9_2.paragraphs[0]
    p.text = "Trilingual Agronomic Advisory"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(2, 132, 199)
    p.space_after = Pt(12)

    a_points2 = [
        ("Trilingual Support: ", "Native localization in English, Hindi (हिन्दी), and Odia (ଓଡ଼ିଆ)."),
        ("Actionable Chemical Dosages: ", "Clear curative instructions with precise chemical concentrations (e.g., g/L, ppm)."),
        ("Voice-Assisted Readout: ", "Integrated with Android Text-to-Speech (TTS) for accessibility to illiterate farmers."),
        ("Farmer Privacy: ", "Images and diagnostic records stay 100% on the local smartphone storage.")
    ]
    for b_title, b_desc in a_points2:
        p = tf9_2.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    create_card_shape(s9, Inches(8.8), y_pos, card_w, card_h, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb9_3 = s9.shapes.add_textbox(Inches(8.9), y_pos + Inches(0.1), card_w - Inches(0.2), card_h - Inches(0.2))
    tf9_3 = tb9_3.text_frame
    tf9_3.word_wrap = True
    p = tf9_3.paragraphs[0]
    p.text = "Offline SQLite & History"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(217, 119, 6) # amber
    p.space_after = Pt(12)

    a_points3 = [
        ("Room SQLite Storage: ", "Maintains structured offline history of previous field scans and diagnoses."),
        ("Temporal Tracking: ", "Allows farmers to monitor whether crop recovery occurs after applying treatments."),
        ("Offline PDF Reports: ", "Exports diagnosis summaries to share with regional agricultural officers."),
        ("Zero Network Overhead: ", "Complete operational autonomy in deep agricultural hinterlands.")
    ]
    for b_title, b_desc in a_points3:
        p = tf9_3.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 10: SCIENTIFIC AUDIT, CITATIONS & ROADMAP
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header_banner(s10, "9. Scientific Audit, References & Future Directions")
    add_footer(s10, 10)

    # Left: Audit Checklist & Future Work
    create_card_shape(s10, Inches(0.6), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb10_l = s10.shapes.add_textbox(Inches(0.8), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf10_l = tb10_l.text_frame
    tf10_l.word_wrap = True
    p = tf10_l.paragraphs[0]
    p.text = "Scientific Audit & Future Directions"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 95, 70)
    p.space_after = Pt(8)

    audit_items = [
        ("Checkpoint Verification: ", "4,870,275 parameters verified (PASS)."),
        ("Mathematical Parity: ", "PyTorch vs ONNX FP32 agreement = 100.00% (PASS)."),
        ("OOD Gatekeeper: ", "100% True Accept / 80% True Reject (PASS)."),
        ("Zero Fabrication: ", "Purged all unscientific severity labels (PASS)."),
        ("Future Learned Gatekeeper: ", "Train a MobileNetV3-Small binary gatekeeper to eliminate 20% weed false-acceptance."),
        ("Lightweight Segmentation: ", "Integrate UNet / DeepLabV3+ for automated pixel-level IoU lesion area quantification.")
    ]
    for at, ad in audit_items:
        p = tf10_l.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + at
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run()
        r2.text = ad
        r2.font.size = Pt(10)
        r2.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(5)

    # Right: Academic Citations
    create_card_shape(s10, Inches(6.8), Inches(1.4), card_w2, card_h2, RGBColor(255, 255, 255), RGBColor(203, 213, 225))
    tb10_r = s10.shapes.add_textbox(Inches(7.0), Inches(1.5), card_w2 - Inches(0.4), card_h2 - Inches(0.3))
    tf10_r = tb10_r.text_frame
    tf10_r.word_wrap = True
    p = tf10_r.paragraphs[0]
    p.text = "Academic Literature & References"
    p.font.name = 'Arial'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(2, 132, 199)
    p.space_after = Pt(8)

    citations = [
        "1. Tan, M. & Le, Q. V. (2019): EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.",
        "2. Hu, J., Shen, L., & Sun, G. (2018): Squeeze-and-Excitation Networks. IEEE CVPR 2018.",
        "3. Hinton, G., Vinyals, O., & Dean, J. (2015): Distilling the Knowledge in a Neural Network. NeurIPS Workshop.",
        "4. Selvaraju, R. R., et al. (2017): Grad-CAM: Visual Explanations from Deep Networks. IEEE ICCV 2017.",
        "5. Howard, A., et al. (2019): Searching for MobileNetV3. IEEE ICCV 2019.",
        "6. Huang, G., et al. (2017): Densely Connected Convolutional Networks. IEEE CVPR 2017.",
        "7. He, K., Zhang, X., Ren, S., & Sun, J. (2016): Deep Residual Learning for Image Recognition. IEEE CVPR 2016.",
        "8. SAR-CLD-2024: Cotton Leaf Disease Benchmark Dataset. Kaggle Open Agriculture Repository."
    ]
    for cit in citations:
        p = tf10_r.add_paragraph()
        p.text = cit
        p.font.size = Pt(8.5)
        p.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(4)

    prs.save(output_pptx_path)
    print(f"[PPTX SUITE] Successfully generated PowerPoint presentation: {output_pptx_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_out = os.path.join(base_dir, "CottonGuard_AI_Project_Presentation.pdf")
    pptx_out = os.path.join(base_dir, "CottonGuard_AI_Project_Presentation.pptx")
    
    print("Generating Presentation Suite...")
    build_pdf_presentation(pdf_out)
    build_pptx_presentation(pptx_out)
    print("SUITE GENERATION COMPLETE!")
