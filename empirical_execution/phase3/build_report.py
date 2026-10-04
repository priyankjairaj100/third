"""Build the phase3 brief from the frozen software evidence."""
from pathlib import Path
from xml.sax.saxutils import escape
import json

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/counterfactual_curation_empirical_preparation.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
INK = colors.HexColor("#19374C")
TEAL = colors.HexColor("#126C78")
GREY = colors.HexColor("#51616C")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="MainTitle", fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle(name="Deck", fontName="Helvetica", fontSize=12, leading=17, textColor=GREY, spaceAfter=18))
styles.add(ParagraphStyle(name="SectionCustom", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=15, spaceAfter=8))
styles.add(ParagraphStyle(name="BodyCustom", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, spaceAfter=9))
styles.add(ParagraphStyle(name="SmallCustom", fontName="Helvetica", fontSize=9, leading=12.5, textColor=GREY, spaceAfter=7))
styles.add(ParagraphStyle(name="MonoCustom", fontName="Courier", fontSize=9.2, leading=14, textColor=INK, spaceBefore=5, spaceAfter=10))
story=[]
def para(text, style="BodyCustom"):
    return Paragraph(text, styles[style])
def add(text, style="BodyCustom"):
    story.append(para(text, style))
def section(text): add(text, "SectionCustom")
def table(rows, widths):
    body=[[para(escape(str(cell)), "SmallCustom") for cell in row] for row in rows]
    t=Table(body, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5EFF2")),("LINEBELOW",(0,0),(-1,0),.7,TEAL),("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#D4DFE3")),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.append(t)

add("Counterfactual curation", "MainTitle")
add("Empirical preparation completed<br/>ACL 2027 research program | 4 October 2026", "Deck")
add("<b>The next preparation stage is implemented and audited.</b> It covers source partitions, leakage guards, a common numerical reference, blinded calibration sampling, an exact statistical quality gate and an executable local workflow. The primary semantic study remains blocked by missing authentic inputs.")
section("Concrete work completed")
table([("Component","Delivered behavior"),
("Population preparation","70/15/15 source hashing; test > calibration > train exact/link guards; directed Civil parent guard; whole-source primary/replication panels."),
("Fixed numerical reference","Calibration, semantic filtering and graph construction use the same coordinate-ordered FP64 score on frozen FP32 vectors."),
("Threshold workflow","Up to 30 pairs from each of 20 score bins; exact inclusion weights; three independent ratings per pair; fresh validation sample up to 200."),
("Quality and integrity","Exact one-sided 95% finite-population lower bound; blank ratings rejected; files, rows, source partitions, vectors and scoring code bound to frozen manifests."),
("Intake corrections","Versioned Civil normalization, community-account and directed-parent fixes. Prior phase 2 code and development results remain unchanged.")],[123,369])
section("What the evidence does not establish")
add("No E5 or MPNet embeddings were produced; no human ratings were collected; no semantic threshold or primary quality gate passed. The available Civil 100 and News 90 previews remain engineering data. The source-rich original archives, semantic assets and actual independent judgments are absent.")
add("The supported acquisition routes exposed metadata, schemas and previews but did not import corpus/model files. No remote computation was launched. Corpus-specific original-data parsers, date/link completeness and the remaining pre-execution lock are still required.","SmallCustom")
story.append(PageBreak())

add("A stronger computational target", "MainTitle")
add("The useful algorithmic improvement is pair-local reference scoring.","Deck")
add("BLAS reductions can depend on matrix shape and operation order. Close to a threshold, that leaves ambiguity between a score used during calibration, a tiled leakage check and a graph rebuilt after deletion. A fixed-graph theorem needs a computational predicate that is stable across those contexts.")
section("Prospectively fixed scoring rule")
add("Promote each stored FP32 vector to FP64. Compute its squared norm by ordered coordinate accumulation, divide by the computed square root, then accumulate each pair's coordinate products in the same order. Apply the strict predicate score &gt; threshold. Do not clip scores; only score-bin indices saturate at the endpoints.")
add("For fixed ID-derived priorities, frozen vectors, threshold and runtime:","BodyCustom")
add("G(D minus F) = induced_graph(G(D), D minus F)","MonoCustom")
add("<b>Reason.</b> Every retained edge depends on the same two vectors, the same sequence of numerical operations and the same ID priorities. Removing other records cannot change any of those inputs. The graph identity follows pair by pair.")
add("This is a finite-precision consistency result within the locked runtime. It is not a real-arithmetic cosine certificate, a semantic-quality guarantee, or a promise of equality on arbitrary hardware.","SmallCustom")
section("Memory is accounted for, not hidden")
table([("Quantity","Cost / limitation"),
("Pair-scoring workspace","O(BD + B squared) for block size B and dimension D."),
("Exhaustive calibration arithmetic","O(N squared D); no large-scale speed claim has been measured."),
("Stored graph","O(N + m) for m edges; dense worst case remains quadratic."),
("CLI calibration gather","O(N_cal D) additional FP32 bytes, beyond the input cache and row metadata.")],[178,314])
section("Historical evidence remains historical")
add("The new scorer is an explicit pre-confirmation amendment. It does not silently change the previously saved phase 2 lexical experiments. Those results and the new software checks have separate numerical targets and clearly stated scope.")
story.append(PageBreak())

add("Independent checks and execution gate", "MainTitle")
add("Passed checks support the software; authentic data and ratings support the study.","Deck")
table([("Independent check","Observed result"),
("Exact confidence bound","2,900 inversion cases and 5,200 coverage cases; maximum exact noncoverage = 0.05."),
("Scalar scoring oracle","All 4,950 pairs at each of 128 and 768 lexical dimensions matched; tile, retained-subset and row-order checks passed."),
("Graph and guards","Graph comparisons passed at three priority seeds; 27 directed guard cases, eight missing/community-owner cases and seven malformed-ID cases passed."),
("Blinded engineering pack","241 sampled natural pairs; 723 blank assignments; exact probabilities checked; zero human responses."),
("Implementation checks (separate)","48 panel checks, 31 calibration checks, 69 intake assertions, 10 graph checks and nine CLI success/refusal cases passed.")],[166,326])
section("What the statistical gate means")
add("Selection chooses the least strict qualifying grid threshold using weighted precision. A separate SRS validation sample then supplies a one-sided 95% exact hypergeometric lower bound. The primary statistical gate needs a lower bound of at least 0.90 and successful development selection. Empty populations have undefined precision; failed validation cannot be rescued by retuning.")
add("Coverage concerns fixed or potential outcomes under the prescribed three-human-rater majority protocol, conditional on independent sampling and rater assignment. It does not cover latent semantic truth or additional annotator variability. JSON declarations and hashes do not authenticate humans or historical provenance.","SmallCustom")
section("What is needed for the next actual run")
add("<b>1.</b> Original source-rich Civil Comments and/or Ask Ubuntu files, with native IDs, source fields and snapshot provenance.<br/><b>2.</b> A pinned E5 FP32 cache with derivation logs, or usable local encoder/tokenizer files and runtime.<br/><b>3.</b> Genuine independent blinded ratings for selection and then fresh quality validation.<br/><b>4.</b> Verified corpus-specific parsing, required panels and the remaining pre-execution lock before the primary comparison.")
add("The ZIP includes the command-line workflow, annotation instructions, versioned intake, independent audits, code hashes and the Civil engineering fixture. README.txt gives the exact local command sequence. The News article bodies are omitted from redistribution. Every automated output keeps full-study readiness false until the outside scientific gates are satisfied.","SmallCustom")

def page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(TEAL);canvas.setLineWidth(.8);canvas.line(51,800,543,800)
    canvas.setFont("Helvetica",8);canvas.setFillColor(GREY)
    canvas.drawString(51,32,"COUNTERFACTUAL CURATION | PREPARATION EVIDENCE, NOT PRIMARY RESULTS")
    canvas.drawRightString(543,32,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=(595,842),rightMargin=51,leftMargin=51,topMargin=58,bottomMargin=53,
    title="Counterfactual Curation: Empirical Preparation",author="Research working draft")
doc.build(story,onFirstPage=page,onLaterPages=page)
print(OUT)
