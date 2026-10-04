#!/usr/bin/env python3
"""Render a measured, explicitly nonconfirmatory execution report."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parent;RESULTS=ROOT/'results'
REPORT=ROOT.parent/'output/pdf/counterfactual_curation_execution_01.pdf'

def main():
    report=json.loads((RESULTS/'execution_summary.json').read_text())
    bounds=json.loads((RESULTS/'boundary_audit.json').read_text())
    audit=json.loads((RESULTS/'structure_audit.json').read_text())
    environment=json.loads((ROOT/'environment_audit.json').read_text())
    entries=sorted(report['numeric'],key=lambda v:v['dimension'])
    ncheck=sum(v['checkpoint_rows_passed'] for v in entries)
    nfail=sum(len(v['failed_trajectories']) for v in entries)
    maxhead=max(v['max_abs_head_error'] for v in entries)
    maxmoment=max(v['max_abs_moment_error'] for v in entries)
    full=next(v for v in bounds['events'] if v['check']=='all_100_natural_records_deleted')
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(6.7,3.2),layout='constrained')
    xs=list(range(len(entries)));width=.34
    base=[v['initial_accounting']['B-E']['all_array_bytes']/1e6 for v in entries]
    prop=[v['initial_accounting']['P-I']['numeric_coefficient_bytes']/1e6 for v in entries]
    ax.bar([x-width/2 for x in xs],base,width,label='Eligible payload + current moments',color='#197b8d')
    ax.bar([x+width/2 for x in xs],prop,width,label='Indexed coefficient arrays',color='#e3a129')
    ax.set_yscale('log');ax.set_xticks(xs,[str(v['dimension']) for v in entries])
    ax.set_ylabel('Initial allocated array MB (log scale)');ax.set_xlabel('Lexical learner dimension; curator fixed at 128 dimensions')
    ax.legend(loc='upper left',frameon=False,fontsize=8)
    for x,b,p in zip(xs,base,prop):
        ax.annotate(f'{b:.3f}',(x-width/2,b),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        ax.annotate(f'{p:.2f}',(x+width/2,p),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylim(.025,800);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(figdir/'engineering_state_arrays.png',dpi=180)
    fig.savefig(figdir/'engineering_state_arrays.svg');plt.close(fig)

    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleInk',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=colors.HexColor('#19374c'),spaceAfter=13))
    styles.add(ParagraphStyle(name='SubInk',fontName='Helvetica-Bold',fontSize=13,leading=17,textColor=colors.HexColor('#19374c'),spaceBefore=13,spaceAfter=7))
    styles.add(ParagraphStyle(name='TextBody',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallBody',fontName='Helvetica',fontSize=8.5,leading=11.5,spaceAfter=6))
    flow=[]
    def p(text,style='TextBody'):flow.append(Paragraph(text,styles[style]))
    def h(text):p(text,'SubInk')
    def table(rows,widths):
        t=Table([[Paragraph(str(cell),styles['SmallBody']) for cell in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f3')),('VALIGN',(0,0),(-1,-1),'TOP'),
          ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8096a4')),
          ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#8096a4'))]))
        flow.append(t);flow.append(Spacer(1,8))
    p('Counterfactual curation<br/>First local execution','TitleInk')
    p('Measured engineering progress | 4 October 2026 | All computation in this workspace','SmallBody')
    p('<b>Implemented and exercised the core repair pipeline on natural text.</b> The independent curation oracle, eligible-payload baseline, indexed coefficient summary, source/record structural analysis and cumulative request harness are now runnable. No remote compute jobs were used.')
    p('<b>Scope:</b> these are implementation checks, not the ACL paper\'s confirmatory experiments. The data are small, connector-rendered convenience previews. Features are fixed lexical hashing, not E5 or MPNet; thresholds lack human semantic calibration. No synthetic documents or labels were introduced.')
    h('Verified results')
    table([
      ['Check','Measured outcome','Interpretation'],
      ['Natural records','100 Civil Comments + 90 CC-News articles','Civil retains original toxicity labels. News receives no invented target.'],
      ['Joint method/oracle checks',f'{ncheck} release checkpoints; {nfail} failed trajectories','Selection, count, moments and heads checked; not independent population trials.'],
      ['Numerical agreement',f'Max head difference {maxhead:.2e}; max moment difference {maxmoment:.2e}','FP64 diagnostic agreement, not a rigorous roundoff certificate.'],
      ['Complete coefficient rebuild','8 preselected checkpoints at learner dimensions 64 and 128','All keys/coordinates compared; this audit does not cover every run.'],
      ['Full deletion',f'{full["releases"]} additional releases; zero final head','Raw cancellation residue remains visible rather than hidden.'],
      ['News source selection',f'{audit["news_source_audit"]["source_subset_checks"]} subset checks; no mismatches','All subsets of five observed hosts at three lexical thresholds.'],
      ['Fresh-process replay','4 outputs match; zero observed guarded Python I/O attempts','Receives state and request IDs only; not an OS security or privacy proof.'],
    ],[115,159,207])
    h('A boundary bug was found and fixed')
    p('When the exact selected count reaches zero, floating cancellation can leave a tiny residual in the stored moments. The decoder now returns the prescribed zero head while exposing that residue to the auditor. Both kernels also reject duplicate/retried, unknown and out-of-budget requests atomically without consuming additional horizon.')

    flow.append(PageBreak())
    p('The memory risk is already visible','TitleInk')
    p('The following measurements use the same 100 Civil records, fixed 128-dimensional lexical curator, threshold 0.6 and deletion horizon eight. Only the lexical learner dimension changes. Each proposed state initially has 99 coefficient keys.')
    flow.append(Image(str(figdir/'engineering_state_arrays.png'),width=440,height=210))
    table([['Learner dimension','B-E arrays (MB)','P-I arrays (MB)','P-I / B-E arrays']]+[
      [v['dimension'],f'{b:.3f}',f'{pval:.3f}',f'{pval/b:.1f}x'] for v,b,pval in zip(entries,base,prop)],
      [122,119,119,121])
    p('<b>These are initial allocated array bytes in decimal MB.</b> They are not total physical memory, process peak, or serialized size. B-E includes eligible FP32 feature rows, FP64 targets and its current full Gram/cross matrices. P-I includes packed coefficient arrays. Metadata, matched public-ID adapters, allocator slack, external corpus/graph assets and workspaces are excluded from this table. Full physical accounting remains required.')
    p('The comparison is unfavorable to dense coefficient storage even though it already packs symmetric matrices. At 768 lexical dimensions, coefficient arrays use about 234 MB while baseline arrays use about 5 MB. This validates the need for the protocol\'s storage gate; it does not prove a universal lower bound or predict the results of the actual E5 model.')
    h('What changed during development')
    p('The initial harness varied curation dimension together with learner dimension. That confounded a dimension comparison. Its outputs were archived; the current pilot was rerun with a fixed curation graph and matched request paths. The amendment predates any confirmatory run and is recorded in ENGINEERING_CHANGELOG.txt.')
    h('Timing interpretation')
    p('Times are logged as unreplicated development observations. No speedup claim is made; full lifecycle costs, matched identifiers and hardware-blocked repetitions remain pending.')

    flow.append(PageBreak())
    p('Acquisition and execution boundaries','TitleInk')
    table([
      ['Resource / gate','Observed status','Consequence'],
      ['Local compute',f'{environment["visible_cpu_count"]} visible CPUs; approximately {environment["physical_memory_bytes_reported"]/1024**3:.2f} GiB reported physical memory','The proposed 128-GiB per-method study envelope is unavailable here. No GPU was detected.'],
      ['Numerical stack','NumPy, SciPy and scikit-learn available','Exact-structure and convex-head engineering work runs here.'],
      ['Transformer stack','No installed torch/transformers runtime or E5/MPNet weights','Primary semantic embeddings have not been computed.'],
      ['Civil provenance','Current connector preview omits original article/publication/parent IDs','Article/source withdrawal is disabled for this slice; no source IDs are guessed.'],
      ['News provenance','90 complete parsed rows; five observed hosts; two pages failed','This is not a complete source-group or representative corpus sample.'],
      ['File fidelity','Untouched connector responses and derived-row hashes retained','Original corpus bytes and immutable Hub revisions remain unverified.'],
      ['Human quality gate','Not performed','No semantic-duplicate precision or restored-information claim.'],
    ],[101,172,208])
    h('Completed software components')
    p('Exhaustive earlier-raw-neighbor graph construction; independently recomputed retained selection; source and record incidence rank; expected-admission formulas; cumulative R/S/U/A request generation; optimized eligible-row bookkeeping; indexed polynomial repair; common FP64 Cholesky decoding; memory preflight; checkpoint/resume; atomic failure checks; local ingestion validation; provenance manifests; and a failure-preserving result registry.')
    h('Next gate for the primary study')
    p('The source-rich original corpus assets and pinned model/runtime files must become available inside this workspace through a permitted import route. The current working connector exposes rendered previews rather than binary dataset/model downloads. The configured network allowlist does not provide a direct route to those hosts. No bypass or external training service was used.')
    p('Once those assets are available here, the next steps remain local: validate checksums and metadata, construct source-disjoint panels, compute pinned embeddings, complete human calibration, and run the native storage preflight before confirmation. A smaller actual memory cap must be locked and all infeasible cells retained. Synthetic replacements and silent substitution of lexical results for semantic results are excluded.')
    h('Reproducibility scope')
    p('The companion archive contains code, exact current outputs, archived development outputs, checksums, environment notes and the CC0 Civil engineering fixture. The raw CC-News text stays in the local working cache; the archive contains its provenance and audit results rather than redistributing the articles. All headline counts in this report resolve to saved JSON records. The main study manifest remains unlocked.')

    def footer(canvas,doc):
        canvas.saveState();canvas.setFillColor(colors.HexColor('#536978'));canvas.setFont('Helvetica',8)
        canvas.drawString(56,30,'Counterfactual curation | Engineering execution 01 | Not confirmatory')
        canvas.drawRightString(A4[0]-56,30,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(REPORT),pagesize=A4,rightMargin=56,leftMargin=56,topMargin=48,bottomMargin=48,
      title='Counterfactual Curation - First Local Execution',author='Research execution workspace').build(flow,onFirstPage=footer,onLaterPages=footer)
    print(REPORT)

if __name__=='__main__':main()
