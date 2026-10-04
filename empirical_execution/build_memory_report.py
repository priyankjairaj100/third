#!/usr/bin/env python3
"""Render the memory repair derivation and actual natural-text verification."""
from pathlib import Path
import json,os,shutil,subprocess

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=PROJECT/'output/pdf'
WORK=PROJECT/'tmp/pdfs'

def main():
    report=json.loads((ROOT/'memory_results/summary.json').read_text())
    peaks=json.loads((ROOT/'memory_results/process_peaks.json').read_text())
    cases=report['cases'];maincase=next(c for c in cases if c['dimension']==768 and c['outputs']==1)
    ratio=maincase['initial_array_bytes']['dense']/maincase['initial_array_bytes']['joint_span']
    rows=[]
    for c in cases:
        b=c['initial_array_bytes']
        rows.append(f"{c['dimension']} & {c['outputs']} & {b['dense']/1e6:.3f} & {b['gram_factor']/1e6:.3f} & {b['joint_span']/1e6:.3f} & {b['compact_payload']/1e6:.3f} \\\\")
    head=max(c['max_head_error']['joint_span'] for c in cases)
    compact=max(c['max_compact_decode_error'] for c in cases)
    full=report['boundary_checks'][-1]['max_head_error']['joint_span']
    peakrows=[]
    names={'dense':'Dense coefficients','gram_factor':'Signed Gram factors','joint_span':'Joint span + reduced decoder','compact_payload':'Compact eligible payload'}
    for p in peaks:
        peakrows.append(f"{names[p['method']]} & {p['peak_RSS_KiB_after_setup']/1024:.2f} & {p['peak_RSS_KiB_after_four_releases']/1024:.2f} \\\\")
    header=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,amssymb,amsthm,booktabs,array}
\usepackage{xcolor,enumitem,fancyhdr}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!55!black]{hyperref}
\definecolor{ink}{HTML}{19374C}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Counterfactual curation}\fancyhead[R]{\small Memory repair, version 1}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlist{nosep,leftmargin=1.5em}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\title{\color{ink}\textbf{Eliminating Quadratic Coefficient Storage}\\[5pt]\Large Algorithms, proofs and local verification\\for counterfactual curation repair}
\author{ACL 2027 research project}
\date{4 October 2026}
\begin{document}\maketitle
\textbf{Outcome.} The quadratic expansion in the original ridge-summary
implementation has been removed. The stronger joint-span algorithm represents
each coefficient through its current Gram/cross-moment range. Its exact-rank
persistent numeric storage is $O(E_h(d+c))$, in place of $O(Kd^2+Kdc)$.
An implementation-safe envelope uses initial eligibility $E_0$, because
floating rank residue and sticky dense fallbacks can outlive exact rank loss.

\textbf{Implemented, not just proposed.} Two compact summaries, a reduced
ridge decoder, deterministic projector-based basis coordinates, streaming
checkpoints, and a stronger payload comparator were implemented and run here.
The matrix factorization primitives are classical; the paper contribution
must rest on their curation-aware guarantee and empirical consequences.

\textbf{Measured result.} On the fixed Civil engineering graph at $d=768$,
the original dense coefficient arrays used 234.483 MB. Joint-span arrays use
1.039 MB: a \textbf{RATIO-fold reduction} (99.56\%). These are owned numerical
arrays, not process peak or total physical memory. A fair compact payload
baseline uses only 0.256 MB, so the new summary is not a universal memory winner.

\textbf{Evidence scope.} The same 100 natural Civil Comments records were
used, with their original labels. Curation is fixed 128-dimensional lexical
hashing at threshold 0.6, independent of learner dimension. No synthetic
documents or labels, semantic encoder claims, or external compute were used.
These are engineering results; full NLP utility and semantic validation remain
outside this local pilot.

\textbf{State scope.} The fast implementation repairs decoded coefficients
and heads. Canonical basis coordinates remove one historical ambiguity,
but sticky dense/compact mode and floating arithmetic prevent a claim of
whole-state equality to fresh construction. The stronger canonical-state
variant requires additional, explicitly charged re-encoding.
\newpage
\section*{Measured comparisons}
All cases share the same records, graph, horizon eight, regularization
$\lambda=0.01$, request generator and independent retained-data oracle.
The seven-output cases use all seven original Civil toxicity annotations;
they introduce no fabricated target. MB below means $10^6$ bytes.
\begin{center}\small
\begin{tabular}{rrrrrr}\toprule
$d$ & $c$ & Dense & Gram factor & Joint span & Compact payload\\\midrule
TABLE_ROWS
\bottomrule\end{tabular}\end{center}
Counts in the two new summaries are exact Python integers, charged in their
separate Python live-size estimates. The table excludes those objects,
identifier dictionaries, graph/corpus input, allocator slack and temporary
workspace. The original dense state includes its count coordinate; that
small difference is not responsible for the reduction.

The signed-Gram ablation still stores $H_J$ densely. Its seven-output cost
exposes why the joint range, rather than Gram-only compression, is needed.
The compact payload comparator retains eligible FP32 features and FP64 labels,
stores no permanent Gram/kernel/head, and chooses primal or dual ridge by the
smaller dimension. Its persistent rows are a different storage contract.

\subsection*{Verification actually run}
\begin{itemize}
\item 544 release checkpoints across three scalar-output dimension panels
and two seven-output panels passed against an independently rescored oracle.
The joint summary's largest head difference was HEADERR; moment difference
was $4.00\times10^{-15}$.
\item The compact decoder was actually invoked at all 544 checkpoints. Its
largest oracle head difference was COMPACTERR.
\item Another 100 sequential releases deleted the entire natural corpus.
Joint-summary head difference was at most FULLERR; the final head was exactly
the prescribed zero. This is a stronger stress than the short request horizon.
\item Twelve atomic-rejection checks and three order/batching comparisons
passed. Twenty checkpoints per compressed method additionally compared every
coefficient to the dense implementation, streamed one key at a time.
\item Independent coordinate-permutation/sign audits used three actual
coefficients. Streamed checkpoint roundtrip reproduced coefficients exactly,
owned disjoint buffers, and supported an identical subsequent repair.
\end{itemize}
These are tolerance checks, not a certified roundoff or privacy proof. The
full coefficient reference is the dense implementation; head/moment checks
use a separate fresh selection-and-training oracle.

\newpage
\section*{Whole-process memory and the decision it supports}
Fresh processes separately constructed each method and serviced four releases.
They loaded the same natural inputs and graph; they did not run oracle/audit
reconstructions. Joint span used its reduced decoder. Each number is one
observed high-water mark on this host, not a replicated performance estimate.
\begin{center}\small
\begin{tabular}{lrr}\toprule
Method & Setup peak (MiB) & Final peak (MiB)\\\midrule
PEAK_ROWS
\bottomrule\end{tabular}\end{center}
The final peak includes the Python/numerical runtime, original inputs,
construction, requests and decoder workspace. Setup peaks must not be
subtracted and interpreted as exact allocation attribution. The 225.6-fold
array reduction is therefore not a 225.6-fold whole-process RAM claim.

\textbf{Recommended implementation.} Use joint-span coefficients for the
aggregate-state method. Keep dense coefficients as an audit reference and
signed-Gram factors as an explanatory ablation. Keep the compact payload
method as a mandatory baseline whenever eligible-row retention is allowed.
No speedup conclusion follows from these unreplicated runs.

\textbf{What is solved.} The summary no longer allocates one full ambient
Gram per coefficient. It is built directly from bounded per-key workspaces,
supports source/record substitution through the existing key algebra, and
can decode a compact current coefficient without an ambient $d\times d$
matrix. The source extension here is algebraic; this new numerical run
validates record deletion because the Civil preview lacks native source IDs.

\textbf{What is not proved.} This is not optimal physical storage, superiority
to payload retention, automatic full-state canonicality, numerical exactness,
privacy, or evidence of semantic NLP quality. A large corpus can still have
many eligible records and substantial key metadata. Construction preflight
caps coefficient arrays only; lifecycle caps must use the uniform envelope
and reserve repair, decode, serialization and external input workspaces.

\textbf{Reproducibility.} The companion package includes all three new
implementations, the exact design, per-checkpoint records, boundary results,
source hashes, independent audit and process measurements. Earlier execution
results remain intact. The code has no required network access during replay.
\newpage
'''
    def sci(v):
        mant,exp=f'{v:.2e}'.split('e')
        return f'${mant}\\times10^{{{int(exp)}}}$'
    header=header.replace('RATIO',f'{ratio:.1f}').replace('TABLE_ROWS','\n'.join(rows)).replace('PEAK_ROWS','\n'.join(peakrows))
    header=header.replace('HEADERR',sci(head)).replace('COMPACTERR',sci(compact)).replace('FULLERR',sci(full))
    fragment=(ROOT/'memory_theory_addendum.tex').read_text()
    text=header+fragment+'\n\\end{document}\n'
    tex=OUT/'counterfactual_curation_memory_repair.tex';tex.write_text(text)
    fonts=Path('/usr/share/texlive/texmf-dist/fonts');lm=Path('/usr/share/texmf/fonts')
    env=os.environ.copy()
    env.update(TEXINPUTS=f'.:{WORK}:/usr/share/texlive/texmf-dist/tex//:/usr/share/texmf/tex//:',TFMFONTS=f'{fonts}/tfm//:{lm}/tfm//:',TEXFONTMAPS=f'{WORK}:{fonts}/map//:{lm}/map//:',T1FONTS=f'{fonts}/type1//:{lm}/type1//:',ENCFONTS=f'{fonts}/enc//:{lm}/enc//:',VFFONTS=f'{fonts}/vf//:{lm}/vf//:',TEXFORMATS=f'{WORK}:')
    for step in (1,2):
        log=WORK/f'memory-compile-{step}.log'
        with log.open('w') as stream:
            p=subprocess.run(['pdftex',f'-fmt={WORK}/pdflatex.fmt','-interaction=nonstopmode','-halt-on-error',f'-output-directory={WORK}',str(tex)],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(log.read_text()[-6000:])
    pdf=OUT/'counterfactual_curation_memory_repair.pdf'
    shutil.copyfile(WORK/pdf.name,pdf)
    print(pdf)

if __name__=='__main__':main()
