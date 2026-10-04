#!/usr/bin/env python3
"""Build the strict-mode research note from executed result records."""
from pathlib import Path
from fractions import Fraction
import json, os, shutil, statistics, subprocess

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=PROJECT/'output/pdf'
WORK=PROJECT/'tmp/pdfs'

def sci(value):
    m,e=f'{float(value):.3e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'

def main():
    result=json.loads((ROOT/'canonical_results/summary.json').read_text())
    regular=[json.loads(s) for s in (ROOT/'canonical_results/checkpoints.jsonl').read_text().splitlines()]
    full=json.loads((ROOT/'canonical_results/full_deletion.json').read_text())
    assert result['status']=='passed' and len(regular)==40 and len(full)==100
    maximum=max(Fraction(r['certificate']['error_upper_bound_decimal']) for r in regular+full)
    memory=[];times=[]
    for case in result['cases']:
        d,c=case['dimension'],case['outputs'];a=case['initial_accounting']
        memory.append(f"{d} & {c} & {a['stored_rational_scalars']:,} & {a['canonical_serialized_bytes']/1e6:.3f} & {a['python_live_bytes_estimate']/1e6:.3f} & {a['maximum_numerator_bits']}/{a['maximum_denominator_bits']} \\\\")
        rows=[r for r in regular if (r['dimension'],r['outputs'])==(d,c)]
        med=lambda key:statistics.median(r[key] for r in rows)
        times.append(f"{d} & {c} & {case['initial_build_seconds_unreplicated']:.3f} & {med('repair_seconds_unreplicated'):.3f} & {med('certificate_seconds_unreplicated'):.3f} & {med('fresh_build_seconds_unreplicated'):.3f} \\\\")
    header=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,amssymb,amsthm,booktabs,array}
\usepackage{xcolor,enumitem,fancyhdr}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!55!black]{hyperref}
\definecolor{ink}{HTML}{19374C}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Counterfactual curation}
\fancyhead[R]{\small Strict exact mode}\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlist{nosep,leftmargin=1.5em}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\title{\color{ink}\textbf{Canonical Repair with Certified Numerical Releases}\\[5pt]
\Large Closing the two remaining implementation qualifications\\for counterfactual data curation}
\author{ACL 2027 research project}
\date{4 October 2026}
\begin{document}\maketitle

\textbf{Outcome.} A new strict mode now has both \emph{byte-canonical aggregate
checkpoints} and \emph{rigorous numerical head certificates}. It was implemented
and validated locally, using the same 100 natural Civil Comments records and
their original responses. The fast floating mode remains available with its
previous contract; this report gives a stronger, separately specified mode.

The decisive construction is a unique rational chart of the joint range of
each Gram/cross coefficient. Its pivot identity rows are implicit, eliminating
the dense-versus-compact mode choice. Exact coefficient updates remove
history-dependent rounding. An exact rational residual then checks a fast
floating head against the exact retained-data ridge objective.

\textbf{Executed evidence.} All \textbf{140 deletion checkpoints} matched fresh
exact retained-data rebuilds byte for byte. Ten additional final-set comparisons
matched batching, singleton order, reversed order and strict reload. Every
head passed the prescribed $10^{-10}$ tolerance; the largest rigorous
Frobenius error radius was MAXIMUM. Full deletion produced empty coefficients,
an exactly zero head and an exactly zero certificate bound.

\textbf{Measured price.} At $d=768,c=1$, the initial canonical checkpoint is
1,514,487 bytes, while the recursive Python live-size estimate is 11,053,348
bytes. Build took 1.869 seconds; median timed repair and certification were
0.919 and 0.638 seconds. These are distinct accounting layers and unreplicated
timers, not a process-memory bound or an end-to-end speed claim.

\textbf{Scope.} Exactness means exact aggregation of frozen FP32 features and
FP64 responses interpreted as binary rationals, with frozen curation decisions.
It does not mean bit equality to legacy ordered floating-point retraining.
The experiment uses lexical engineering features, not a semantic encoder or
an NLP utility evaluation. No synthetic empirical documents or labels and no
external computation were used.
\newpage

\section*{Evidence ledger and cost}
The curator uses 128-dimensional lexical hashing, cosine threshold 0.6 and
a stable ID-derived priority. Learner feature dimensions are 16, 64 and 768.
The seven-output panel uses all seven original toxicity fields. The regular
run has horizon eight and binary64 input $\lambda=0.01$; requests are the
existing R, U and A families. Deletion checkpoints are 1, 2, 4 and 8, so timed
repair batches have sizes 1, 1, 2 and 4. The full-deletion stress uses $d=16$.

This is a convenience preview, not a representative corpus sample. The
original acquisition note is included with the fixture and distinguishes
connector-rendered records from original corpus bytes.

\begin{center}\small
\begin{tabular}{rrr rrr}\toprule
$d$ & $c$ & Rational entries & JSON MB & Python MB & Max bits N/D\\\midrule
MEMORY_ROWS
\bottomrule\end{tabular}\end{center}
MB means $10^6$ bytes. JSON includes the canonical logical state. Python live
size is a recursive estimate including rational objects, integer limbs and
metadata; it excludes external inputs, process/runtime overhead, allocator
slack and transient repair/serialization/native workspaces. At $d=768,c=1$,
the numerator/denominator integer payload alone is 184,525 bytes. Neither that
payload nor JSON size should be presented as actual total RAM. No exact-mode
process-peak experiment is claimed here.

\begin{center}\small
\begin{tabular}{rrrrrr}\toprule
$d$ & $c$ & Build s & Median repair s & Median cert. s & Median fresh s\\\midrule
TIME_ROWS
\bottomrule\end{tabular}\end{center}
Repair, fresh construction and certification timers are separate. They do
not include a complete service transaction; in particular some loading,
serialization and numerical-candidate work are outside those timers. The
certification cost can consume the apparent repair advantage. These local
observations do not establish a speedup or corpus-scale throughput.

\begin{itemize}
\item Forty regular checkpoints span the three scalar panels and one
seven-output panel. Every oracle rescored the retained graph independently,
retained stable public IDs, rebuilt exact coefficients and compared all bytes.
\item One hundred additional sequential deletion checkpoints also compared
fresh exact bytes and checked certificates. The last result is exactly empty.
\item Ten comparisons exercised batch/singleton/reversed histories and strict
reload. Same-runtime numerical heads also agreed; portable floating-head
bytes are not part of the theorem.
\item Duplicate, unknown, over-budget and retried requests were rejected
without changing input bytes. The service additionally accepted the required
tolerance and rejected an unattainable zero tolerance without release.
\end{itemize}
All five central source hashes in the run ledger and the natural-data hash
were independently rechecked after execution. The archive additionally hashes
the curator, dependency helpers and compiled native binary.
\newpage

\section*{Algorithm and release boundary}
For each current coefficient, the strict implementation stores its exact
joint-range chart, pivot Gram/cross cores and count. The authoritative state
is one canonical JSON byte string containing sorted keys, normalized rational
numbers, sorted alive IDs, fixed dimensions, public universe and remaining
horizon. Solver output, request history and previous factor charts are absent.

\begin{enumerate}
\item \textbf{Initialize once.} Freeze the numerical inputs, build exact
coefficient endpoints in blocks of at most $\min(d,128)$ contributors, and
canonicalize each block/merge. No family of dense ambient Grams is built first.
\item \textbf{Repair.} Validate the request; substitute the deleted IDs into
keys, prune degrees above the decreased horizon, merge collisions in rational
arithmetic, and remove exactly zero coefficients. Commit only after the staged
new state succeeds. The current reference scans/sorts all keys.
\item \textbf{Propose.} Reconstruct the current coefficient's rational basis,
convert for a numerical QR-based reduced solve, and obtain a finite candidate
head. The rational basis is not orthonormal, so the old reduced equation
cannot be applied directly.
\item \textbf{Certify.} Prove the current Gram core is PSD by an exact rational
Schur-complement test and evaluate the full candidate residual exactly.
Check the requested error tolerance as a rational inequality.
\item \textbf{Release.} Return canonical updated bytes, the candidate head
and its certificate only if the check passes. The pure bytes-in/bytes-out
service leaves its input unchanged on failure.
\end{enumerate}

The certificate covers numerical fidelity to the supplied exact target. A
valid retained-data target is established by the repair theorem and verified
against fresh rebuilds in this run. A structurally canonical but maliciously
altered state is not automatically authentic. Dataset/configuration hashes
and provenance belong in the surrounding execution manifest.

\textbf{Independent checks.} The numerical certificate passed 243 exact
rational oracle comparisons and 1,984 upward-square-root checks. The chart
passed 66 independent ambient dense-RREF comparisons plus three large-integer
cases. Twenty native/Python algebra fixtures and 50 certificate module checks
also passed. Six malformed checkpoint variants reject even under
\texttt{python -O}; that audit found and fixed a Boolean/integer alias before
the frozen full run. These small algebra fixtures are software tests, not
synthetic empirical datasets.

\textbf{Implementation trust and portability.} Fast exact canonicalization
uses the installed GMP 6.3.0 runtime through a local Linux LP64 helper.
Its explicit ABI declarations have static checks, startup arithmetic checks
and independent large-integer comparisons. Source and binary are included.
An exact Python \texttt{Fraction} fallback avoids that platform dependency
at additional runtime cost. Both produce the same canonical representation.
\newpage

\section{Fixed-blocker curation and the repair polynomial}
This note concerns a specified fixed-blocker selector. For record $v$, let
$o(v)$ be its deletion-unit owner and let $B(v)$ be its distinct blocker-owner
set. A record is selected precisely when its owner survives and all its blocker
owners are deleted. In the engineering curator, blockers are all earlier
priority neighbors above the frozen similarity threshold. This is not a
connected-component representative rule or a universal description of
semantic deduplicators.

Let $t_u\in\{0,1\}$ indicate deletion of unit $u$. With frozen feature $z_v$
and response $y_v$, write
\[
s_v=(z_vz_v^\top,z_vy_v^\top,1),\qquad
P(t)=\sum_v s_v(1-t_{o(v)})\prod_{u\in B(v)}t_u.
\]
If $o(v)\in B(v)$, its Boolean contribution is zero. Otherwise each record
has a positive coefficient endpoint $B(v)$ and a negative endpoint
$B(v)\cup\{o(v)\}$. At horizon $h$, retain endpoints of degree at most $h$.
For every request of at most $h$ deletions, all omitted monomials evaluate to
zero, so this truncated polynomial gives exact selected-set moments.

After a request $F$, substitute $t_u=1$ for $u\in F$, giving the key map
$J\mapsto J\setminus F$, and reduce the horizon to $h-|F|$. A discarded
degree-$k$ term has new degree at least $k-|F|>h-|F|$ and cannot reenter.
A record whose owner is deleted has identical positive and negative endpoints,
which cancel. A surviving record has blocker set $B(v)\setminus F$.
Consequently repaired coefficients equal fresh retained-data coefficients.

Define $E_h$ as the number of current records with own owner absent from their
blocker set and at most $h$ distinct blocker owners. Let $m_J$ count nonzero
record contributions incident to coefficient $J$. Then
\[
K\le 2E_h,\quad \sum_Jm_J\le 2E_h,\quad
\operatorname{rank}([G_J\ H_J])\le m_J.
\]
These are the rank and key bounds used below. They remain true for repeated
source owners, signed cancellations, horizon zero and an empty state. In a
fresh retained rebuild, blocker row positions must be remapped while original
public owner IDs and the public universe remain fixed.

An aggregate can encode recoverable facts about individual records. The
contract is absence of a separate payload table and of repair-time raw-row
reads, not cryptographic privacy. Similarly, semantic curation itself requires
separate empirical validation. The proofs below hold for the stated selector.
\newpage
'''
    header=header.replace('MAXIMUM',sci(maximum)).replace('MEMORY_ROWS','\n'.join(memory)).replace('TIME_ROWS','\n'.join(times))
    footer=r'''
\clearpage
\section*{Paper claim and reproducibility}
The supportable claim is: \emph{for fixed-blocker curation with frozen binary
features and ridge sufficient statistics, there is an exact finite-horizon
coefficient repair representation with unique history-independent checkpoint
bytes, linear current-eligibility rational-entry count, and an exact-residual
certificate for approximate numerical releases.} The bit-growth and scan/merge
work bounds are explicit. The present implementation is a constructive
correctness result, accompanied by natural-text engineering checks.

The chart and residual/coercivity argument use classical algebra and verified
numerics. Novelty must be assessed for the curation-aware combination, deletion
semantics and empirical consequence; it should not be attributed to RREF,
rational arithmetic or residual bounds in isolation. For classical verified
numerics, see S. M. Rump, \emph{Verification methods: rigorous results using
floating-point arithmetic}, Acta Numerica 19 (2010), 287--449,
\href{https://doi.org/10.1017/S096249291000005X}{doi:10.1017/S096249291000005X}.
The native helper documents the consulted official GMP interface references.

The companion archive includes executable code, the CC0 natural-data fixture,
all strict-mode result records, exact checkpoint and release certificate,
independent audit reports, native code, pinned requirements, complete
report source and integrity manifest. The strict-mode README gives the replay
commands. Replay requires no network and no external compute. Prior floating-mode
results are not overwritten by this report.

Before rerunning, preserve \texttt{canonical\_results/}, because the harness
replaces those output records. From the extracted \texttt{empirical\_execution}
directory run:
\begin{quote}\small\ttfamily
OPENBLAS\_NUM\_THREADS=1 OMP\_NUM\_THREADS=1\\
MKL\_NUM\_THREADS=1 python3 run\_canonical\_validation.py
\end{quote}
The environment assignments belong on the same shell command line (or may be
exported first). The algebra and certificate audit scripts can be run
independently. The included complete \LaTeX{} source can be rendered using
standard PDFLaTeX with its declared packages; the builder also supports the
already initialized local TeX format used for this execution.
\end{document}
'''
    tex=OUT/'counterfactual_curation_strict_mode.tex'
    tex.write_text(header+(ROOT/'canonical_theory_addendum.tex').read_text()+footer)
    fonts=Path('/usr/share/texlive/texmf-dist/fonts');lm=Path('/usr/share/texmf/fonts')
    env=os.environ.copy()
    env.update(TEXINPUTS=f'.:{WORK}:/usr/share/texlive/texmf-dist/tex//:/usr/share/texmf/tex//:',TFMFONTS=f'{fonts}/tfm//:{lm}/tfm//:',TEXFONTMAPS=f'{WORK}:{fonts}/map//:{lm}/map//:',T1FONTS=f'{fonts}/type1//:{lm}/type1//:',ENCFONTS=f'{fonts}/enc//:{lm}/enc//:',VFFONTS=f'{fonts}/vf//:{lm}/vf//:',TEXFORMATS=f'{WORK}:')
    for step in (1,2):
        log=WORK/f'strict-compile-{step}.log'
        with log.open('w') as stream:
            p=subprocess.run(['pdftex',f'-fmt={WORK}/pdflatex.fmt','-interaction=nonstopmode','-halt-on-error',f'-output-directory={WORK}',str(tex)],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(log.read_text()[-6000:])
    pdf=OUT/'counterfactual_curation_strict_mode.pdf'
    shutil.copyfile(WORK/pdf.name,pdf)
    print(pdf)

if __name__=='__main__':main()
