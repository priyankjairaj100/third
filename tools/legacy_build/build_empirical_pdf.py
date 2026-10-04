from pathlib import Path
import os, shutil, subprocess
root=Path('/workspace/scratch/35d4d4d8ba2c')
work=root/'tmp/pdfs'
fonts=Path('/usr/share/texlive/texmf-dist/fonts')
lm=Path('/usr/share/texmf/fonts')
mapfiles=[fonts/'map/dvips/amsfonts'/name for name in ['cm.map','cmextra.map','symbols.map','latxfont.map']]+[lm/'map/dvips/lm/lm.map']
(work/'pdftex.map').write_text('\n'.join(p.read_text() for p in mapfiles))
env=os.environ.copy()
env.update(TEXINPUTS=f".:{work}:/usr/share/texlive/texmf-dist/tex//:/usr/share/texmf/tex//:", TFMFONTS=f"{fonts}/tfm//:{lm}/tfm//:", TEXFONTMAPS=f"{work}:{fonts}/map//:{lm}/map//:", T1FONTS=f"{fonts}/type1//:{lm}/type1//:", ENCFONTS=f"{fonts}/enc//:{lm}/enc//:", VFFONTS=f"{fonts}/vf//:{lm}/vf//:", TEXFORMATS=f"{work}:")
for step in (1,2):
    logfile=work/f'empirical-compile-{step}.log'
    with logfile.open('w') as out:
        result=subprocess.run(['pdftex',f'-fmt={work}/pdflatex.fmt','-interaction=nonstopmode','-halt-on-error',f'-output-directory={work}',str(root/'output/pdf/counterfactual_curation_empirical_protocol.tex')],env=env,cwd=root,stdout=out,stderr=subprocess.STDOUT)
    if result.returncode:
        print(logfile.read_text()[-6000:])
        raise SystemExit(result.returncode)
    print(f'Pass {step} completed')
shutil.copy2(work/'counterfactual_curation_empirical_protocol.pdf',root/'output/pdf/counterfactual_curation_empirical_protocol.pdf')
print((work/'empirical-compile-2.log').read_text()[-2500:])
