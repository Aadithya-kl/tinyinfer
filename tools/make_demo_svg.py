# Animated terminal SVG built from REAL output recorded on the RTX 3050 Laptop GPU.
import html, sys
W=900; LH=19; PAD=18; TOP=44; CW=7.83  # approx char width at 13px mono
static = len(sys.argv) > 1 and sys.argv[1] == "static"
P="#7ee787"; C="#e6edf3"; D="#8b949e"; G="#3fb950"; Y="#d29922"; B="#79c0ff"
# (text, color, is_command, delay_after_previous)
L=[
 ("PS> tinyinfer.exe -m models/tinyllama-q4.tinf --bench 256", C, True, 0.6),
 ("tinyinfer 0.3.0 | tinyllama-q4.tinf | dim 2048, 22 layers, 32/4 heads, q4 (g64)", D, False, 1.6),
 ("GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (sm_86, 20 SMs, 6.0 GiB, ~168 GB/s)", D, False, 0.25),
 ("engine cuda+graph loaded in 0.67 s | weights 649 MiB + KV cache 44 MiB | split-K GQA", D, False, 0.45),
 ("gemv autotune (rows/warp): 2560x2048:2 2048x2048:1 11264x2048:1 2048x5632:2 32000x2048:1", D, False, 0.35),
 ("decode           : 256 tokens in 1.088 s", C, False, 1.2),
 ("throughput       : 235.2 tok/s  (4.25 ms/token)", G, False, 0.3),
 ("weight bandwidth : 129.3 GB/s  (77% of 168 GB/s peak)", G, False, 0.3),
 ("", C, False, 0.3),
 ("PS> tinyinfer.exe -m models/tinyllama-q4.tinf --verify 16", C, True, 0.8),
 ("  pos    token    max|diff|          rel   argmax", D, False, 1.4),
 ("    0        1    1.699e-05    3.004e-06       ok", D, False, 0.12),
 ("    1      450    7.085e-04    4.924e-05       ok", D, False, 0.12),
 ("    2     4996    2.110e-03    1.361e-04       ok", D, False, 0.12),
 ("  ...", D, False, 0.12),
 ("   15     7698    1.601e-03    9.550e-05       ok", D, False, 0.12),
 ("worst relative error 2.521e-04, argmax mismatches 0/16 -> PASS   (GPU vs CPU reference)", G, False, 0.5),
]
T_TYPE=1.4   # seconds to type a command
t=0; times=[]
for text,col,cmd,dly in L:
    t+=dly; times.append(t)
    if cmd: t+=T_TYPE
TOTAL=t+4.0
H=TOP+len(L)*LH+PAD
out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="tinyinfer benchmark on an RTX 3050 Laptop GPU: 235.2 tokens per second">']
css=[".t{font-family:'Cascadia Mono',Consolas,'SFMono-Regular',Menlo,'DejaVu Sans Mono',monospace;font-size:13px;white-space:pre}"]
body=[]
for i,(text,col,cmd,_) in enumerate(L):
    y=TOP+i*LH; st=times[i]/TOTAL*100; tx=html.escape(text)
    if static:
        body.append(f'<text class="t" xml:space="preserve" x="{PAD}" y="{y}" fill="{col}">{tx}</text>'); continue
    if cmd:
        prompt, rest = text[:4], text[4:]
        en=(times[i]+T_TYPE)/TOTAL*100
        w=len(rest)*CW+4
        # a background-coloured cover slides right in character steps ("typing"), then fades out
        css.append(f"@keyframes c{i}{{0%,{st:.2f}%{{transform:translateX(0);opacity:1}}"
                   f"{en:.2f}%{{transform:translateX({w:.0f}px);opacity:1}}{en+0.01:.2f}%,97%{{transform:translateX({w:.0f}px);opacity:0}}"
                   f"100%{{transform:translateX(0);opacity:1}}}}")
        css.append(f"@keyframes p{i}{{0%,{st:.2f}%{{opacity:0}}{st+0.01:.2f}%,97%{{opacity:1}}100%{{opacity:0}}}}")
        css.append(f".c{i}{{animation:c{i} {TOTAL:.1f}s infinite;animation-timing-function:steps({len(rest)},end)}} .p{i}{{animation:p{i} {TOTAL:.1f}s infinite}}")
        x2=PAD+4*CW
        body.append(f'<text class="t p{i}" xml:space="preserve" x="{PAD}" y="{y}" fill="{P}">PS&gt;</text>')
        body.append(f'<text class="t p{i}" xml:space="preserve" x="{x2:.1f}" y="{y}" fill="{col}">{html.escape(rest)}</text>')
        body.append(f'<rect class="c{i}" x="{x2-1:.1f}" y="{y-14}" width="{w:.0f}" height="{LH}" fill="#0d1117"/>')
    else:
        css.append(f"@keyframes l{i}{{0%,{st:.2f}%{{opacity:0}}{st+0.01:.2f}%,97%{{opacity:1}}100%{{opacity:0}}}}")
        body.append(f'<text class="t" xml:space="preserve" x="{PAD}" y="{y}" fill="{col}" style="animation:l{i} {TOTAL:.1f}s infinite">{tx}</text>')
out.append("<style>"+"".join(css)+"</style>")
out.append(f'<rect width="{W}" height="{H}" rx="10" fill="#0d1117" stroke="#30363d"/>')
out.append(f'<rect width="{W}" height="28" rx="10" fill="#161b22"/><rect y="18" width="{W}" height="10" fill="#161b22"/>')
for k,c in enumerate(["#ff5f56","#ffbd2e","#27c93f"]): out.append(f'<circle cx="{18+k*18}" cy="14" r="5.5" fill="{c}"/>')
out.append(f'<text x="{W/2}" y="18" text-anchor="middle" fill="#8b949e" style="font-family:Segoe UI,Helvetica,Arial,sans-serif;font-size:12px">tinyinfer — RTX 3050 6GB Laptop GPU (recorded output)</text>')
out+=body; out.append("</svg>")
print("\n".join(out))
