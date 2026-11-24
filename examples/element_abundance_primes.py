import numpy as np
class sympy:
    isprime=staticmethod(lambda n: n>1 and all(n%q for q in range(2,int(n**.5)+1)))
# Lodders 2010 (arXiv:1010.2746) Table 6, solar system 4.56 Ga ago, atoms per 1e6 Si
T="""1 2.59E+10|2 2.51E+09|3 55.6|4 0.612|5 18.8|6 7.19E+06|7 2.12E+06|8 1.57E+07|9 804|10 3.29E+06|11 57700|12 1.03E+06|13 84600|14 1.00E+06|15 8300|16 4.21E+05|17 5170|18 92700|19 3760|20 60400|21 34.4|22 2470|23 286|24 13100|25 9220|26 8.48E+05|27 2350|28 49000|29 541|30 1300|31 36.6|32 115|33 6.10|34 67.5|35 10.7|36 55.8|37 7.23|38 23.3|39 4.63|40 10.8|41 0.780|42 2.55|44 1.78|45 0.370|46 1.36|47 0.489|48 1.57|49 0.178|50 3.60|51 0.313|52 4.69|53 1.10|54 5.46|55 0.371|56 4.47|57 0.457|58 1.18|59 0.172|60 0.856|62 0.267|63 0.10|64 0.360|65 0.06|66 0.404|67 0.09|68 0.262|69 0.04|70 0.256|71 0.0380|72 0.156|73 0.0210|74 0.137|75 0.0581|76 0.678|77 0.672|78 1.27|79 0.195|80 0.458|81 0.182|82 3.31|83 0.138|90 0.0440|92 0.0238"""
A={int(a):float(b) for a,b in (r.split() for r in T.split("|"))}
L={z:np.log10(v) for z,v in A.items()}
# Residual: log abundance minus the average of the two same-parity neighbours (Z-2, Z+2).
# That removes the steep downward trend AND the even/odd effect at once.
res={z:L[z]-(L[z-2]+L[z+2])/2 for z in L if z>=6 and z-2 in L and z+2 in L}
odd=[z for z in res if z%2]
pr=[z for z in odd if sympy.isprime(z)]; co=[z for z in odd if not sympy.isprime(z)]
diff=np.mean([res[z] for z in pr])-np.mean([res[z] for z in co])
rng=np.random.default_rng(1); vals=np.array([res[z] for z in odd]); k=len(pr)
perm=[]
for _ in range(100000):
    rng.shuffle(vals); perm.append(vals[:k].mean()-vals[k:].mean())
perm=np.array(perm); p=np.mean(np.abs(perm)>=abs(diff))
print(f"odd Z tested: {len(odd)} ({len(pr)} prime, {len(co)} composite)")
print(f"prime minus composite residual: {diff:+.3f} dex (factor {10**diff:.2f}), permutation p = {p:.2f}")
# Calibration: the same method must see the even/odd effect
eo={z:L[z]-(L[z-1]+L[z+1])/2 for z in L if z>=6 and z-1 in L and z+1 in L}
ev=np.mean([v for z,v in eo.items() if z%2==0]); od=np.mean([v for z,v in eo.items() if z%2])
print(f"calibration: even Z vs odd neighbours {ev:+.2f} dex, odd Z vs even neighbours {od:+.2f} dex  (Oddo-Harkins, factor {10**(ev-od):.1f})")
# multiples of 4 by mass? use Z multiple of 2 (alpha: Z even, A=2Z): C,O,Ne,Mg,Si,S,Ar,Ca
alpha=[6,8,10,12,14,16,18,20]
print("alpha-element residuals vs even neighbours:", {z:round(L[z]-(L[z-2]+L[z+2])/2,2) for z in alpha if z-2 in L and z+2 in L})
print("top 10 by abundance (Z):", sorted(A,key=A.get,reverse=True)[:10])
print()
for z in sorted(odd): print(f"Z={z:2} {'prime' if z in pr else 'comp '} residual {res[z]:+.2f}")
def test(zs,label):
    p_=[z for z in zs if z in pr]; c_=[z for z in zs if z in co]
    v=np.array([res[z] for z in zs]); k=len(p_)
    d=np.mean([res[z] for z in p_])-np.mean([res[z] for z in c_])
    dm=np.median([res[z] for z in p_])-np.median([res[z] for z in c_])
    r=np.random.default_rng(2); P=[]
    for _ in range(50000):
        r.shuffle(v); P.append(v[:k].mean()-v[k:].mean())
    print(f"{label:40} n={len(zs):2}  mean diff {d:+.2f}  median diff {dm:+.2f}  p={np.mean(np.abs(P)>=abs(d)):.2f}")
test(odd,"all odd Z >= 7")
test([z for z in odd if z>=11],"odd Z >= 11 (drop Li-Be-B neighbours)")
test([z for z in odd if z>=31],"odd Z >= 31 (beyond the iron peak)")
