import json, sys, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
for r in rows:
    if "njoints" in r: print("META", r)
lanes = ["lib_eager", "proto_level_eager", "proto_jump_eager", "lib_compile", "proto_level_compile", "proto_level_compile_ro"]
short = ["libE", "lvlE", "jmpE", "libC", "lvlC", "lvlRO"]
t = collections.defaultdict(dict); k = {}
for r in rows:
    if "lane" in r:
        t[(r["device"], r["model"], r["pass"], r["batch"])][r["lane"]] = r["median_ms"]
        if "kernels" in r: k[(r["model"], r["pass"], r["lane"])] = r["kernels"]
print("dev model pass B | " + " ".join(short))
for key, v in t.items():
    print(*key, "|", " ".join(f"{v[l]:.3g}" if l in v else "-" for l in lanes))
print("KERNELS b256", k)
