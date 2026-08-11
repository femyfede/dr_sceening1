import os, sys, time, faulthandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "outputs_multidataset", "multidataset"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools", "rrwnet"))

import yaml
import pipeline

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.txt")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

with open(os.path.join(ROOT, "outputs_multidataset", "multidataset", "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg_dir = os.path.join(ROOT, "outputs_multidataset", "multidataset")
for key in ("project_scripts", "rrwnet_module_dir", "rrwnet_weights", "outputs"):
    p = cfg.get(key)
    if p:
        cfg[key] = os.path.abspath(os.path.join(cfg_dir, p))
pipeline.ensure_paths(cfg)
device = pipeline.pick_device(cfg["device"])
model = pipeline.load_rrwnet(cfg, device)

base = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
                    "sehastrajits", "fundus-aptosddridirdeyepacsmessidor",
                    "versions", "1", "split_dataset")
times = []
imgs = []
for cls in range(5):
    d = os.path.join(base, "train", str(cls))
    imgs += [(os.path.join(d, f), cls) for f in os.listdir(d)[:2]]
for i, (path, cls) in enumerate(imgs):
    row = {"image_id": f"bench_{i}", "dataset": "combined", "split": "train",
           "label_6": cls, "path": path}
    t0 = time.time()
    res, err = pipeline.process_one(cfg, model, row)
    dt = time.time() - t0
    times.append(dt)
    log(f"img={i} cls={cls} {dt:.2f}s err={err}")
mean = sum(times) / len(times)
log(f"mean={mean:.2f}s -> {len(times)/sum(times)*3600:.0f} img/hr")
log("DONE")
