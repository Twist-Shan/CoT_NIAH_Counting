"""Check archive-only declarations and explicit seed ranges before freezing IDs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tick = time.monotonic()
    raw_audit = a.audit.read_bytes()
    audit = json.loads(raw_audit)
    assert audit["status"] == "PASS_UNPACKED_INVENTORY"
    candidates = set(audit["candidate_seeds"])
    pattern = re.compile(rb'(?<!\d)(?:' + b'|'.join(str(v).encode() for v in sorted(candidates)) + rb')(?!\d)')
    checked, conflicts, ranges, errors = [], [], [], []

    def inspect(raw, label):
        hits = sorted({int(m[0]) for m in pattern.finditer(raw)})
        if hits:
            conflicts.append({"file":label,"seeds":hits})
        # Only JSON objects with actual start/end or count declarations need
        # semantic range expansion. Arrays of explicitly listed seeds were
        # already covered by the inventory and prospective integer scan.
        if b"seed_start" not in raw and b"seed_min" not in raw:
            return
        def walk(obj):
            if isinstance(obj,dict):
                lo = obj.get("seed_start",obj.get("seed_min"))
                hi = obj.get("seed_end",obj.get("seed_max",obj.get("max_seed")))
                stop = obj.get("seed_stop")
                count = obj.get("num_seeds",obj.get("seed_count",obj.get("max_seeds")))
                if isinstance(lo,int):
                    if hi is None and isinstance(stop,int): hi = stop-1
                    if hi is None and isinstance(count,int): hi = lo+count-1
                    if isinstance(hi,int) and hi >= lo:
                        hits = sorted(v for v in candidates if lo <= v <= hi)
                        ranges.append({"file":label,"start":lo,"end":hi,"conflicts":hits})
                        if hits: conflicts.append(ranges[-1])
                for value in obj.values():
                    if isinstance(value,(dict,list)): walk(value)
            elif isinstance(obj,list):
                for value in obj:
                    if isinstance(value,(dict,list)): walk(value)
        try:
            if label.endswith(".jsonl"):
                for line in raw.split(b"\n"):
                    if line.strip() and (b"seed_start" in line or b"seed_min" in line): walk(json.loads(line))
            elif label.endswith(".json"):
                walk(json.loads(raw))
        except Exception as e:
            errors.append({"file":label,"error":repr(e)})

    for item in audit["files"]:
        # Avoid rereading all large raw outputs: range declarations are JSON
        # configuration/manifest metadata; JSONL source seeds are explicit.
        if item["path"].endswith(".json"):
            raw = Path(item["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                errors.append({"file":item["path"],"error":"changed_since_inventory"})
            inspect(raw,item["path"])
    for filename in audit["archives_not_opened"]:
        path=Path(filename)
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
        members=[]
        def consume(name, stream):
            raw=stream.read()
            inspect(raw,filename+"::"+name)
            members.append({"path":name,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
        if filename.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    if not entry.is_dir() and entry.filename.endswith((".json",".jsonl",".csv",".tsv")):
                        with archive.open(entry) as stream: consume(entry.filename,stream)
        else:
            with tarfile.open(path,"r|gz") as archive:
                for entry in archive:
                    if entry.isfile() and entry.name.endswith((".json",".jsonl",".csv",".tsv")):
                        with archive.extractfile(entry) as stream: consume(entry.name,stream)
        checked.append({"path":filename,"sha256":digest.hexdigest(),"metadata_members":members})
        print(json.dumps({"archive":filename,"metadata_members":len(members),"seconds":time.monotonic()-tick}),flush=True)
    result={"status":"PASS" if not conflicts and not errors else "FAIL",
            "history_audit_sha256":hashlib.sha256(raw_audit).hexdigest(),"candidate_seeds":sorted(candidates),
            "ranges":ranges,"archives":checked,"conflicts":conflicts,"errors":errors,"seconds":time.monotonic()-tick}
    with a.output.open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ("status","conflicts","errors","seconds")}),flush=True)
    if result["status"] != "PASS": raise RuntimeError("History review did not pass")


if __name__ == "__main__":
    main()
