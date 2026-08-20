"""serve a live read-only architecture search dashboard."""

import argparse
import json
import math
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


dashboard = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>speck search</title>
<style>
:root{color-scheme:dark;--bg:#0b0d10;--panel:#14181d;--line:#29313a;--ink:#e8edf2;--muted:#87939f;--cyan:#58d6c7;--amber:#f0b45a;--red:#e66b6b;--blue:#6d9fff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}header{padding:28px max(24px,4vw) 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:end}h1,h2{margin:0;font-weight:500}h1{font-size:24px;letter-spacing:-1px}h2{font-size:14px;color:var(--muted);margin-bottom:14px}.live{color:var(--cyan)}main{padding:24px max(24px,4vw) 60px;display:grid;gap:18px}.cards{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}.card{padding:15px}.card b{display:block;font-size:23px;font-weight:500}.card span{color:var(--muted);font-size:11px}.panel{padding:18px;overflow:auto}.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr);gap:18px}.progress{height:8px;background:#222831;border-radius:9px;overflow:hidden}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan))}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:10px;font-weight:500;position:sticky;top:0;background:var(--panel)}tr[data-id]{cursor:pointer}tr[data-id]:hover{background:#1b2229}.tag{border:1px solid var(--line);padding:2px 6px;border-radius:10px;font-size:10px}.frontier{color:var(--cyan)}.failed{color:var(--red)}.running{color:var(--amber)}select{background:#0f1317;color:var(--ink);border:1px solid var(--line);padding:6px;border-radius:5px;max-width:100%}svg{width:100%;height:250px;overflow:visible}.layers{display:grid;gap:5px;margin:12px 0}.layer{display:grid;grid-template-columns:30px minmax(70px,1fr) 130px;gap:8px;align-items:center}.bar{height:18px;background:linear-gradient(90deg,#315d88,var(--blue));border-radius:2px;min-width:4px}.bar.attn{background:linear-gradient(90deg,#2f7b70,var(--cyan))}.small{color:var(--muted);font-size:11px}pre{white-space:pre-wrap;word-break:break-word;color:#aeb9c3;font-size:11px;max-height:340px;overflow:auto}details{margin-top:12px}summary{cursor:pointer;color:var(--muted)}.empty{color:var(--muted);padding:30px 0;text-align:center}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}header{align-items:start;gap:12px;flex-direction:column}}
</style>
<header><div><h1>speck architecture search</h1><div class="small" id="study"></div></div><div class="live" id="clock">connecting</div></header>
<main>
<section class="cards" id="cards"></section>
<section class="panel"><h2>study progress</h2><div class="progress"><i id="progress"></i></div><div class="small" id="progress-label" style="margin-top:8px"></div></section>
<section class="grid">
  <div class="panel"><h2>best observed objective</h2><select id="objective"></select><svg id="trend" viewBox="0 0 800 250"></svg></div>
  <div class="panel"><h2>selected architecture</h2><div id="detail" class="empty">select a candidate below</div></div>
</section>
<section class="panel"><h2>pareto frontier</h2><div id="frontier"></div></section>
<section class="panel"><h2>candidate history</h2><div id="candidates"></div></section>
<section class="panel"><details><summary>study configuration and provenance</summary><pre id="raw-study"></pre></details></section>
</main>
<script>
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'—':typeof x==='number'?(Math.abs(x)>=1e6?(x/1e6).toFixed(2)+'m':x.toFixed(x<10?4:2)):x;
let state,selected,objective;
function table(rows,cols){if(!rows.length)return'<div class="empty">no data yet</div>';return'<table><thead><tr>'+cols.map(c=>'<th>'+esc(c[0])+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr data-id="'+r.id+'">'+cols.map(c=>'<td>'+c[1](r)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function path(values,all,w=800,h=220,p=28){if(!values.length)return'';let xs=all.map(x=>x[0]),ys=all.map(x=>x[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),sx=x=>p+(x-xmin)/(xmax-xmin||1)*(w-2*p),sy=y=>p+(ymax-y)/(ymax-ymin||1)*(h-2*p);return values.map((v,i)=>(i?'L':'M')+sx(v[0]).toFixed(1)+' '+sy(v[1]).toFixed(1)).join(' ')}
function plot(svg,sets){let all=sets.flatMap(s=>s.values),ys=all.map(x=>x[1]);if(!all.length){svg.innerHTML='<text x="400" y="120" text-anchor="middle" fill="#87939f">waiting for completed candidates</text>';return}let ymin=Math.min(...ys),ymax=Math.max(...ys);svg.innerHTML='<line x1="28" y1="220" x2="772" y2="220" stroke="#29313a"/><text x="28" y="242" fill="#87939f">tokens / candidate</text><text x="6" y="22" fill="#87939f">'+esc(fmt(ymax))+'</text><text x="6" y="218" fill="#87939f">'+esc(fmt(ymin))+'</text>'+sets.map(s=>'<path d="'+path(s.values,all)+'" fill="none" stroke="'+s.color+'" stroke-width="2"/><text x="650" y="'+(18+sets.indexOf(s)*16)+'" fill="'+s.color+'">'+esc(s.name)+'</text>').join('')}
function drawTrend(){objective=$('objective').value;let best=Infinity,points=[];(state.candidates||[]).filter(c=>c.objectives&&c.objectives[objective]!=null).sort((a,b)=>a.id-b.id).forEach(c=>{best=Math.min(best,c.objectives[objective]);points.push([c.id,best])});plot($('trend'),[{name:objective,color:'#58d6c7',values:points}])}
function render(s){state=s;let n=s.counts,max=s.config.max_evaluations,done=(n.completed||0)+(n.failed||0);$('study').textContent=s.path;$('clock').textContent=s.status+' · updated '+new Date().toLocaleTimeString();$('cards').innerHTML=[['completed',n.completed||0],['running',n.running||0],['pending',n.pending||0],['failed',n.failed||0],['frontier',s.frontier.length]].map(x=>'<div class="card"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');$('progress').style.width=(100*done/max).toFixed(1)+'%';$('progress-label').textContent=done+' / '+max+' terminal candidates · '+(100*done/max).toFixed(1)+'%';let names=s.objectives||[];let old=$('objective').value;$('objective').innerHTML=names.map(x=>'<option>'+esc(x)+'</option>').join('');$('objective').value=names.includes(old)?old:names[0]||'';drawTrend();let base=[['id',r=>'#'+r.id],['status',r=>'<span class="tag '+r.status+'">'+esc(r.status)+'</span>'],['mutation',r=>esc(r.mutation.operator)],['parent',r=>esc(r.parents.join(',')||'—')],['params',r=>fmt(r.parameters)],['nll',r=>fmt(r.objectives?.['quality.validation_nll'])],['rank',r=>fmt(r.pareto_rank)]];$('frontier').innerHTML=table(s.frontier,base.concat(names.map(name=>[name,r=>fmt(r.objectives?.[name])])));$('candidates').innerHTML=table([...s.candidates].reverse(),base.concat([['created',r=>esc((r.completed_at||r.started_at||r.created_at||'').slice(11,19))],['error',r=>esc(r.error||'')]]));document.querySelectorAll('tr[data-id]').forEach(r=>r.onclick=()=>loadCandidate(+r.dataset.id));$('raw-study').textContent=JSON.stringify({config:s.config,provenance:s.provenance},null,2)}
async function loadCandidate(id){selected=id;let d=await fetch('/api/candidate/'+id).then(r=>r.json()),layers=d.config.layers||[],mx=Math.max(...layers.map(x=>x.hidden_size),1),q=d.result?.quality;$('detail').innerHTML='<b>candidate #'+id+'</b> <span class="tag '+d.status+'">'+esc(d.status)+'</span><div class="small">'+esc(d.mutation.operator)+' · '+fmt(d.parameters)+' parameters · parents '+esc(d.parents.join(',')||'—')+'</div><div class="layers">'+layers.map((l,i)=>'<div class="layer"><span>'+i+'</span><div class="bar '+(l.num_key_value_heads!=null?'attn':'')+'" style="width:'+(100*l.hidden_size/mx)+'%"></div><span class="small">h'+l.hidden_size+' · f'+l.intermediate_size+' · '+(l.num_key_value_heads==null?'mlp':'kv'+l.num_key_value_heads)+'</span></div>').join('')+'</div><h2>loss curves</h2><svg id="loss" viewBox="0 0 800 250"></svg><details><summary>complete candidate record</summary><pre>'+esc(JSON.stringify(d,null,2))+'</pre></details>';let train=(q?.train_curve||[]).map(x=>[x.tokens,x.loss]),val=(q?.validation_curve||[]).map(x=>[x.tokens,x.loss]);plot($('loss'),[{name:'train',color:'#6d9fff',values:train},{name:'validation',color:'#f0b45a',values:val}])}
async function refresh(){try{render(await fetch('/api/state',{cache:'no-store'}).then(r=>r.json()))}catch(e){$('clock').textContent='disconnected'}setTimeout(refresh,3000)}
$('objective').onchange=drawTrend;refresh();
</script>
</html>"""


def _connect(path):
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=1)


def _load(value) -> Any:
    return json.loads(value) if value else None


def _clean(value) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def snapshot(path) -> dict[str, Any]:
    connection = _connect(path)
    connection.row_factory = sqlite3.Row
    try:
        study = connection.execute("select * from study where id = 1").fetchone()
        rows = connection.execute("select * from candidates order by id").fetchall()
        parent_rows = connection.execute("select * from parents order by candidate_id").fetchall()
    finally:
        connection.close()
    parents = {}
    for row in parent_rows:
        parents.setdefault(row["candidate_id"], []).append(row["parent_id"])
    candidates = []
    counts = {}
    objectives = set()
    for row in rows:
        result = _load(row["result_json"])
        values = result.get("objectives", {}) if result else {}
        objectives.update(values)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        config = _load(row["architecture_json"])
        candidates.append({
            "id": row["id"],
            "status": row["status"],
            "mutation": _load(row["mutation_json"]),
            "parents": parents.get(row["id"], []),
            "parameters": result.get("model", {}).get("parameters") if result else None,
            "layers": len(config.get("layers", [])),
            "objectives": values,
            "in_population": bool(row["in_population"]),
            "is_frontier": bool(row["is_frontier"]),
            "pareto_rank": row["pareto_rank"],
            "crowding": row["crowding"],
            "novelty": row["novelty"],
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        })
    return _clean({
        "path": str(Path(path).parent),
        "status": study["status"],
        "config": _load(study["config_json"]),
        "provenance": _load(study["provenance_json"]),
        "counts": counts,
        "objectives": sorted(objectives),
        "candidates": candidates,
        "frontier": [candidate for candidate in candidates if candidate["is_frontier"]],
    })


def candidate(path, candidate_id) -> dict[str, Any]:
    connection = _connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "select * from candidates where id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        parents = [
            item["parent_id"]
            for item in connection.execute(
                "select parent_id from parents where candidate_id = ? order by parent_id",
                (candidate_id,),
            )
        ]
        attempts = [
            dict(item)
            for item in connection.execute(
                "select * from attempts where candidate_id = ? order by id", (candidate_id,)
            )
        ]
    finally:
        connection.close()
    value = dict(row)
    value.update(
        config=_load(value.pop("architecture_json")),
        mutation=_load(value.pop("mutation_json")),
        repairs=_load(value.pop("repairs_json")),
        result=_load(value.pop("result_json")),
        parents=parents,
        attempts=attempts,
    )
    value["parameters"] = (value["result"] or {}).get("model", {}).get("parameters")
    return _clean(value)


def handler(database):
    class Handler(BaseHTTPRequestHandler):
        def send(self, value, content_type="application/json", status=200):
            data = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("content-type", f"{content_type}; charset=utf-8")
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self.send(dashboard.encode(), "text/html")
                elif path == "/api/state":
                    self.send(snapshot(database))
                elif path.startswith("/api/candidate/"):
                    self.send(candidate(database, int(path.rsplit("/", 1)[-1])))
                else:
                    self.send({"error": "not found"}, status=404)
            except (KeyError, ValueError):
                self.send({"error": "not found"}, status=404)
            except sqlite3.Error as error:
                self.send({"error": str(error)}, status=503)

        def log_message(self, format, *args):
            pass

    return Handler


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = arguments()
    value = Path(args.study)
    if value.suffix == ".sqlite3":
        database = value
    else:
        root = Path(os.environ.get("speck_base_dir", Path.home() / ".cache" / "speck"))
        database = root / "search" / args.study / "study.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"search study not found: {database}")
    server = ThreadingHTTPServer((args.host, args.port), handler(database))
    print(f"dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
