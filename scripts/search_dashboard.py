"""serve a live read-only architecture search dashboard."""

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


_fallback_dashboard = r"""<!doctype html>
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
  <div class="panel"><h2>selected architecture</h2><div id="detail" class="empty">select an architecture below</div></div>
</section>
<section class="panel"><h2>pareto frontier</h2><div id="frontier"></div></section>
<section class="panel"><h2>architecture history</h2><div id="candidates"></div></section>
<section class="panel"><details><summary>study configuration and provenance</summary><pre id="raw-study"></pre></details></section>
</main>
<script>
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'—':typeof x==='number'?(Math.abs(x)>=1e6?(x/1e6).toFixed(2)+'m':x.toFixed(x<10?4:2)):x;
let state,selected,objective;
function table(rows,cols){if(!rows.length)return'<div class="empty">no data yet</div>';return'<table><thead><tr>'+cols.map(c=>'<th>'+esc(c[0])+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr data-id="'+r.id+'">'+cols.map(c=>'<td>'+c[1](r)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function path(values,all,w=800,h=220,p=28){if(!values.length)return'';let xs=all.map(x=>x[0]),ys=all.map(x=>x[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),sx=x=>p+(x-xmin)/(xmax-xmin||1)*(w-2*p),sy=y=>p+(ymax-y)/(ymax-ymin||1)*(h-2*p);return values.map((v,i)=>(i?'L':'M')+sx(v[0]).toFixed(1)+' '+sy(v[1]).toFixed(1)).join(' ')}
function plot(svg,sets,label){let all=sets.flatMap(s=>s.values),ys=all.map(x=>x[1]);if(!all.length){svg.innerHTML='<text x="400" y="120" text-anchor="middle" fill="#87939f">waiting for completed architectures</text>';return}let ymin=Math.min(...ys),ymax=Math.max(...ys);svg.innerHTML='<line x1="28" y1="220" x2="772" y2="220" stroke="#29313a"/><text x="28" y="242" fill="#87939f">'+esc(label)+'</text><text x="6" y="22" fill="#87939f">'+esc(fmt(ymax))+'</text><text x="6" y="218" fill="#87939f">'+esc(fmt(ymin))+'</text>'+sets.map(s=>'<path d="'+path(s.values,all)+'" fill="none" stroke="'+s.color+'" stroke-width="2"/><text x="650" y="'+(18+sets.indexOf(s)*16)+'" fill="'+s.color+'">'+esc(s.name)+'</text>').join('')}
function drawTrend(){objective=$('objective').value;let best=Infinity,points=[];(state.candidates||[]).filter(c=>c.objectives&&c.objectives[objective]!=null).sort((a,b)=>a.id-b.id).forEach(c=>{best=Math.min(best,c.objectives[objective]);points.push([c.id,best])});plot($('trend'),[{name:objective,color:'#58d6c7',values:points}],'architecture id')}
function render(s){state=s;let n=s.architecture_counts||s.counts,t=s.trial_counts||n,max=s.config.max_architectures||s.config.max_evaluations||1,v2=s.format_version===2,done=v2?(s.screened||0):(n.completed||0)+(n.failed||0),cards=v2?[['completed architectures',n.completed||0],['running trials',t.running||0],['pending trials',t.pending||0],['failed trials',t.failed||0],['frontier',s.frontier.length]]:[['completed',n.completed||0],['running',n.running||0],['pending',n.pending||0],['failed',n.failed||0],['frontier',s.frontier.length]];$('study').textContent=s.path+(s.frontier_rung!=null?' · frontier rung '+s.frontier_rung+(s.frontier_closed?' closed':' partial'):'');$('clock').textContent=s.status+' · updated '+new Date().toLocaleTimeString();$('cards').innerHTML=cards.map(x=>'<div class="card"><b>'+x[1]+'</b><span>'+x[0]+'</span></div>').join('');$('progress').style.width=(100*done/max).toFixed(1)+'%';$('progress-label').textContent=done+' / '+max+(v2?' screened architectures · ':' terminal architectures · ')+(100*done/max).toFixed(1)+'%';let names=s.objectives||[],qname=s.primary_quality||names.find(x=>x.startsWith('quality.')),old=$('objective').value;$('objective').innerHTML=names.map(x=>'<option>'+esc(x)+'</option>').join('');$('objective').value=names.includes(old)?old:names[0]||'';drawTrend();let base=[['id',r=>'#'+r.id],['status',r=>'<span class="tag '+r.status+'">'+esc(r.status)+'</span>'],['operation',r=>esc(r.mutation.operator)],['parent',r=>esc(r.parents.join(',')||'—')],['params',r=>fmt(r.parameters)],['nll',r=>fmt(qname?r.objectives?.[qname]:null)],['rank',r=>fmt(r.pareto_rank)]];$('frontier').innerHTML=table(s.frontier,base.concat(names.map(name=>[name,r=>fmt(r.objectives?.[name])])));$('candidates').innerHTML=table([...s.candidates].reverse(),base.concat([['updated',r=>esc((r.completed_at||r.started_at||r.created_at||'').slice(11,19))],['error',r=>esc(r.error||'')]]));document.querySelectorAll('tr[data-id]').forEach(r=>r.onclick=()=>loadCandidate(+r.dataset.id));$('raw-study').textContent=JSON.stringify({config:s.config,provenance:s.provenance,rungs:s.rungs,recommendations:s.recommendations},null,2)}
async function loadCandidate(id){selected=id;let d=await fetch('/api/candidate/'+id).then(r=>r.json()),layers=d.config.layers||[],mx=Math.max(...layers.map(x=>x.hidden_size),1),q=d.result?.quality,trial=d.result_trial,trialLabel=trial?'trial curves · rung '+trial.rung+' · seed index '+trial.seed_index+' · seed '+trial.seed:'no completed trial';$('detail').innerHTML='<b>architecture #'+id+'</b> <span class="tag '+d.status+'">'+esc(d.status)+'</span><div class="small">'+esc(d.mutation.operator)+' · '+fmt(d.parameters)+' parameters · parents '+esc(d.parents.join(',')||'—')+'</div><div class="layers">'+layers.map((l,i)=>'<div class="layer"><span>'+i+'</span><div class="bar '+(l.num_key_value_heads!=null?'attn':'')+'" style="width:'+(100*l.hidden_size/mx)+'%"></div><span class="small">h'+l.hidden_size+' · f'+l.intermediate_size+' · '+(l.num_key_value_heads==null?'mlp':'kv'+l.num_key_value_heads)+'</span></div>').join('')+'</div><h2>'+esc(trialLabel)+'</h2><svg id="loss" viewBox="0 0 800 250"></svg><details><summary>complete architecture record</summary><pre>'+esc(JSON.stringify(d,null,2))+'</pre></details>';let train=(q?.train_curve||[]).map(x=>[x.tokens,x.loss]),val=(q?.validation_curve||[]).map(x=>[x.tokens,x.loss]);plot($('loss'),[{name:'train',color:'#6d9fff',values:train},{name:'validation',color:'#f0b45a',values:val}],'trained tokens')}
async function refresh(){try{render(await fetch('/api/state',{cache:'no-store'}).then(r=>r.json()))}catch(e){$('clock').textContent='disconnected'}setTimeout(refresh,3000)}
$('objective').onchange=drawTrend;refresh();
</script>
</html>"""

dashboard_path = Path(__file__).with_name("search_dashboard.html")
dashboard = (
    dashboard_path.read_text(encoding="utf-8")
    if dashboard_path.is_file()
    else _fallback_dashboard
)


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


def _artifact_json(connection, database, digest):
    row = connection.execute(
        "select uri, media_type, size from artifacts where digest = ?", (digest,)
    ).fetchone()
    if row is None or row["media_type"] != "application/json":
        return None
    root = (Path(database).resolve().parent / "artifacts").resolve()
    path = (root / row["uri"]).resolve()
    if root not in path.parents or not path.is_file():
        return None
    try:
        data = path.read_bytes()
        if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != digest:
            return None
        return json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


class UnsupportedSearchStudy(Exception):
    pass


def _study_format(connection):
    tables = {
        row["name"]
        for row in connection.execute(
            "select name from sqlite_master where type = 'table'"
        )
    }
    if "candidates" in tables:
        schema = connection.execute(
            "select value from metadata where key = 'schema_version'"
        ).fetchone()
        if schema is None or int(schema["value"]) not in {1, 2, 3}:
            raise UnsupportedSearchStudy("unsupported legacy search schema")
        return 1
    if "architectures" in tables:
        v3_schema = connection.execute(
            "select value from metadata where key = 'database_schema_version'"
        ).fetchone()
        v3_semantics = connection.execute(
            "select value from metadata where key = 'study_semantics_version'"
        ).fetchone()
        if (
            v3_schema is not None
            and int(v3_schema["value"]) == 7
            and v3_semantics is not None
            and int(v3_semantics["value"]) == 3
        ):
            return 3
        search_format = connection.execute(
            "select value from metadata where key = 'search_format_version'"
        ).fetchone()
        schema = connection.execute(
            "select value from metadata where key = 'schema_version'"
        ).fetchone()
        if (
            search_format is None
            or int(search_format["value"]) != 2
            or schema is None
            or int(schema["value"]) not in {1, 2}
        ):
            raise UnsupportedSearchStudy("unsupported multi-fidelity search schema")
        return 2
    raise UnsupportedSearchStudy("database is not an architecture search study")


def _v1_snapshot(connection, path):
    study = connection.execute("select * from study where id = 1").fetchone()
    rows = connection.execute("select * from candidates order by id").fetchall()
    parent_rows = connection.execute(
        "select * from parents order by candidate_id"
    ).fetchall()
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
            "architecture_hash": row["architecture_hash"],
            "status": row["status"],
            "comparison_status": row["status"],
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
    revision = max(
        [study["updated_at"]]
        + [
            value
            for row in rows
            for value in (row["created_at"], row["started_at"], row["completed_at"])
            if value
        ]
    )
    return _clean({
        "format_version": 1,
        "path": str(Path(path).parent),
        "status": study["status"],
        "created_at": study["created_at"],
        "updated_at": revision,
        "data_revision": revision,
        "config": _load(study["config_json"]),
        "provenance": _load(study["provenance_json"]),
        "counts": counts,
        "architecture_counts": counts,
        "trial_counts": counts,
        "generated": len(candidates),
        "screened": counts.get("completed", 0) + counts.get("failed", 0),
        "primary_quality": "quality.validation_nll",
        "objectives": sorted(objectives),
        "candidates": candidates,
        "frontier": [candidate for candidate in candidates if candidate["is_frontier"]],
    })


def _v2_status(architecture_id, rungs, trials):
    architecture_rungs = [
        item for item in rungs if item["architecture_id"] == architecture_id
    ]
    if not architecture_rungs:
        return "pending"
    highest_rung = max(item["rung"] for item in architecture_rungs)
    highest = next(item for item in architecture_rungs if item["rung"] == highest_rung)
    architecture_trials = [
        item
        for item in trials
        if item["architecture_id"] == architecture_id
        and item["rung"] == highest_rung
    ]
    if any(item["status"] == "running" for item in architecture_trials):
        return "running"
    if any(item["status"] == "pending" for item in architecture_trials):
        return "pending"
    if highest["status"] == "failed" or any(
        item["status"] == "failed" for item in architecture_trials
    ):
        return "failed"
    if highest["aggregate_json"]:
        return "completed"
    return "pending"


def _v2_rung_status(architecture_id, rung, rungs, trials):
    selected = next(
        (
            item
            for item in rungs
            if item["architecture_id"] == architecture_id and item["rung"] == rung
        ),
        None,
    )
    if selected is None:
        return None
    selected_trials = [
        item
        for item in trials
        if item["architecture_id"] == architecture_id and item["rung"] == rung
    ]
    if any(item["status"] == "running" for item in selected_trials):
        return "running"
    if any(item["status"] == "pending" for item in selected_trials):
        return "pending"
    if selected["status"] == "failed" or any(
        item["status"] == "failed" for item in selected_trials
    ):
        return "failed"
    if selected["aggregate_json"]:
        return "completed"
    return selected["status"]


def _v2_rung_closed(rungs, config, rung):
    selected = [item for item in rungs if item["rung"] == rung]
    if any(item["status"] == "active" for item in selected):
        return False
    if rung == 0:
        return len(selected) >= config["rungs"][0]["architecture_limit"]
    if not _v2_rung_closed(rungs, config, rung - 1):
        return False
    previous_successes = sum(
        bool(item["aggregate_json"])
        for item in rungs
        if item["rung"] == rung - 1
    )
    expected = min(
        config["rungs"][rung]["architecture_limit"], previous_successes
    )
    return len(selected) >= expected


def _frontier_ids(values):
    ids = set()
    for architecture_id, objectives in values.items():
        dominated = False
        for other_id, other in values.items():
            if architecture_id == other_id:
                continue
            if all(other[name] <= objectives[name] for name in objectives) and any(
                other[name] < objectives[name] for name in objectives
            ):
                dominated = True
                break
        if not dominated:
            ids.add(architecture_id)
    return ids


def _pareto_ranks(values):
    remaining = dict(values)
    ranks = {}
    rank = 0
    while remaining:
        frontier = _frontier_ids(remaining)
        for architecture_id in frontier:
            ranks[architecture_id] = rank
            del remaining[architecture_id]
        rank += 1
    return ranks


def _directional_pareto_ranks(values, directions):
    normalized = {
        architecture_id: {
            name: value if directions.get(name) == "minimize" else -value
            for name, value in objectives.items()
        }
        for architecture_id, objectives in values.items()
    }
    return _pareto_ranks(normalized)


def _counts(rows, key="status"):
    values = {}
    for row in rows:
        value = row[key]
        values[value] = values.get(value, 0) + 1
    return values


def _v3_architecture_ids(connection):
    values = {}
    for row in connection.execute(
        "select sequence, payload_json from events where kind = 'architecture_added' order by sequence"
    ):
        digest = _load(row["payload_json"])["digest"]
        values[digest] = row["sequence"]
    return values


def _v3_objective_sets(connection, config):
    rows = connection.execute("select * from objective_sets").fetchall()
    values = [
        {
            "digest": row["digest"],
            "name": row["name"],
            **_load(row["definition_json"]),
        }
        for row in rows
    ]
    ordered = []
    for configured in config.get("objective_sets", []):
        match = next(
            (
                item
                for item in values
                if item["name"] == configured["name"]
                and item["objectives"] == configured["objectives"]
            ),
            None,
        )
        if match is not None and match not in ordered:
            ordered.append(match)
    known = {item["digest"] for item in ordered}
    result = ordered + [item for item in values if item["digest"] not in known]
    name_counts = {
        name: sum(item["name"] == name for item in result)
        for name in {item["name"] for item in result}
    }
    for item in result:
        item["label"] = (
            f"{item['name']} / {item['digest'][:8]}"
            if name_counts[item["name"]] > 1
            else item["name"]
        )
    return result


def _v3_estimate(rows):
    values = [row["value"] for row in rows]
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "stdev": stdev,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def _v3_action_architecture(row, runs):
    if row["architecture_digest"]:
        return row["architecture_digest"]
    if row["run_id"] in runs:
        return runs[row["run_id"]]["architecture_digest"]
    return (_load(row["payload_json"]) or {}).get("architecture_digest")


def _v3_status(digest, runs, actions, has_evidence):
    selected_runs = [row for row in runs.values() if row["architecture_digest"] == digest]
    selected_actions = [row for row in actions if row["_architecture_digest"] == digest]
    if any(row["status"] == "running" for row in selected_actions + selected_runs):
        return "running"
    if any(row["status"] == "pending" for row in selected_actions):
        return "pending"
    if any(row["status"] == "paused" for row in selected_runs):
        return "paused"
    if has_evidence or any(row["status"] in {"completed", "archived"} for row in selected_runs):
        return "completed"
    if any(row["status"] == "failed" for row in selected_actions + selected_runs):
        return "failed"
    return "pending"


def _v3_comparison_phase(config, tokens):
    calibration = config.get("calibration", {})
    if tokens is None:
        return None
    if tokens <= calibration.get("noise_tokens", -1):
        return "noise"
    if tokens <= calibration.get("broad_tokens", -1):
        return "broad"
    return "anchor"


def _v3_snapshot(connection, path, requested_objective_set=None, requested_tokens=None):
    study = connection.execute("select * from study where id = 1").fetchone()
    config = _load(study["config_json"])
    objective_sets = _v3_objective_sets(connection, config)
    if not objective_sets:
        raise UnsupportedSearchStudy("v3 study has no objective sets")
    if requested_objective_set is None:
        selected_set = objective_sets[0]
    else:
        selected_set = next(
            (item for item in objective_sets if item["digest"] == requested_objective_set),
            None,
        )
        if selected_set is None:
            named = [item for item in objective_sets if item["name"] == requested_objective_set]
            if len(named) > 1:
                raise ValueError("objective set name is ambiguous; use its digest")
            selected_set = named[0] if named else None
    if selected_set is None:
        raise ValueError("unknown objective set")

    architecture_ids = _v3_architecture_ids(connection)
    architectures = connection.execute(
        "select * from architectures order by created_at, digest"
    ).fetchall()
    run_rows = connection.execute("select * from runs order by id").fetchall()
    runs = {row["id"]: dict(row) for row in run_rows}
    action_rows = connection.execute("select * from actions order by id").fetchall()
    actions = []
    for row in action_rows:
        value = dict(row)
        value["_architecture_digest"] = _v3_action_architecture(row, runs)
        actions.append(value)
    observations = connection.execute(
        "select * from observations where objective_set_digest = ? order by id",
        (selected_set["digest"],),
    ).fetchall()
    available_horizons = sorted({
        row["tokens"]
        for row in observations
        if row["objective_name"] == "quality.target_nll" and row["tokens"] is not None
    })
    coverage = {
        tokens: len({
            row["architecture_digest"]
            for row in observations
            if row["objective_name"] == "quality.target_nll" and row["tokens"] == tokens
        })
        for tokens in available_horizons
    }
    if requested_tokens is not None:
        comparison_tokens = int(requested_tokens)
        if comparison_tokens not in available_horizons:
            raise ValueError("token horizon has no quality observations")
    else:
        comparison_tokens = max(
            available_horizons,
            key=lambda tokens: (coverage[tokens], tokens),
            default=None,
        )

    definitions = {item["name"]: item for item in selected_set["objectives"]}
    directions = {name: item["direction"] for name, item in definitions.items()}
    required = {
        name for name, item in definitions.items() if item.get("required_for_selection", True)
    }
    latest_report = connection.execute(
        "select * from posterior_reports order by created_at desc, digest desc limit 1"
    ).fetchone()
    anchors = set()
    display_anchors = set()
    posterior_report = None
    posterior_metrics = {}
    if latest_report is not None:
        anchors = {
            row["architecture_digest"]
            for row in connection.execute(
                "select architecture_digest from posterior_anchors where report_digest = ?",
                (latest_report["digest"],),
            )
        }
        posterior_report = dict(latest_report)
        posterior_report["definition"] = _load(posterior_report.pop("definition_json"))
        posterior_report["anchors"] = sorted(
            anchors, key=lambda digest: architecture_ids[digest]
        )
        shadow = _artifact_json(connection, path, posterior_report["artifact_digest"])
        report_objective_set = next(
            (
                item
                for item in (shadow or {}).get("evidence", {}).get("objective_sets", [])
                if item["name"] == selected_set["name"]
                and item["objectives"] == selected_set["objectives"]
            ),
            None,
        )
        display_anchors = anchors if report_objective_set is not None else set()
        selected_posterior = (
            (shadow or {}).get("posterior", {}).get(selected_set["name"], {})
            if report_objective_set is not None
            else {}
        )
        posterior_metrics = {
            item["architecture_digest"]: item
            for item in selected_posterior.get("metrics", [])
        }
        posterior_report["calibration"] = (
            (shadow or {}).get("calibration", {}).get(selected_set["name"])
            if report_objective_set is not None
            else None
        )
        posterior_report["frontier_calibration"] = (
            (shadow or {}).get("frontier_calibration", {}).get(selected_set["name"])
            if report_objective_set is not None
            else None
        )
        posterior_report["noise"] = (
            (shadow or {}).get("noise", {}).get(selected_set["name"])
            if report_objective_set is not None
            else None
        )

    values_by_architecture = {}
    estimates_by_architecture = {}
    for row in architectures:
        grouped = {}
        for observation in observations:
            if observation["architecture_digest"] != row["digest"]:
                continue
            definition = definitions.get(observation["objective_name"], {})
            if definition.get("role") == "quality" and observation["tokens"] != comparison_tokens:
                continue
            if (
                definition.get("role") != "quality"
                and observation["tokens"] is not None
                and observation["tokens"] != comparison_tokens
            ):
                continue
            grouped.setdefault(observation["objective_name"], []).append(observation)
        estimates = {name: _v3_estimate(items) for name, items in grouped.items()}
        estimates_by_architecture[row["digest"]] = estimates
        values_by_architecture[row["digest"]] = {
            name: estimate["mean"] for name, estimate in estimates.items()
        }

    comparable = {
        architecture_ids[digest]: {
            name: values_by_architecture[digest][name] for name in required
        }
        for digest in architecture_ids
        if required <= values_by_architecture.get(digest, {}).keys()
    }
    pareto_ranks = _directional_pareto_ranks(comparable, directions)
    candidates = []
    architecture_counts = {}
    for row in architectures:
        digest = row["digest"]
        architecture_id = architecture_ids[digest]
        static = _load(row["static_json"])
        operation = _load(row["operation_json"]) or {"operator": "unknown"}
        parent_digests = [
            item.get("digest") if isinstance(item, dict) else item
            for item in operation.get("parents", [])
        ]
        parent_ids = [architecture_ids[item] for item in parent_digests if item in architecture_ids]
        architecture_actions = [item for item in actions if item["_architecture_digest"] == digest]
        architecture_runs = [item for item in runs.values() if item["architecture_digest"] == digest]
        values = values_by_architecture[digest]
        complete = required <= values.keys()
        status = _v3_status(digest, runs, actions, bool(values))
        architecture_counts[status] = architecture_counts.get(status, 0) + 1
        candidates.append({
            "id": architecture_id,
            "architecture_hash": digest,
            "architecture_digest": digest,
            "status": status,
            "comparison_status": "completed" if complete else "partial" if values else "pending",
            "mutation": operation,
            "parents": parent_ids,
            "parameters": static.get("parameters"),
            "layers": static.get("logical_depth"),
            "unique_parameter_blocks": static.get("unique_parameter_blocks"),
            "objectives": values,
            "estimates": estimates_by_architecture[digest],
            "comparison_tokens": comparison_tokens if complete else None,
            "rung": comparison_tokens if complete else None,
            "is_frontier": pareto_ranks.get(architecture_id) == 0,
            "pareto_rank": pareto_ranks.get(architecture_id),
            "is_anchor": digest in display_anchors,
            "posterior": posterior_metrics.get(digest),
            "error": next(
                (item["error"] for item in reversed(architecture_actions) if item["error"]),
                None,
            ),
            "created_at": row["created_at"],
            "started_at": next(
                (item["claimed_at"] for item in architecture_actions if item["claimed_at"]),
                None,
            ),
            "completed_at": max(
                [item["completed_at"] for item in architecture_actions if item["completed_at"]]
                + [item["updated_at"] for item in architecture_runs if item["tokens"] > 0],
                default=None,
            ),
        })

    action_counts = _counts(actions)
    run_counts = _counts(run_rows)
    screened = len({
        row["architecture_digest"]
        for row in observations
        if row["objective_name"] == "quality.target_nll"
    })
    checkpoint_tokens = config.get("quality", {}).get("checkpoint_tokens", [])
    horizons = [
        {
            "tokens": tokens,
            "evaluated": coverage.get(tokens, 0),
            "selected": tokens == comparison_tokens,
            "phase": _v3_comparison_phase(config, tokens),
        }
        for tokens in checkpoint_tokens
    ]
    latest_event = connection.execute(
        "select sequence, created_at from events order by sequence desc limit 1"
    ).fetchone()
    observed_objectives = sorted({name for values in values_by_architecture.values() for name in values})
    primary_quality = next(
        (name for name, item in definitions.items() if item["role"] == "quality" and name in observed_objectives),
        None,
    )
    phase = "bootstrap" if posterior_report is None else "anchors"
    if posterior_report is not None and anchors:
        target = config.get("calibration", {}).get("anchor_tokens")
        evaluated = {
            row["architecture_digest"]
            for row in observations
            if row["objective_name"] == "quality.target_nll" and row["tokens"] == target
        }
        if anchors <= evaluated:
            phase = "anchor_complete"
    return _clean({
        "format_version": 3,
        "path": str(Path(path).parent),
        "status": study["status"],
        "phase": phase,
        "created_at": study["created_at"],
        "updated_at": latest_event["created_at"] if latest_event else study["updated_at"],
        "data_revision": latest_event["sequence"] if latest_event else 0,
        "config": config,
        "provenance": _load(study["provenance_json"]),
        "versions": _load(study["versions_json"]),
        "counts": architecture_counts,
        "architecture_counts": architecture_counts,
        "trial_counts": run_counts,
        "run_counts": run_counts,
        "action_counts": action_counts,
        "generated": len(candidates),
        "screened": screened,
        "architecture_budget": config.get("calibration", {}).get("broad_architectures"),
        "objective_sets": objective_sets,
        "active_objective_set": selected_set,
        "objective_directions": directions,
        "available_horizons": available_horizons,
        "comparison_tokens": comparison_tokens,
        "comparison_phase": _v3_comparison_phase(config, comparison_tokens),
        "horizons": horizons,
        "frontier_closed": (
            comparison_tokens == config.get("calibration", {}).get("broad_tokens")
            and len(comparable) >= config.get("calibration", {}).get("broad_architectures", math.inf)
        ),
        "primary_quality": primary_quality,
        "objectives": observed_objectives,
        "posterior_report": posterior_report,
        "anchor_ids": sorted((architecture_ids[item] for item in display_anchors)),
        "candidates": candidates,
        "frontier": [item for item in candidates if item["is_frontier"]],
    })


def _v2_snapshot(connection, path, requested_rung=None):
    study = connection.execute("select * from study where id = 1").fetchone()
    study_config = _load(study["config_json"])
    architectures = connection.execute(
        "select * from architectures order by cohort, slot"
    ).fetchall()
    parent_rows = connection.execute(
        "select * from architecture_parents order by child_id, role"
    ).fetchall()
    rungs = connection.execute(
        "select * from architecture_rungs order by rung, architecture_id"
    ).fetchall()
    trials = connection.execute(
        "select * from trials order by rung, architecture_id, seed_index"
    ).fetchall()
    parents = {}
    for row in parent_rows:
        parents.setdefault(row["child_id"], []).append({
            "id": row["parent_id"],
            "role": row["role"],
        })
    aggregate_rows = [row for row in rungs if row["aggregate_json"]]
    available_rungs = sorted({row["rung"] for row in aggregate_rows})
    closed_rungs = [
        rung
        for rung in available_rungs
        if _v2_rung_closed(rungs, study_config, rung)
    ]
    if requested_rung is not None and requested_rung not in available_rungs:
        raise ValueError("rung has no completed architectures")
    frontier_rung = requested_rung
    if frontier_rung is None:
        frontier_rung = (
            closed_rungs[-1]
            if closed_rungs
            else available_rungs[0] if available_rungs else None
        )
    by_architecture_rung = {
        (row["architecture_id"], row["rung"]): row for row in rungs
    }
    objectives = set()
    display_values = {}
    if frontier_rung is not None:
        for row in aggregate_rows:
            if row["rung"] != frontier_rung:
                continue
            aggregate = _load(row["aggregate_json"])
            display_values[row["architecture_id"]] = {
                name: estimate["mean"]
                for name, estimate in aggregate["objectives"].items()
            }
    pareto_ranks = _pareto_ranks(display_values)
    frontier_ids = {
        architecture_id
        for architecture_id, rank in pareto_ranks.items()
        if rank == 0
    }
    candidates = []
    architecture_counts = {}
    screened = 0
    for row in architectures:
        rung_zero = by_architecture_rung.get((row["id"], 0))
        if rung_zero is not None and rung_zero["status"] != "active":
            screened += 1
        architecture_status = _v2_status(row["id"], rungs, trials)
        architecture_counts[architecture_status] = (
            architecture_counts.get(architecture_status, 0) + 1
        )
        selected_rung = (
            by_architecture_rung.get((row["id"], frontier_rung))
            if frontier_rung is not None
            else None
        )
        aggregate = _load(selected_rung["aggregate_json"]) if selected_rung else None
        estimates = aggregate.get("objectives", {}) if aggregate else {}
        values = {
            name: estimate["mean"] for name, estimate in estimates.items()
        }
        objectives.update(values)
        architecture_trials = [
            item for item in trials if item["architecture_id"] == row["id"]
        ]
        candidates.append({
            "id": row["id"],
            "architecture_hash": row["architecture_hash"],
            "cohort": row["cohort"],
            "slot": row["slot"],
            "status": architecture_status,
            "comparison_status": (
                _v2_rung_status(row["id"], frontier_rung, rungs, trials)
                if frontier_rung is not None
                else None
            ),
            "mutation": _load(row["operation_json"]),
            "parents": [item["id"] for item in parents.get(row["id"], [])],
            "parameters": _load(row["static_json"]).get("parameters"),
            "layers": len(_load(row["architecture_json"]).get("layers", [])),
            "objectives": values,
            "estimates": estimates,
            "rung": frontier_rung if aggregate else None,
            "in_population": False,
            "is_frontier": row["id"] in frontier_ids,
            "pareto_rank": pareto_ranks.get(row["id"]),
            "crowding": selected_rung["crowding"] if selected_rung else None,
            "novelty": selected_rung["novelty"] if selected_rung else None,
            "error": next(
                (item["error"] for item in architecture_trials if item["error"]),
                None,
            ),
            "created_at": row["created_at"],
            "started_at": next(
                (item["started_at"] for item in architecture_trials if item["started_at"]),
                None,
            ),
            "completed_at": (
                selected_rung["completed_at"] if selected_rung else None
            ),
        })
    trial_counts = {}
    for row in trials:
        trial_counts[row["status"]] = trial_counts.get(row["status"], 0) + 1
    rung_summary = []
    for rung in sorted({row["rung"] for row in rungs}):
        selected = [row for row in rungs if row["rung"] == rung]
        selected_trials = [row for row in trials if row["rung"] == rung]
        selected_trial_counts = {}
        for trial in selected_trials:
            selected_trial_counts[trial["status"]] = (
                selected_trial_counts.get(trial["status"], 0) + 1
            )
        rung_summary.append({
            "rung": rung,
            "name": study_config["rungs"][rung]["name"],
            "closed": _v2_rung_closed(rungs, study_config, rung),
            "total": len(selected),
            "active": sum(row["status"] == "active" for row in selected),
            "successful": sum(bool(row["aggregate_json"]) for row in selected),
            "failed": sum(row["status"] == "failed" for row in selected),
            "trial_counts": selected_trial_counts,
        })
    primary_quality = next(
        (
            f"quality.validation_nll.{item['name']}"
            for item in study_config["validation_slices"]
            if item.get("objective", True)
        ),
        None,
    )
    revision = max(
        [study["updated_at"]]
        + [row["created_at"] for row in architectures]
        + [
            value
            for row in rungs
            for value in (row["created_at"], row["completed_at"])
            if value
        ]
        + [
            value
            for row in trials
            for value in (row["created_at"], row["started_at"], row["completed_at"])
            if value
        ]
    )
    return _clean({
        "format_version": 2,
        "path": str(Path(path).parent),
        "status": study["status"],
        "created_at": study["created_at"],
        "updated_at": revision,
        "data_revision": revision,
        "config": study_config,
        "provenance": _load(study["provenance_json"]),
        "recommendations": _load(study["recommendations_json"]),
        "counts": architecture_counts,
        "architecture_counts": architecture_counts,
        "trial_counts": trial_counts,
        "generated": len(candidates),
        "screened": screened,
        "rungs": rung_summary,
        "frontier_rung": frontier_rung,
        "frontier_rung_name": (
            study_config["rungs"][frontier_rung]["name"]
            if frontier_rung is not None
            else None
        ),
        "available_rungs": available_rungs,
        "frontier_closed": (
            _v2_rung_closed(rungs, study_config, frontier_rung)
            if frontier_rung is not None
            else False
        ),
        "primary_quality": primary_quality,
        "objectives": sorted(objectives),
        "candidates": candidates,
        "frontier": [candidate for candidate in candidates if candidate["is_frontier"]],
    })


def snapshot(path, rung=None, objective_set=None, tokens=None) -> dict[str, Any]:
    connection = _connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("begin")
        format_version = _study_format(connection)
        if format_version == 3:
            if rung is not None:
                raise ValueError("v3 studies use token horizons instead of rungs")
            return _v3_snapshot(connection, path, objective_set, tokens)
        if format_version == 2:
            if objective_set is not None or tokens is not None:
                raise ValueError("v2 studies do not contain objective sets or token horizons")
            return _v2_snapshot(connection, path, rung)
        if rung is not None or objective_set is not None or tokens is not None:
            raise ValueError("legacy studies do not contain rungs")
        return _v1_snapshot(connection, path)
    finally:
        connection.close()


def _v1_candidate(connection, candidate_id):
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
            "select * from attempts where candidate_id = ? order by id",
            (candidate_id,),
        )
    ]
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
    value["result_trial"] = (
        {"rung": "legacy", "seed_index": 0, "seed": value["seed"]}
        if value["result"]
        else None
    )
    return _clean(value)


def _v2_candidate(connection, architecture_id):
    row = connection.execute(
        "select * from architectures where id = ?", (architecture_id,)
    ).fetchone()
    if row is None:
        raise KeyError(architecture_id)
    parents = [
        {"id": item["parent_id"], "role": item["role"]}
        for item in connection.execute(
            """
            select parent_id, role from architecture_parents
            where child_id = ? order by role
            """,
            (architecture_id,),
        )
    ]
    rung_rows = connection.execute(
        """
        select * from architecture_rungs
        where architecture_id = ? order by rung
        """,
        (architecture_id,),
    ).fetchall()
    trial_rows = connection.execute(
        """
        select * from trials where architecture_id = ?
        order by rung desc, seed_index
        """,
        (architecture_id,),
    ).fetchall()
    attempts = [
        dict(item)
        for item in connection.execute(
            """
            select attempts.* from attempts join trials on trials.id = attempts.trial_id
            where trials.architecture_id = ? order by attempts.id
            """,
            (architecture_id,),
        )
    ]
    rungs = []
    for item in rung_rows:
        value = dict(item)
        value.update(
            aggregate=_load(value.pop("aggregate_json")),
            decision=_load(value.pop("decision_json")),
        )
        rungs.append(value)
    trials = []
    for item in trial_rows:
        value = dict(item)
        value["result"] = _load(value.pop("result_json"))
        trials.append(value)
    result = next(
        (item["result"] for item in trials if item["result"] is not None),
        None,
    )
    result_trial = next(
        (
            {
                "id": item["id"],
                "rung": item["rung"],
                "seed_index": item["seed_index"],
                "seed": item["seed"],
            }
            for item in trials
            if item["result"] is not None
        ),
        None,
    )
    return _clean({
        "id": row["id"],
        "architecture_hash": row["architecture_hash"],
        "config": _load(row["architecture_json"]),
        "static": _load(row["static_json"]),
        "parameters": _load(row["static_json"]).get("parameters"),
        "status": _v2_status(architecture_id, rung_rows, trial_rows),
        "generation_seed": row["generation_seed"],
        "mutation": _load(row["operation_json"]),
        "repairs": _load(row["repairs_json"]),
        "parents": [item["id"] for item in parents],
        "parent_roles": parents,
        "result": result,
        "result_trial": result_trial,
        "rungs": rungs,
        "trials": trials,
        "attempts": attempts,
        "created_at": row["created_at"],
    })


def _v3_candidate(connection, architecture_id, path):
    architecture_ids = _v3_architecture_ids(connection)
    digest = next(
        (digest for digest, stable_id in architecture_ids.items() if stable_id == architecture_id),
        None,
    )
    if digest is None:
        raise KeyError(architecture_id)
    row = connection.execute(
        "select * from architectures where digest = ?", (digest,)
    ).fetchone()
    if row is None:
        raise KeyError(architecture_id)
    static = _load(row["static_json"])
    operation = _load(row["operation_json"]) or {"operator": "unknown"}
    parent_digests = operation.get("parents", [])
    parent_roles = []
    for position, parent in enumerate(parent_digests):
        if isinstance(parent, dict):
            parent_digest = parent.get("digest")
            role = parent.get("role", f"parent-{position}")
        else:
            parent_digest = parent
            role = f"parent-{position}"
        if parent_digest in architecture_ids:
            parent_roles.append({"id": architecture_ids[parent_digest], "role": role})

    run_rows = connection.execute(
        "select * from runs where architecture_digest = ? order by id", (digest,)
    ).fetchall()
    runs = []
    for item in run_rows:
        value = dict(item)
        value["protocol"] = _load(value.pop("protocol_json"))
        value["seed_bundle"] = _load(value.pop("seed_bundle_json"))
        runs.append(value)

    all_run_rows = {
        item["id"]: dict(item)
        for item in connection.execute("select id, architecture_digest from runs")
    }
    actions = []
    for item in connection.execute("select * from actions order by id"):
        if _v3_action_architecture(item, all_run_rows) != digest:
            continue
        value = dict(item)
        value["payload"] = _load(value.pop("payload_json"))
        value["result"] = _load(value.pop("result_json"))
        value.pop("claim_token", None)
        actions.append(value)

    observations = [
        {
            **dict(item),
        }
        for item in connection.execute(
            "select * from observations where architecture_digest = ? order by id",
            (digest,),
        )
    ]
    checkpoints = []
    for item in connection.execute(
        """
        select run_checkpoints.*, pruned_checkpoint_payloads.reason as prune_reason,
            pruned_checkpoint_payloads.pruned_at
        from run_checkpoints
        left join pruned_checkpoint_payloads
            on pruned_checkpoint_payloads.artifact_digest = run_checkpoints.artifact_digest
        join runs on runs.id = run_checkpoints.run_id
        where runs.architecture_digest = ?
        order by run_checkpoints.run_id, run_checkpoints.tokens
        """,
        (digest,),
    ):
        value = dict(item)
        value["payload_pruned"] = value["pruned_at"] is not None
        checkpoints.append(value)
    evaluations = [
        dict(item)
        for item in connection.execute(
            """
            select quality_evaluations.*, run_checkpoints.tokens as training_tokens
            from quality_evaluations
            join runs on runs.id = quality_evaluations.run_id
            join run_checkpoints
                on run_checkpoints.artifact_digest = quality_evaluations.checkpoint_digest
            where runs.architecture_digest = ?
            order by quality_evaluations.run_id, run_checkpoints.tokens
            """,
            (digest,),
        )
    ]
    profile_results = []
    for item in connection.execute(
        """
        select profile_results.*, actions.payload_json, actions.result_json
        from profile_results join actions on actions.id = profile_results.action_id
        where profile_results.architecture_digest = ?
        order by profile_results.scenario_digest, profile_results.repetition
        """,
        (digest,),
    ):
        value = dict(item)
        payload = _load(value.pop("payload_json"))
        value["scenario"] = payload.get("scenario")
        value["objective_set_digest"] = payload.get("objective_set_digest")
        value["result"] = _load(value.pop("result_json"))
        profile_results.append(value)

    posterior_reports = []
    for report in connection.execute(
        "select * from posterior_reports order by created_at, digest"
    ):
        report_anchors = [
            item["architecture_digest"]
            for item in connection.execute(
                "select architecture_digest from posterior_anchors where report_digest = ? order by position",
                (report["digest"],),
            )
        ]
        if digest not in report_anchors:
            continue
        value = dict(report)
        value["definition"] = _load(value.pop("definition_json"))
        value["anchor_position"] = report_anchors.index(digest)
        shadow = _artifact_json(connection, path, value["artifact_digest"])
        value["posterior"] = {
            name: next(
                (
                    metric
                    for metric in section.get("metrics", [])
                    if metric["architecture_digest"] == digest
                ),
                None,
            )
            for name, section in (shadow or {}).get("posterior", {}).items()
        }
        posterior_reports.append(value)

    action_ids = {item["id"] for item in actions}
    planning_decisions = []
    if action_ids:
        placeholders = ",".join("?" for _ in action_ids)
        for item in connection.execute(
            f"""
            select distinct planning_decisions.* from planning_decisions
            join planning_actions
                on planning_actions.decision_digest = planning_decisions.digest
            where planning_actions.action_id in ({placeholders})
            order by planning_decisions.event_sequence
            """,
            tuple(sorted(action_ids)),
        ):
            value = dict(item)
            value["definition"] = _load(value.pop("definition_json"))
            planning_decisions.append(value)

    artifact_digests = {
        item["artifact_digest"] for item in observations if item["artifact_digest"]
    }
    artifact_digests.update(item["artifact_digest"] for item in checkpoints)
    artifact_digests.update(item["artifact_digest"] for item in evaluations)
    artifact_digests.update(item["artifact_digest"] for item in profile_results)
    artifact_digests.update(item["artifact_digest"] for item in posterior_reports)
    artifacts = []
    artifact_edges = []
    if artifact_digests:
        placeholders = ",".join("?" for _ in artifact_digests)
        artifacts = [
            dict(item)
            for item in connection.execute(
                f"select * from artifacts where digest in ({placeholders}) order by created_at, digest",
                tuple(sorted(artifact_digests)),
            )
        ]
        artifact_edges = [
            dict(item)
            for item in connection.execute(
                f"""
                select * from artifact_edges
                where parent_digest in ({placeholders}) or child_digest in ({placeholders})
                order by parent_digest, child_digest, relation
                """,
                tuple(sorted(artifact_digests)) * 2,
            )
        ]

    run_progress = {
        run["id"]: max(
            (item["training_tokens"] for item in evaluations if item["run_id"] == run["id"]),
            default=-1,
        )
        for run in runs
    }
    curve_run = min(
        runs,
        key=lambda item: (
            -run_progress[item["id"]],
            item["seed_bundle"].get("index", math.inf),
            item["id"],
        ),
        default=None,
    )
    validation_curve = (
        [
            {"tokens": item["training_tokens"], "loss": item["nll"]}
            for item in evaluations
            if curve_run is not None and item["run_id"] == curve_run["id"]
        ]
        if curve_run is not None
        else []
    )
    result = {
        "quality": {"train_curve": [], "validation_curve": validation_curve},
    }
    result_trial = (
        {
            "id": curve_run["id"],
            "rung": "checkpoint",
            "seed_index": curve_run["seed_bundle"].get("index"),
            "seed": curve_run["seed_bundle"].get("numerical_seed"),
        }
        if curve_run is not None and validation_curve
        else None
    )
    has_evidence = bool(observations)
    all_actions = [
        {**item, "_architecture_digest": digest}
        for item in actions
    ]
    return _clean({
        "format_version": 3,
        "id": architecture_id,
        "architecture_hash": digest,
        "architecture_digest": digest,
        "config": _load(row["architecture_json"]),
        "static": static,
        "parameters": static.get("parameters"),
        "status": _v3_status(digest, {item["id"]: item for item in runs}, all_actions, has_evidence),
        "mutation": operation,
        "parents": [item["id"] for item in parent_roles],
        "parent_roles": parent_roles,
        "runs": runs,
        "trials": runs,
        "actions": actions,
        "attempts": [],
        "observations": observations,
        "checkpoints": checkpoints,
        "quality_evaluations": evaluations,
        "profile_results": profile_results,
        "planning_decisions": planning_decisions,
        "posterior_reports": posterior_reports,
        "artifacts": artifacts,
        "artifact_edges": artifact_edges,
        "result": result,
        "result_trial": result_trial,
        "rungs": [],
        "created_at": row["created_at"],
    })


def candidate(path, candidate_id) -> dict[str, Any]:
    connection = _connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("begin")
        format_version = _study_format(connection)
        if format_version == 3:
            return _v3_candidate(connection, candidate_id, path)
        if format_version == 2:
            return _v2_candidate(connection, candidate_id)
        return _v1_candidate(connection, candidate_id)
    finally:
        connection.close()


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
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/":
                    self.send(dashboard.encode(), "text/html")
                elif path == "/api/state":
                    query = parse_qs(parsed.query)
                    values = query.get("rung")
                    rung = int(values[0]) if values else None
                    objective_set = query.get("objective_set", [None])[0]
                    tokens = query.get("tokens", [None])[0]
                    self.send(snapshot(database, rung, objective_set, tokens))
                elif path.startswith("/api/candidate/"):
                    self.send(candidate(database, int(path.rsplit("/", 1)[-1])))
                else:
                    self.send({"error": "not found"}, status=404)
            except (KeyError, ValueError):
                self.send({"error": "not found"}, status=404)
            except UnsupportedSearchStudy as error:
                self.send({"error": str(error)}, status=409)
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
        matches = [
            path
            for path in (
                root / "search-v3" / args.study / "study.sqlite3",
                root / "search" / args.study / "study.sqlite3",
            )
            if path.is_file()
        ]
        if len(matches) > 1:
            raise ValueError("study name exists in both search and search-v3; pass its database path")
        database = matches[0] if matches else root / "search" / args.study / "study.sqlite3"
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
