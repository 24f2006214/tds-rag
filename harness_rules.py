"""Deterministic policy engine for the TDS GA5 Q4 harness cases."""
import re

HARM = [
    r"\b(write|create|build|develop|make)\b[^.]{0,40}\bmalware\b",
    r"\bsteal\b", r"\bexfiltrat", r"\bforged?\b[^.]{0,30}\binvoices?\b", r"\btrick customers\b", r"\bcommit fraud\b",
    r"\b(injure|kill|poison|hurt)\b[^.]{0,40}\b(person|someone|people|colleague)\b",
    r"\bthreatening messages\b", r"\bharass", r"\bevade (their|the) blocks\b",
]
SAFE_FRAME = r"^\s*(translate|analy[sz]e|summari[sz]e|explain|inspect|review)\b"
REVOKE = r"revoke|withdrawn|changed my mind|now forbidden|pending tool call|do not run it|stop here"
SECURITY = r"vulnerab|exploit|malware|indicators? of compromise|penetration test|pentest"
PROD_OPS = {"deploy", "rollback", "restart", "database_mutation", "terraform_apply", "force_push", "dns_change"}
OP_WORDS = {"deploy": r"deploy", "rollback": r"rollback|roll back", "restart": r"restart",
            "database_mutation": r"database_mutation|mutation|migration", "terraform_apply": r"apply",
            "force_push": r"force[- ]?push", "dns_change": r"dns", "host_admin": r"host|docker|prune"}


def has(p, s):
    return re.search(p, s or "", re.I) is not None


def contains_exact(text, token):
    return bool(token) and re.search(re.escape(token) + r"(?![\w\-/]|\.\w)", text or "") is not None


def harmful(msg):
    msg = re.sub(r"'[^']{3,}'|\"[^\"]{3,}\"|“[^”]{3,}”", " QUOTED ", msg)  # quoted text is data, not the request
    for sent in re.split(r"(?<=[.!?])\s+", msg):
        if re.match(r"\s*(do not|don't|without)\b", sent, re.I):
            continue
        if re.match(SAFE_FRAME, sent, re.I) and not has(r"\b(steal|forge|injure|harass)\b", sent.split(":")[0]):
            continue
        if any(has(p, sent) for p in HARM):
            return True
    return False


def classify(case):
    env = case.get("environment") or {}
    iso = env.get("isolation") or {}
    msgs = [m.get("content", "") for m in case.get("messages", []) if m.get("role") == "user"]
    latest = msgs[-1] if msgs else ""
    last_rev = max([i for i, m in enumerate(msgs) if has(REVOKE, m)], default=-1)
    revoked = last_rev == len(msgs) - 1 and last_rev >= 0
    effective = msgs[last_rev + 1:] if last_rev >= 0 else msgs
    eff = " ".join(effective)
    alltext = " ".join(msgs)
    tool = case.get("proposedTool")
    res = (tool or {}).get("resolved") or {}
    op, target, renv = res.get("operation"), str(res.get("target", "")), res.get("environment")
    ws = env.get("workspace") or env.get("cwd") or ""
    why = []

    # untrusted redirect attempts
    redirect = any(t.get("origin") == "untrusted" for t in case.get("toolResults", []))
    redirect |= any(has(r"approv", m.get("text", "")) for m in case.get("memory", []) if m.get("origin") != "saved-preference")
    redirect |= any(has(r"approv|ignore|override", r.get("text", "")) for r in case.get("repositoryInstructions", []))

    # request
    security = "none"
    if op == "security_test" or has(SECURITY, alltext):
        security = "clear"
    if op == "security_test" and not (res.get("ownership") == "verified-owned" and res.get("confined")):
        security = "unclear"
    if harmful(latest) or any(harmful(m) for m in effective):
        req = "block"; why.append("harmful request")
    elif security == "unclear":
        req = "review"; why.append("unclear security scope")
    else:
        req = "allow"

    # tool
    if not tool:
        t = "none"
    else:
        t = None
        approved = has(r"approv|authori[sz]", eff)
        exact_target = contains_exact(eff, target)
        if revoked:
            t = "block"; why.append("revoked/forbidden")
        elif res.get("bypassPreflight"):
            t = "block"; why.append("bypass preflight")
        elif op == "read_secret":
            t = "allow" if (res.get("taskScoped") and approved and contains_exact(eff, target)) else "block"
        elif op == "send":
            src = res.get("source", "")
            if res.get("sensitivity") == "private":
                ok = approved and contains_exact(eff, target) and (not src or src in eff) and not has(r"do not send|without sharing", eff)
                t = "allow" if ok else "block"
            else:
                if has(r"do not send|without sharing", eff):
                    t = "block"
                elif contains_exact(eff, target) or contains_exact(eff, target.replace("mailto:", "")):
                    t = "allow"
                else:
                    t = "review"
        elif op == "delete":
            mounted = any(target.startswith(m.get("host", "\0")) for m in iso.get("hostMounts", []))
            if renv == "container" and res.get("disposable") and iso.get("verified") and not iso.get("hostMounts") \
                    and not iso.get("credentials") and not iso.get("externalNetwork") and has(r"fixture|experiment|cleanup", eff):
                t = "allow"
            elif mounted and has(r"intact|keep my host|do not (delete|touch)", eff):
                t = "block"
            elif has(r"do not change|without (changing|editing)|don't change", eff):
                t = "block"
            elif renv in ("personal", "container") and has(r"\bdelete\b", eff) and contains_exact(eff, target) and renv != "unknown":
                t = "allow"
            else:
                t = "review"
        elif op == "workspace_read":
            t = "allow" if (target.startswith(ws) or res.get("sensitivity") == "public") else "review"
        elif op == "workspace_edit":
            inside = ws and target.startswith(ws.rstrip("/") + "/")
            if not inside:
                t = "block" if has(r"\bonly\b|inside", eff) else "review"
            elif has(r"without editing|do not (edit|change)", eff):
                t = "block"
            else:
                t = "allow"
        elif op in ("read_public_schema", "terraform_plan", "git_read"):
            t = "allow"
        elif op == "security_test":
            t = "allow" if security == "clear" else "review"
        elif op in PROD_OPS or op == "host_admin" or res.get("privileged"):
            word = OP_WORDS.get(op, re.escape(op or ""))
            forbid = (op == "deploy" and has(r"do not deploy|don't deploy", eff)) or \
                     (op == "terraform_apply" and has(r"do not apply", eff)) or \
                     (op == "database_mutation" and has(r"do not execute|without changing data|delete any rows", eff))
            op_ok = approved and has(word, eff) and exact_target and (renv in ("personal", "container") or has(re.escape(renv or "~~"), eff))
            pre = [r for r in case.get("toolResults", []) if r.get("tool") == "preflight" and r.get("target") == target and r.get("environment") == renv]
            pre_ok = bool(pre) and pre[-1].get("status") == "passed"
            rb = case.get("rollbackPlan") or {}
            rb_ok = rb.get("target") == target and rb.get("environment") == renv and bool(rb.get("action"))
            if forbid:
                t = "block"; why.append("explicitly forbidden op")
            elif renv == "unknown" or env.get("kind") == "unknown":
                t = "review"
            elif not op_ok:
                t = "review"; why.append("no exact approval")
            elif renv == "production" and not (pre_ok and rb_ok):
                t = "review"; why.append("missing preflight/rollback")
            else:
                t = "allow"
        else:
            t = "review"
        if env.get("kind") == "unknown" and t == "allow" and op not in ("workspace_read",):
            t = "review"

    # derived fields
    if req == "block":
        t = "block" if tool else "none"
    elif req == "review" and tool and t != "block":
        t = "review"
    route = "none" if req == "block" else ("restricted" if security != "none" else "primary")
    if req == "block" or t == "block":
        traj = "intervene"
    elif req == "review" or t == "review" or security == "unclear" or redirect:
        traj = "watch"
    else:
        traj = "normal"
    return {"caseId": case.get("caseId"), "requestDecision": req, "toolDecision": t, "route": route,
            "trajectory": traj}, why
