/*
 * Si/Ge 炉子 · 应用层 (app.js)
 * 训练任务 / 控件绑定 / 主循环 / 诊断读数 / 报告。
 * 工具函数、滑块绑定、画布管理、通用图表由 engine.js 提供。
 */
(function (M) {
  "use strict";
  var $ = M.$, P = M.P;
  var fmtHMS = M.fmtHMS, setText = M.setText, setBadge = M.setBadge, clsBadge = M.clsBadge, pct = M.pct, bindSlider = M.bindSlider;

  function logEvent(msg) {
    if (!M.st) return;
    var logs = M.st.logs, last = logs[logs.length - 1];
    if (last && last.msg === msg) return;
    logs.push({ t: M.st.realT, msg: msg }); if (logs.length > 7) logs.shift(); renderLogs();
  }
  function renderLogs() {
    var box = $("eventLog"); if (!box || !M.st) return;
    box.innerHTML = M.st.logs.map(function (it) { return "<div><b>" + fmtHMS(it.t) + "</b><span>" + it.msg + "</span></div>"; }).join("");
  }
  function taskEntry(id) { for (var i = 0; i < M.TASKS.length; i++) if (M.TASKS[i].id === id) return M.TASKS[i]; return M.TASKS[0]; }
  function currentTask() {
    if (M.st && M.activeTaskData && M.activeTaskData.id === M.st.taskId) return M.activeTaskData;
    return taskEntry(M.st ? M.st.taskId : ($("taskSelect") ? $("taskSelect").value : "qw"));
  }
  function setInputValue(id, value, fmt) { var el = $(id), out = $(id + "V"); if (!el || value == null) return; el.value = value; if (out) out.textContent = fmt ? fmt(value) : value; }
  function applyTaskData(data) {
    if (!data) return; M.activeTaskData = data;
    var entry = taskEntry(data.id); if (entry) { entry.title = data.title || entry.title; entry.target = data.target || entry.target; }
    var r = data.recipe || {}, c = data.conditions || {};
    if (r.nPer != null) P.nPer = r.nPer;
    if (r.siThick != null) P.siThick = r.siThick;
    if (r.sigeThick != null) P.sigeThick = r.sigeThick;
    if (r.geFrac != null) P.geFrac = r.geFrac;
    if (c.tSub != null) P.tSub = c.tSub;
    if (c.powSi != null) P.powSi = c.powSi;
    if (c.powGe != null) P.powGe = c.powGe;
    if (c.dope != null) P.dope = !!c.dope;
    M.updateCellStates();
    setInputValue("nPer", P.nPer);
    setInputValue("siThick", P.siThick, function (v) { return Number(v).toFixed(1) + " nm"; });
    setInputValue("sigeThick", P.sigeThick, function (v) { return Number(v).toFixed(1) + " nm"; });
    setInputValue("geFrac", P.geFrac, function (v) { return Number(v).toFixed(2); });
    setInputValue("tSub", P.tSub, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("powSi", P.powSi, function (v) { return Number(v).toFixed(0) + " %"; });
    setInputValue("powGe", P.powGe, function (v) { return Number(v).toFixed(0) + " %"; });
    if ($("dope")) $("dope").checked = P.dope;
  }
  async function loadTaskData(taskId) {
    var entry = taskEntry(taskId); M.activeTaskData = null;
    if (!entry || !entry.dataUrl || !window.fetch) return entry;
    try { var res = await window.fetch(entry.dataUrl, { cache: "no-store" }); if (!res.ok) throw 0; var data = await res.json(); applyTaskData(data); return data; }
    catch (e) { return entry; }
  }

  function syncAutoShutters() { if (!M.manualShutters) M.setShutters(M.stepShutters(M.curStep())); }
  function updateShutterButtons() {
    var st = M.st, guided = st && st.mode === "引导模式" && !st.done, rec = guided ? M.stepShutters(M.curStep()) : null;
    M.ELEMENTS.forEach(function (k) { var btn = $("sh" + k); if (btn) btn.className = "shutter " + (M.shutters[k] ? "on" : "") + (rec && rec[k] ? " guide" : ""); });
    var chem = M.shutterChemistry();
    setText("chemNow", chem.txt);
    clsBadge("chemRisk", chem.risk === "high" ? "风险高" : chem.risk === "med" ? "需注意" : chem.risk === "idle" ? "待机" : "正常", chem.cls);
  }

  function updateReadout() {
    var st = M.st, Q = M.surfaceQuality(), s = M.curStep(), STAGES = M.STAGES, x = M.geFraction();
    syncAutoShutters();
    for (var i = 0; i < STAGES.length; i++) { var el = $("stg_" + STAGES[i].id); if (el) el.className = "stg" + (STAGES[i].id === M.stageOf() ? " on" : (M.stageIdx(STAGES[i].id) < M.stageIdx(M.stageOf()) ? " done" : "")); }
    if ($("progBar")) $("progBar").style.width = (st.realT / st.totalReal * 100).toFixed(1) + "%";
    setText("taskTitle", currentTask().title); setText("taskTarget", currentTask().target);
    setText("stageTitle", s.label); setText("modeTitle", st.mode);
    setText("cycleNow", M.curPeriod() + " / " + P.nPer);
    setText("shutterNow", M.shutterChemistry().txt);
    setText("layerNow", (s.mat || "Thermal") + " · " + (s.nm ? (s.nm * st.sp).toFixed(2) + " / " + s.nm.toFixed(2) + " nm" : "0 nm"));

    setText("eSub", (st ? st.actTSub : M.subTemp()).toFixed(1) + " ℃");
    setText("eRot", M.CELL.rot.toFixed(1) + " rpm");
    setText("eSi", "e-beam " + P.powSi.toFixed(0) + "% (" + M.siRate().toFixed(3) + " nm/s)");
    setText("eGe", "e-beam " + P.powGe.toFixed(0) + "% (" + M.geRate().toFixed(3) + " nm/s)");
    setText("eB", P.dope ? P.tempB.toFixed(0) + " ℃ (ON)" : "OFF");
    setText("eStep", (st.si + 1) + " / " + st.steps.length);
    setText("ePeriod", M.curPeriod() + " / " + P.nPer);
    setText("eElapsed", fmtHMS(st.realT)); setText("eRemain", fmtHMS(st.totalReal - st.realT));
    setText("eLabel", s.label); setText("eThick", st.thickNm.toFixed(2) + " nm");

    var rheed = M.rheedState(Q), risk = M.relaxRisk(), sc = M.score();
    setText("scoreV", sc.total + " / 100"); clsBadge("scoreGrade", sc.grade, sc.cls);
    setText("qV", (Q * 100).toFixed(0) + " %");
    setText("strainV", st.accStrain.toFixed(0)); setBadge("strainStat", M.strainStatus(st.accStrain));
    setText("geActual", x.toFixed(3)); setText("geTarget", P.geFrac.toFixed(2));
    setText("egV", M.sigeBandgap(x).toFixed(3) + " eV");
    setText("misfitV", (4.18 * P.geFrac).toFixed(2) + " %");
    clsBadge("relaxRisk", risk.txt, risk.cls); setText("relaxWhy", risk.why);
    setBadge("rheedState", rheed); setText("surfaceDiag", rheed.desc);

    if (sc.parts) { setText("partRecipe", sc.parts.recipe.toFixed(0)); setText("partSurface", sc.parts.surface.toFixed(0)); setText("partStrain", sc.parts.strain.toFixed(0)); setText("partGe", sc.parts.gecomp.toFixed(0)); setText("partOps", sc.parts.ops.toFixed(0)); }

    updateShutterButtons(); renderLogs();
    if (Q < 0.55) logEvent("RHEED 变差：表面粗化，检查温度/Ge 组分/速率");
    if (Math.abs(st.accStrain) > M.STRAIN_CRIT) logEvent("应变超过临界厚度：SiGe 弛豫、失配位错风险升高");
  }
  M.updateReadout = updateReadout;

  function showSummary() {
    var st = M.st, x = M.geFraction(), avg = st.iCount ? st.iSum / st.iCount : 0, ss = M.strainStatus(st.accStrain), sc = M.score(), risk = M.relaxRisk();
    setText("sStruct", "SiGe " + P.sigeThick.toFixed(1) + "/Si " + P.siThick.toFixed(1) + " nm × " + P.nPer + " 周期, x=" + x.toFixed(2));
    setText("sScore", sc.total + " / 100"); setText("sThick", st.thickNm.toFixed(0) + " nm");
    setText("sFlat", (avg * 100).toFixed(0) + " %"); setText("sStrain", st.accStrain.toFixed(0) + " (" + ss.txt + ")");
    setText("sGe", x.toFixed(3)); setText("sEg", M.sigeBandgap(x).toFixed(3) + " eV"); setText("sRisk", risk.txt);
    setText("sMode", M.rheedState(M.surfaceQuality()).txt);
    if ($("summary")) $("summary").classList.add("show");
    logEvent("训练完成：生成评分与工艺复盘");
  }
  function hideSummary() { if ($("summary")) $("summary").classList.remove("show"); }

  var lastTs = 0;
  function loop(ts) {
    if (!lastTs) lastTs = ts;
    var dt = Math.min(0.05, (ts - lastTs) / 1000); lastTs = ts;
    var st = M.st, Q = M.surfaceQuality(), running = st && st.playing && !st.done;
    if (running) {
      st.realT += dt * P.sim * st.speedMul;
      if (st.realT >= st.totalReal) { st.realT = st.totalReal; st.done = true; st.playing = false; $("play").textContent = "重新生长"; }
      var si = M.locate(st.realT), s = st.steps[si], frac = M.clamp((st.realT - s.t0) / s.dur, 0, 1);
      if (si !== st.si) logEvent("进入步骤：" + s.label);
      st.si = si; st.sp = frac;
      st.thickNm = s.cumNm0 + s.nm * frac; st.accStrain = s.cumStrain0 + s.rate * s.nm * frac;
      st.phase = st.thickNm / M.ML_NM;
      st.strainHist.push({ t: st.realT, s: st.accStrain }); if (st.strainHist.length > 5000) st.strainHist.shift();
      var oscVal = s.mat ? M.rheedClean(st.phase) : 0.5;
      var curI = (0.2 + 0.8 * Q) * (0.6 + 0.4 * oscVal) * (1 - st.oxideOpacity * 0.85);
      st.rheedHist.push({ t: st.realT, v: curI }); if (st.rheedHist.length > 5000) st.rheedHist.shift();
      var targetT = M.lerp(s.subA, s.subB, frac); if (s.stage === "sl") targetT = P.tSub;
      var dtSim = dt * P.sim * st.speedMul;
      st.actTSub = st.actTSub + (targetT - st.actTSub) * (1 - Math.exp(-dtSim / 45.0));
      if (s.stage === "deoxide") {
        if (st.deoxideDoneLogged) st.oxideOpacity = 0.0;
        else if (st.actTSub >= 800) {
          st.oxideOpacity = Math.max(0, 1 - (st.actTSub - 800) / 70);
          if (st.oxideOpacity === 0 && !st.deoxideDoneLogged) { logEvent("Si 衬底脱氧成功：SiO2 脱附，2×1 重构条纹显现！"); st.deoxideDoneLogged = true; }
          else if (st.oxideOpacity < 1.0 && !st.deoxideStartLogged) { logEvent("Si 表面脱氧中：温度达 SiO2 解离点，氧化层开始升华..."); st.deoxideStartLogged = true; }
        } else st.oxideOpacity = 1.0;
      } else if (s.stage === "load") st.oxideOpacity = 1.0; else st.oxideOpacity = 0.0;
      if (s.stage === "sl") { st.history.surfaceQualities.push(Q); var chem = M.shutterChemistry(); st.history.shutterRisks.push(chem.risk === "high" ? 0.45 : chem.risk === "med" ? 0.7 : 0.92); }
      syncAutoShutters(); M.recordRHEED(Q);
      if (st.done) showSummary();
      updateReadout();
    }
    if (running || M.needsRedraw) { M.drawChamber(Q); M.drawStackMap(); M.drawStrain(); M.drawRheed(); M.drawBand(); M.needsRedraw = false; }
    M.drawScreen(Q);
    requestAnimationFrame(loop);
  }

  function pullInputs() {
    P.nPer = parseFloat($("nPer").value); P.siThick = parseFloat($("siThick").value);
    P.sigeThick = parseFloat($("sigeThick").value); P.geFrac = parseFloat($("geFrac").value);
    P.powSi = parseFloat($("powSi").value); P.powGe = parseFloat($("powGe").value);
    if ($("tempB")) P.tempB = parseFloat($("tempB").value);
  }
  function doReset() {
    var taskId = $("taskSelect") ? $("taskSelect").value : "qw";
    pullInputs(); M.manualShutters = false; M.st = M.freshState(); M.atoms = [];
    M.st.taskId = taskId; M.st.mode = "引导模式";
    M.st.logs = [{ t: 0, msg: "训练任务载入：" + currentTask().title }];
    $("play").textContent = "开始生长"; hideSummary(); syncAutoShutters(); M.needsRedraw = true; updateReadout();
  }
  function setSequence(name) {
    M.manualShutters = true;
    if (name === "si") M.setShutters({ Si: true, Ge: false, B: !!P.dope });
    if (name === "sige") M.setShutters({ Si: true, Ge: true, B: !!P.dope });
    if (name === "ge") M.setShutters({ Si: false, Ge: true, B: !!P.dope });
    if (name === "auto") { M.manualShutters = false; syncAutoShutters(); }
    logEvent("快门切换：" + M.shutterChemistry().txt); M.needsRedraw = true; updateReadout();
  }

  async function init() {
    M.registerCanvas("chamber", "cx", "chW", "chH");
    M.registerCanvas("stackMap", "mcx", "mpW", "mpH");
    M.registerCanvas("strain", "sx", "stW", "stH");
    M.registerCanvas("screen", "scx", "scW", "scH");
    M.registerCanvas("rheed", "rx", "rhW", "rhH");
    M.registerCanvas("band", "bcx", "bdW", "bdH");
    M.resizeAll();
    var rz; window.addEventListener("resize", function () { clearTimeout(rz); rz = setTimeout(M.resizeAll, 100); });

    if ($("fluxTable")) $("fluxTable").innerHTML = M.FLUX.map(function (f) { return "<span><b>" + f[0] + "</b> " + f[1] + "</span>"; }).join("");

    if ($("taskSelect")) $("taskSelect").addEventListener("change", async function () { await loadTaskData(this.value); doReset(); logEvent("训练任务切换：" + currentTask().title); });
    $("play").addEventListener("click", function () { if (M.st.done) doReset(); M.st.playing = !M.st.playing; M.st.speedMul = 1; this.textContent = M.st.playing ? "暂停" : "继续"; M.needsRedraw = true; });
    $("reset").addEventListener("click", doReset);
    $("fast").addEventListener("click", function () { if (M.st.done) doReset(); M.st.playing = true; M.st.speedMul = 10; $("play").textContent = "暂停"; M.needsRedraw = true; });

    bindSlider("nPer", "nPer", null, doReset);
    bindSlider("siThick", "siThick", function (v) { return v.toFixed(1) + " nm"; }, doReset);
    bindSlider("sigeThick", "sigeThick", function (v) { return v.toFixed(1) + " nm"; }, doReset);
    bindSlider("geFrac", "geFrac", function (v) { return v.toFixed(2); }, doReset);
    bindSlider("tSub", "tSub", function (v) { return v.toFixed(0) + " ℃"; });
    bindSlider("powSi", "powSi", function (v) { return v.toFixed(0) + " %"; }, function () { M.updateCellStates(); doReset(); });
    bindSlider("powGe", "powGe", function (v) { return v.toFixed(0) + " %"; }, function () { M.updateCellStates(); doReset(); });
    bindSlider("tempB", "tempB", function (v) { return v.toFixed(0) + " ℃"; }, function () { M.updateCellStates(); });
    if ($("dope")) $("dope").addEventListener("change", function () { P.dope = this.checked; M.updateCellStates(); doReset(); });

    M.ELEMENTS.forEach(function (k) { var b = $("sh" + k); if (b) b.addEventListener("click", function () { M.manualShutters = true; M.shutters[k] = !M.shutters[k]; logEvent("手动快门：" + k + " " + (M.shutters[k] ? "ON" : "OFF")); M.needsRedraw = true; updateReadout(); }); });
    ["si", "sige", "ge", "auto"].forEach(function (k) { var b = $("seq_" + k); if (b) b.addEventListener("click", function () { setSequence(k); }); });

    await loadTaskData($("taskSelect") ? $("taskSelect").value : "qw");
    doReset(); requestAnimationFrame(loop);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(window.MBE);
