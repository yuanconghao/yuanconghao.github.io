/*
 * InAs/GaSb 炉子 · 应用层 (app.js)
 * 训练任务 / 控件绑定 / 主循环 / 诊断读数 / 报告。
 * 工具函数(fmtHMS/setText/...)、滑块绑定、画布管理由 engine.js 提供。
 */
(function (M) {
  "use strict";
  var $ = M.$, P = M.P;
  var fmtHMS = M.fmtHMS, setText = M.setText, setBadge = M.setBadge, clsBadge = M.clsBadge, pct = M.pct, bindSlider = M.bindSlider;

  function logEvent(msg) {
    if (!M.st) return;
    var logs = M.st.logs, last = logs[logs.length - 1];
    if (last && last.msg === msg) return;
    logs.push({ t: M.st.realT, msg: msg });
    if (logs.length > 7) logs.shift();
    renderLogs();
  }
  function renderLogs() {
    var box = $("eventLog");
    if (!box || !M.st) return;
    box.innerHTML = M.st.logs.map(function (it) {
      return "<div><b>" + fmtHMS(it.t) + "</b><span>" + it.msg + "</span></div>";
    }).join("");
  }

  function taskEntry(id) {
    for (var i = 0; i < M.TASKS.length; i++) if (M.TASKS[i].id === id) return M.TASKS[i];
    return M.TASKS[0];
  }
  function currentTask() {
    if (M.st && M.activeTaskData && M.activeTaskData.id === M.st.taskId) return M.activeTaskData;
    return taskEntry(M.st ? M.st.taskId : $("taskSelect").value);
  }

  function setInputValue(id, value, fmt) {
    var el = $(id), out = $(id + "V");
    if (!el || value == null) return;
    el.value = value;
    if (out) out.textContent = fmt ? fmt(value) : value;
  }
  function applyTaskData(data) {
    if (!data) return;
    M.activeTaskData = data;
    var entry = taskEntry(data.id);
    if (entry) { entry.title = data.title || entry.title; entry.target = data.target || entry.target; }
    var r = data.recipe || {}, c = data.conditions || {}, itf = data.interface || {};
    if (r.nPer != null) P.nPer = r.nPer;
    if (r.inas != null) P.inas = r.inas;
    if (r.gasb != null) P.gasb = r.gasb;
    if (r.targetLam != null) P.targetLam = r.targetLam;
    if (c.tSub != null) P.tSub = c.tSub;
    if (c.cellTemp != null) {
      if (c.cellTemp.in != null) P.tempIn = c.cellTemp.in;
      if (c.cellTemp.ga != null) P.tempGa = c.cellTemp.ga;
      if (c.cellTemp.al != null) P.tempAl = c.cellTemp.al;
      if (c.cellTemp.as != null) P.tempAs = c.cellTemp.as;
      if (c.cellTemp.sb != null) P.tempSb = c.cellTemp.sb;
    }
    P.alEnabled = !!(c.alEnabled || (c.cellTemp && c.cellTemp.al != null && c.cellTemp.al > 0));
    if (itf.comp != null) P.comp = !!itf.comp;
    if (itf.insb != null) P.insb = itf.insb;
    if (itf.soak != null) P.soak = itf.soak;
    if (itf.switchDelay != null) P.switchDelay = itf.switchDelay;

    M.updateRatios();
    M.updateCellStates();

    setInputValue("nPer", P.nPer);
    setInputValue("inas", P.inas, function (v) { return Number(v).toFixed(2) + " nm"; });
    setInputValue("gasb", P.gasb, function (v) { return Number(v).toFixed(2) + " nm"; });
    setInputValue("targetLam", P.targetLam, function (v) { return Number(v).toFixed(1) + " μm"; });
    setInputValue("tSub", P.tSub, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("tempIn", P.tempIn, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("tempGa", P.tempGa, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("tempAl", P.tempAl, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("tempAs", P.tempAs, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("tempSb", P.tempSb, function (v) { return Number(v).toFixed(0) + " ℃"; });
    setInputValue("insb", P.insb, function (v) { return Number(v).toFixed(3) + " nm"; });
    setInputValue("soak", P.soak, function (v) { return Number(v).toFixed(1) + " s"; });
    setInputValue("switchDelay", P.switchDelay, function (v) { return Number(v).toFixed(1) + " s"; });
    $("comp").checked = P.comp;

    var alRow = $("alRow"), alHint = $("tempAlRowHint");
    if (alRow) alRow.style.display = P.alEnabled ? "flex" : "none";
    if (alHint) alHint.style.display = P.alEnabled ? "flex" : "none";
  }
  async function loadTaskData(taskId) {
    var entry = taskEntry(taskId);
    M.activeTaskData = null;
    if (!entry || !entry.dataUrl || !window.fetch) return entry;
    try {
      var res = await window.fetch(entry.dataUrl, { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      var data = await res.json();
      applyTaskData(data);
      return data;
    } catch (err) {
      return entry;
    }
  }

  function syncAutoShutters() {
    if (!M.manualShutters) M.setShutters(M.stepShutters(M.curStep()));
  }
  function updateShutterButtons() {
    var st = M.st;
    var isGuided = st && st.mode === "引导模式" && !st.done;
    var rec = isGuided ? M.stepShutters(M.curStep()) : null;
    ["In", "Ga", "Al", "As", "Sb"].forEach(function (k) {
      var btn = $("sh" + k);
      if (btn) {
        var isRecommended = rec && rec[k];
        btn.className = "shutter " + (M.shutters[k] ? "on" : "") + (isRecommended ? " guide" : "");
      }
    });
    var chem = M.shutterChemistry();
    setText("chemNow", chem.txt);
    clsBadge("chemRisk", chem.risk === "high" ? "风险高" : chem.risk === "med" ? "需注意" : chem.risk === "idle" ? "待机" : "正常", chem.cls);
  }

  function updateConditionHints() {
    var st = M.st;
    var isGuided = st ? st.mode === "引导模式" : true;
    
    // Hide standard layout recommendations in Free & Exam Modes
    var displays = document.querySelectorAll(".group-feedback, .hint");
    displays.forEach(function (el) {
      el.style.display = isGuided ? "" : "none";
    });
    
    // Toggle the timing reference cheatsheet card
    var timingCard = $("timing") ? $("timing").closest(".card") : null;
    if (timingCard) {
      var isExam = st && st.mode === "考核模式" && !st.done;
      timingCard.style.display = isExam ? "none" : "block";
    }

    if (isGuided) {
      var t = M.rangeStatus(P.tSub, 370, 410, " ℃");
      var tin = M.rangeStatus(P.tempIn, 770, 790, " ℃");
      var tga = M.rangeStatus(P.tempGa, 890, 910, " ℃");
      var tal = M.rangeStatus(P.tempAl, 1030, 1070, " ℃");
      var tas = M.rangeStatus(P.tempAs, 300, 320, " ℃");
      var tsb = M.rangeStatus(P.tempSb, 420, 440, " ℃");

      clsBadge("tSubStat", t.txt, t.cls); setText("tSubHint", t.note + "；控制表面迁移、脱附和互扩散");
      clsBadge("tempInStat", tin.txt, tin.cls); setText("tempInHint", tin.note + "；主控 InAs 生长速率与层厚");
      clsBadge("tempGaStat", tga.txt, tga.cls); setText("tempGaHint", tga.note + "；主控 GaSb 生长速率与层厚");
      clsBadge("tempAlStat", tal.txt, tal.cls); setText("tempAlHint", tal.note + "；主控势垒层 AlSb 组分与厚度");
      clsBadge("tempAsStat", tas.txt, tas.cls); setText("tempAsHint", tas.note + "；主控 As/In 气压比与 InAs 重构质量");
      clsBadge("tempSbStat", tsb.txt, tsb.cls); setText("tempSbHint", tsb.note + "；主控 Sb/Ga 气压比与 InSb 界面补偿效果");
    } else {
      clsBadge("tSubStat", "--", "warn");
      clsBadge("tempInStat", "--", "warn");
      clsBadge("tempGaStat", "--", "warn");
      clsBadge("tempAlStat", "--", "warn");
      clsBadge("tempAsStat", "--", "warn");
      clsBadge("tempSbStat", "--", "warn");
    }
  }

  function updateReadout() {
    var st = M.st, Q = M.surfaceQuality(), s = M.curStep(), STAGES = M.STAGES, eg = M.effGap(), lam = M.cutoff(eg);
    var chem = M.shutterChemistry();
    syncAutoShutters();
    for (var i = 0; i < STAGES.length; i++) {
      $("stg_" + STAGES[i].id).className = "stg" + (STAGES[i].id === M.stageOf() ? " on" : (M.stageIdx(STAGES[i].id) < M.stageIdx(M.stageOf()) ? " done" : ""));
    }
    $("progBar").style.width = (st.realT / st.totalReal * 100).toFixed(1) + "%";
    setText("taskTitle", currentTask().title);
    setText("taskTarget", currentTask().target);
    setText("taskEvidence", M.activeTaskData && M.activeTaskData.evidence ? M.activeTaskData.evidence[0] : "任务数据文件未加载时使用内置任务目标。");
    setText("stageTitle", s.label);
    setText("modeTitle", st.mode);
    setText("cycleNow", M.curPeriod() + " / " + P.nPer);
    setText("shutterNow", chem.txt);
    setText("layerNow", (s.mat || "Thermal") + " · " + (s.nm ? (s.nm * st.sp).toFixed(2) + " / " + s.nm.toFixed(2) + " nm" : "0 nm"));
    setText("stackMapTitle", P.nPer + " 周期全栈压缩总览 · 主画布显示当前局部放大");
    setText("stackMapNote", "横向表示生长顺序：GaSb 衬底 → GaSb buffer → 周期序列；每个周期从左到右为 InSb-like / InAs / InSb-like / GaSb，红框为当前周期。");

    setText("eSub", (st ? st.actTSub : M.subTemp()).toFixed(1) + " ℃");
    setText("eRot", M.CELL.rot.toFixed(2) + " rpm");
    setText("eGa", M.CELL.Ga.temp.toFixed(0) + " ℃ (" + M.CELL.Ga.bep + ")");
    setText("eIn", M.CELL.In.temp.toFixed(0) + " ℃ (" + M.CELL.In.bep + ")");
    setText("eAl", P.alEnabled ? M.CELL.Al.temp.toFixed(0) + " ℃ (" + M.CELL.Al.bep + ")" : "OFF");
    
    var actSb = (s.mat === "GaSb" || s.mat === "InSb" || s.mat === "AlSb" || s.label.indexOf("soak") >= 0);
    var actAs = (s.mat === "InAs");
    var bepSbVal = actSb ? M.cellTempToBEP("Sb", P.tempSb) : 0;
    var bepAsVal = actAs ? M.cellTempToBEP("As", P.tempAs) : 0;
    setText("eSb", M.fmtBEP(bepSbVal) + " (Sb/Ga: " + P.sbGa.toFixed(2) + ")");
    setText("eAs", M.fmtBEP(bepAsVal) + " (As/In: " + P.asIn.toFixed(2) + ")");
    setText("asInV", P.asIn.toFixed(2));
    setText("sbGaV", P.sbGa.toFixed(2));
    setText("asInReadout", P.asIn.toFixed(2));
    setText("sbGaReadout", P.sbGa.toFixed(2));
    
    var In_RATE = 0.1034, Ga_RATE = 0.1148;
    var rate_InAs = In_RATE * M.cellTempToFlux("In", P.tempIn);
    var rate_GaSb = Ga_RATE * M.cellTempToFlux("Ga", P.tempGa);
    setText("inasRateReadout", rate_InAs.toFixed(3) + " nm/s");
    setText("gasbRateReadout", rate_GaSb.toFixed(3) + " nm/s");
    setText("eStep", (st.si + 1) + " / " + st.steps.length);
    setText("ePeriod", M.curPeriod() + " / " + P.nPer);
    setText("eElapsed", fmtHMS(st.realT));
    setText("eRemain", fmtHMS(st.totalReal - st.realT));
    setText("eLabel", s.label);
    setText("eThick", st.thickNm.toFixed(2) + " nm");

    var intf = M.interfaceMetrics(), rheed = M.rheedState(Q), risk = M.darkCurrentRisk(), sc = M.score();
    setText("scoreV", sc.total + " / 100");
    clsBadge("scoreGrade", sc.grade, sc.cls);
    setText("qV", (Q * 100).toFixed(0) + " %");
    setText("strainV", st.accStrain.toFixed(0));
    setBadge("strainStat", M.strainStatus(st.accStrain));
    setText("eg", eg.toFixed(3) + " eV");
    setText("lam", lam.toFixed(1) + " μm");
    setText("bandName", M.bandName(lam));
    
    setText("intfInSb", pct(intf.inSbLike));
    setText("intfGaAs", pct(intf.gaAsLike));
    clsBadge("intfAbrupt", intf.text, intf.cls);
    setText("strainDiag", M.strainStatus(st.accStrain).txt);
    setText("lambdaMatch", pct(M.targetMatch()));
    setText("opticalDiag", "目标 " + P.targetLam.toFixed(1) + " μm，当前偏差 " + Math.abs(lam - P.targetLam).toFixed(1) + " μm");
    clsBadge("darkRisk", risk.txt, risk.cls);
    setText("darkWhy", risk.why);
    
    setText("partRecipe", sc.parts.recipe.toFixed(0));
    setText("partSurface", sc.parts.surface.toFixed(0));
    setText("partInterface", sc.parts.interface.toFixed(0));
    setText("partStrain", sc.parts.strain.toFixed(0));
    setText("partLambda", sc.parts.lambda.toFixed(0));
    setText("partOps", sc.parts.ops.toFixed(0));

    setText("insbHint", "理论平衡约 " + M.balanceInSb().toFixed(2) + " nm/界面");
    $("insbRow").style.opacity = P.comp ? "1" : ".45";

    setText("surfaceDiag", rheed.desc); setBadge("rheedState", rheed);

    updateConditionHints();
    updateShutterButtons();
    renderLogs();

    if (Q < 0.55) logEvent("RHEED 变差：表面粗糙度增加，建议检查温度/V-III/速率");
    if (M.shutterChemistry().mat === "GaAs") logEvent("快门组合 Ga + As：正在形成 GaAs-like 界面");
    if (Math.abs(st.accStrain) > M.STRAIN_CRIT) logEvent("应变超过临界线：位错和暗电流风险显著升高");
  }
  M.updateReadout = updateReadout;

  function showSummary() {
    var st = M.st, eg = M.effGap(), lam = M.cutoff(eg), avg = st.iCount ? st.iSum / st.iCount : 0, ss = M.strainStatus(st.accStrain), sc = M.score(), risk = M.darkCurrentRisk();
    setText("sStruct", P.inas.toFixed(2) + "/" + P.gasb.toFixed(2) + " nm x " + P.nPer + " periods");
    setText("sScore", sc.total + " / 100");
    setText("sThick", st.thickNm.toFixed(0) + " nm");
    setText("sFlat", (avg * 100).toFixed(0) + " %");
    setText("sStrain", st.accStrain.toFixed(0) + " (" + ss.txt + ")");
    setText("sMode", M.rheedState(M.surfaceQuality()).txt);
    setText("sEg", eg.toFixed(3) + " eV"); setText("sLam", lam.toFixed(1) + " μm"); setText("sBand", M.bandName(lam));
    setText("sRisk", risk.txt);
    $("summary").classList.add("show");
    logEvent("训练完成：生成评分与工艺复盘报告");
  }
  function hideSummary() { $("summary").classList.remove("show"); }

  var lastTs = 0;
  function loop(ts) {
    if (!lastTs) lastTs = ts;
    var dt = Math.min(0.05, (ts - lastTs) / 1000); lastTs = ts;
    var st = M.st, Q = M.surfaceQuality();
    var running = st && st.playing && !st.done;
    if (running) {
      st.realT += dt * P.sim * st.speedMul;
      if (st.realT >= st.totalReal) {
        st.realT = st.totalReal; st.done = true; st.playing = false; $("play").textContent = "重新生长";
      }
      var si = M.locate(st.realT), s = st.steps[si], frac = M.clamp((st.realT - s.t0) / s.dur, 0, 1);
      if (si !== st.si) logEvent("进入步骤：" + s.label);
      st.si = si; st.sp = frac;
      st.thickNm = s.cumNm0 + s.nm * frac;
      st.accStrain = s.cumStrain0 + s.rate * s.nm * frac;
      st.phase = st.thickNm / M.ML_NM;
      st.strainHist.push({ t: st.realT, s: st.accStrain });
      if (st.strainHist.length > 5000) st.strainHist.shift();

      var growing = !!s.mat;
      var oscVal = growing ? M.rheedClean(st.phase) : 0.5;
      var curI = (0.2 + 0.8 * Q) * (0.6 + 0.4 * oscVal) * (1 - st.oxideOpacity * 0.85);
      st.rheedHist.push({ t: st.realT, v: curI });
      if (st.rheedHist.length > 5000) st.rheedHist.shift();

          // Thermal inertia calculation (actual temperature response)
      var targetT = M.lerp(s.subA, s.subB, frac);
      if (s.stage === "sl") targetT = P.tSub;
      var dtSim = dt * P.sim * st.speedMul;
      st.actTSub = st.actTSub + (targetT - st.actTSub) * (1 - Math.exp(-dtSim / 45.0));

      // Oxide desorption calculation
      if (s.stage === "deoxide") {
        if (st.deoxideDoneLogged) {
          st.oxideOpacity = 0.0;
        } else if (st.actTSub >= 480) {
          st.oxideOpacity = Math.max(0, 1 - (st.actTSub - 480) / 40);
          if (st.oxideOpacity === 0 && !st.deoxideDoneLogged) {
            logEvent("衬底脱氧成功：氧化层完全脱附，晶面重建条纹显现！");
            st.deoxideDoneLogged = true;
          } else if (st.oxideOpacity < 1.0 && !st.deoxideStartLogged) {
            logEvent("衬底表面脱氧中：温度达到氧化物解离点，氧化层开始分解脱附...");
            st.deoxideStartLogged = true;
          }
        } else {
          st.oxideOpacity = 1.0;
        }
      } else if (s.stage === "load") {
        st.oxideOpacity = 1.0;
      } else {
        st.oxideOpacity = 0.0;
      }

      // Record real-time metrics for dynamic history evaluation during SL Growth
      if (s.stage === "sl") {
        st.history.surfaceQualities.push(Q);
        var intf = M.interfaceMetrics();
        st.history.abruptnessValues.push(intf.abruptness);
        st.history.inSbLikeValues.push(intf.inSbLike);
        st.history.gaAsLikeValues.push(intf.gaAsLike);
        var chem = M.shutterChemistry();
        st.history.shutterRisks.push(chem.risk === "high" ? 0.45 : chem.risk === "med" ? 0.68 : 0.92);
      }

      syncAutoShutters();
      M.recordRHEED(Q);
      if (st.done) showSummary();
      updateReadout();
    }
    
    // Throttled Canvas drawing (using M.needsRedraw)
    if (running || M.needsRedraw) {
      M.drawChamber(Q);
      M.drawStackMap();
      M.drawStrain();
      M.drawRheed();
      M.drawTiming();
      M.drawBand();
      M.needsRedraw = false;
    }
    M.drawScreen(Q); // RHEED screen is continuously drawn for phase oscillations and visual phosphor noise
    requestAnimationFrame(loop);
  }

  function pullStaticInputs() {
    P.nPer = parseFloat($("nPer").value); P.inas = parseFloat($("inas").value);
    P.gasb = parseFloat($("gasb").value); P.insb = parseFloat($("insb").value);
    P.targetLam = parseFloat($("targetLam").value);
    P.soak = parseFloat($("soak").value); P.switchDelay = parseFloat($("switchDelay").value);
    P.tempIn = parseFloat($("tempIn").value); P.tempGa = parseFloat($("tempGa").value);
    P.tempAl = parseFloat($("tempAl").value); P.tempAs = parseFloat($("tempAs").value);
    P.tempSb = parseFloat($("tempSb").value);
  }
  function doReset() {
    var taskId = $("taskSelect").value;
    pullStaticInputs();
    M.manualShutters = false; M.st = M.freshState(); M.atoms = [];
    M.st.taskId = taskId; M.st.mode = "引导模式";
    M.st.logs = [{ t: 0, msg: "训练任务载入：" + currentTask().title }];
    $("play").textContent = "开始生长"; hideSummary(); syncAutoShutters();
    M.needsRedraw = true;
    updateReadout();
  }
  function setSequence(name) {
    M.manualShutters = true;
    if (name === "inas") M.setShutters({ In: true, Ga: false, As: true, Sb: false });
    if (name === "insb") M.setShutters({ In: true, Ga: false, As: false, Sb: true });
    if (name === "gasb") M.setShutters({ In: false, Ga: true, As: false, Sb: true });
    if (name === "soak") M.setShutters({ In: false, Ga: false, As: false, Sb: true });
    if (name === "auto") { M.manualShutters = false; syncAutoShutters(); }
    logEvent("快门序列切换：" + M.shutterChemistry().txt);
    M.needsRedraw = true;
    updateReadout();
  }

  async function init() {
    M.registerCanvas("chamber", "cx", "chW", "chH");
    M.registerCanvas("stackMap", "mcx", "mpW", "mpH");
    M.registerCanvas("strain", "sx", "stW", "stH");
    M.registerCanvas("screen", "scx", "scW", "scH");
    M.registerCanvas("rheed", "rx", "rhW", "rhH");
    M.registerCanvas("timing", "tcx", "tmW", "tmH");
    M.registerCanvas("band", "bcx", "bdW", "bdH");
    M.resizeAll();
    var rz; window.addEventListener("resize", function () { clearTimeout(rz); rz = setTimeout(M.resizeAll, 100); });

    $("fluxTable").innerHTML = M.FLUX.map(function (f) { return "<span><b>" + f[0] + "</b> " + f[1] + "</span>"; }).join("");

    $("taskSelect").addEventListener("change", async function () { await loadTaskData(this.value); doReset(); logEvent("训练任务切换：" + currentTask().title); });
    $("play").addEventListener("click", function () { if (M.st.done) doReset(); M.st.playing = !M.st.playing; M.st.speedMul = 1; this.textContent = M.st.playing ? "暂停" : "继续"; M.needsRedraw = true; });
    $("reset").addEventListener("click", doReset);
    $("fast").addEventListener("click", function () { if (M.st.done) doReset(); M.st.playing = true; M.st.speedMul = 10; $("play").textContent = "暂停"; M.needsRedraw = true; });

    bindSlider("nPer", "nPer", null, doReset);
    bindSlider("inas", "inas", function (v) { return v.toFixed(2) + " nm"; }, doReset);
    bindSlider("gasb", "gasb", function (v) { return v.toFixed(2) + " nm"; }, doReset);
    bindSlider("insb", "insb", function (v) { return v.toFixed(3) + " nm"; }, doReset);
    bindSlider("targetLam", "targetLam", function (v) { return v.toFixed(1) + " μm"; }, doReset);
    bindSlider("tSub", "tSub", function (v) { return v.toFixed(0) + " ℃"; });
    bindSlider("tempIn", "tempIn", function (v) { return v.toFixed(0) + " ℃"; }, function () { M.updateRatios(); M.updateCellStates(); doReset(); });
    bindSlider("tempGa", "tempGa", function (v) { return v.toFixed(0) + " ℃"; }, function () { M.updateRatios(); M.updateCellStates(); doReset(); });
    bindSlider("tempAl", "tempAl", function (v) { return v.toFixed(0) + " ℃"; }, function () { M.updateRatios(); M.updateCellStates(); doReset(); });
    bindSlider("tempAs", "tempAs", function (v) { return v.toFixed(0) + " ℃"; }, function () { M.updateRatios(); M.updateCellStates(); });
    bindSlider("tempSb", "tempSb", function (v) { return v.toFixed(0) + " ℃"; }, function () { M.updateRatios(); M.updateCellStates(); });
    bindSlider("soak", "soak", function (v) { return v.toFixed(1) + " s"; }, doReset);
    bindSlider("switchDelay", "switchDelay", function (v) { return v.toFixed(1) + " s"; }, doReset);
    $("comp").addEventListener("change", function () { P.comp = this.checked; doReset(); });

    ["In", "Ga", "Al", "As", "Sb"].forEach(function (k) {
      $("sh" + k).addEventListener("click", function () { M.manualShutters = true; M.shutters[k] = !M.shutters[k]; logEvent("手动快门：" + k + " " + (M.shutters[k] ? "ON" : "OFF")); M.needsRedraw = true; updateReadout(); });
    });
    ["inas", "insb", "gasb", "soak", "auto"].forEach(function (k) { $("seq_" + k).addEventListener("click", function () { setSequence(k); M.needsRedraw = true; }); });

    await loadTaskData($("taskSelect").value);
    doReset(); requestAnimationFrame(loop);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(window.MBE);
