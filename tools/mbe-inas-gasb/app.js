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
  function currentMode() {
    var el = $("modeSelect");
    return el && el.value ? el.value : "引导模式";
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
    M.activeShutterSequence = data.shutterSequence || null;
    var entry = taskEntry(data.id);
    if (entry) { entry.title = data.title || entry.title; entry.target = data.target || entry.target; }
    var r = data.recipe || {}, c = data.conditions || {}, itf = data.interface || {};
    if (r.nPer != null) P.nPer = r.nPer;
    if (r.bufferNm != null) P.bufferNm = r.bufferNm;
    if (r.inas != null) P.inas = r.inas;
    if (r.gasb != null) P.gasb = r.gasb;
    if (r.alSb != null) P.alSb = r.alSb;
    else if (!c.alEnabled) P.alSb = 0;
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
    setInputValue("bufferNm", P.bufferNm, function (v) { return Number(v).toFixed(0) + " nm"; });
    setInputValue("inas", P.inas, function (v) { return Number(v).toFixed(2) + " nm"; });
    setInputValue("gasb", P.gasb, function (v) { return Number(v).toFixed(2) + " nm"; });
    setInputValue("alSb", P.alSb, function (v) { return Number(v).toFixed(2) + " nm"; });
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

    var alRow = $("alRow"), alHint = $("tempAlRowHint"), alSbRow = $("alSbRow");
    if (alRow) alRow.style.display = P.alEnabled ? "flex" : "none";
    if (alHint) alHint.style.display = P.alEnabled ? "flex" : "none";
    if (alSbRow) alSbRow.style.display = P.alEnabled ? "grid" : "none";
  }
  async function loadTaskData(taskId) {
    var entry = taskEntry(taskId);
    M.activeTaskData = null;
    M.activeShutterSequence = null;
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

  function setCardCollapsed(card, body, toggle, collapsed) {
    card.classList.toggle("is-collapsed", collapsed);
    body.hidden = collapsed;
    toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    toggle.setAttribute("aria-label", collapsed ? "展开" : "收起");
  }
  function initCollapsibleControls() {
    var cards = document.querySelectorAll(".area-ctrl > .card");
    for (var i = 0; i < cards.length; i++) {
      var card = cards[i], head = card.querySelector(".sech");
      if (!head || head.parentNode !== card || card.classList.contains("collapsible-card")) continue;
      var title = head.textContent.trim();
      card.classList.add("collapsible-card");

      var body = document.createElement("div");
      body.className = "collapse-body";
      while (head.nextSibling) body.appendChild(head.nextSibling);
      card.appendChild(body);

      head.classList.add("collapse-head");
      head.setAttribute("role", "button");
      head.setAttribute("tabindex", "0");
      head.textContent = "";
      var label = document.createElement("span");
      label.className = "collapse-title";
      label.textContent = title;
      var toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "collapse-toggle";
      head.appendChild(label);
      head.appendChild(toggle);

      var key = "mbe-inas-gasb:collapsed:" + title;
      var saved = null;
      try { saved = window.localStorage ? window.localStorage.getItem(key) : null; } catch (err) { saved = null; }
      var defaultCollapsed = false;
      setCardCollapsed(card, body, toggle, saved == null ? defaultCollapsed : saved === "1");

      (function (c, b, t, storageKey, h) {
        function flip() {
          var next = !c.classList.contains("is-collapsed");
          setCardCollapsed(c, b, t, next);
          try { if (window.localStorage) window.localStorage.setItem(storageKey, next ? "1" : "0"); } catch (err) {}
        }
        h.addEventListener("click", flip);
        h.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            flip();
          }
        });
      })(card, body, toggle, key, head);
    }
  }

  function updateReadout() {
    var st = M.st, s = M.curStep(), STAGES = M.STAGES;
    syncAutoShutters();
    var Q = M.surfaceQuality(), opt = M.opticalEstimate(), eg = M.effGap(), lam = opt.lambda;
    var strainM = M.strainMetrics(), app = M.applicability();
    var chem = M.shutterChemistry(), dep = M.currentDeposition(s);
    var isExamActive = st.mode === "考核模式" && !st.done;
    for (var i = 0; i < STAGES.length; i++) {
      $("stg_" + STAGES[i].id).className = "stg" + (STAGES[i].id === M.stageOf() ? " on" : (M.stageIdx(STAGES[i].id) < M.stageIdx(M.stageOf()) ? " done" : ""));
    }
    $("progBar").style.width = (st.realT / st.totalReal * 100).toFixed(1) + "%";
    setText("taskTitle", currentTask().title);
    setText("taskTarget", currentTask().target);
    setText("taskEvidence", isExamActive ? "考核模式：复现依据、推荐范围和即时提示将在生长完成后进入报告。" : (M.activeTaskData && M.activeTaskData.evidence ? M.activeTaskData.evidence[0] : "任务数据文件未加载时使用内置任务目标。"));
    clsBadge("scopeBadge", app.txt, app.cls);
    setText("scopeText", "置信度 " + (app.confidence * 100).toFixed(0) + "% · " + (app.issues.length ? app.issues.slice(0, 2).join("；") : opt.extrapolation));
    setText("stageTitle", s.label);
    setText("modeTitle", st.mode);
    setText("cycleNow", M.curPeriod() + " / " + P.nPer);
    setText("shutterNow", chem.txt);
    setText("layerNow", M.manualShutters ?
      ((dep.mat || "No deposition") + " · actual rate " + dep.rate.toFixed(3) + " nm/s") :
      ((s.mat || "Thermal") + " · " + (s.nm ? (s.nm * st.sp).toFixed(2) + " / " + s.nm.toFixed(2) + " nm" : "0 nm")));
    setText("stackMapTitle", P.nPer + " 周期全栈压缩总览 · 主画布显示当前局部放大");
    var barrierText = P.alEnabled && P.alSb > 0 ? "GaSb / AlSb / GaSb" : "GaSb";
    setText("stackMapNote", "横向表示放倒后的生长截面：GaSb 衬底 → GaSb buffer → 周期序列；每个周期从左到右为 InSb 界面 / InAs / InSb 界面 / " + barrierText + "，红框为当前周期。");

    setText("eSub", (st ? st.actTSub : M.subTemp()).toFixed(1) + " ℃");
    setText("eRot", M.CELL.rot.toFixed(2) + " rpm");
    setText("eGa", M.CELL.Ga.temp.toFixed(0) + " ℃ (" + M.CELL.Ga.bep + ")");
    setText("eIn", M.CELL.In.temp.toFixed(0) + " ℃ (" + M.CELL.In.bep + ")");
    setText("eAl", P.alEnabled ? M.CELL.Al.temp.toFixed(0) + " ℃ (" + M.CELL.Al.bep + ")" : "OFF");
    
    var actSb = (dep.mat === "GaSb" || dep.mat === "InSb" || dep.mat === "AlSb" || s.label.indexOf("soak") >= 0);
    var actAs = (dep.mat === "InAs" || dep.mat === "GaAs");
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

    var intf = M.interfaceMetrics(), rheed = M.rheedState(Q), risk = M.darkCurrentRisk(), sc = M.score(), ch = M.characterizationMetrics();
    setText("scoreV", isExamActive ? "考核中" : sc.total + " / 100");
    clsBadge("scoreGrade", isExamActive ? "隐藏" : sc.grade, isExamActive ? "warn" : sc.cls);
    setText("qV", (Q * 100).toFixed(0) + " %");
    setText("strainV", strainM.signed.toFixed(1));
    setBadge("strainStat", M.strainStatus(st.accStrain));
    setText("eg", eg.toFixed(3) + " eV");
    setText("lam", lam.toFixed(1) + " μm");
    setText("bandName", M.bandName(lam));
    
    setText("intfInSb", pct(intf.inSbLike));
    setText("intfGaAs", pct(intf.gaAsLike));
    clsBadge("intfAbrupt", intf.text, intf.cls);
    setText("strainDiag", strainM.txt + "；平均失配 " + strainM.avgMismatch.toFixed(3) + "%，松弛指数 " + strainM.relaxationIndex.toFixed(2) + (strainM.worst ? "，最敏感层 " + strainM.worst.mat + " hc≈" + strainM.worst.hc.toFixed(1) + " nm" : ""));
    setText("lambdaMatch", pct(M.targetMatch()));
    setText("opticalDiag", "目标 " + P.targetLam.toFixed(1) + " μm；插值 λc " + opt.low.toFixed(1) + "-" + opt.high.toFixed(1) + " μm，置信 " + (opt.confidence * 100).toFixed(0) + "%");
    clsBadge("darkRisk", risk.txt, risk.cls);
    setText("darkWhy", risk.why);
    setText("charXrd", ch.xrdFwhm.toFixed(0) + " arcsec");
    setText("charAfm", ch.afmRmsA.toFixed(1) + " Å");
    setText("charHrtem", pct(ch.hUniform));
    setText("charPl", ch.plPeak.toFixed(1) + " μm / " + ch.plFwhm.toFixed(0) + " meV");
    setText("charIv", "R0A ~ " + ch.r0a.toExponential(1));
    setText("charSpec", "QE " + (ch.qe * 100).toFixed(0) + "% / D* " + ch.dstar.toExponential(1));
    
    setText("partRecipe", isExamActive ? "--" : sc.parts.recipe.toFixed(0));
    setText("partSurface", isExamActive ? "--" : sc.parts.surface.toFixed(0));
    setText("partInterface", isExamActive ? "--" : sc.parts.interface.toFixed(0));
    setText("partStrain", isExamActive ? "--" : sc.parts.strain.toFixed(0));
    setText("partLambda", isExamActive ? "--" : sc.parts.lambda.toFixed(0));
    setText("partOps", isExamActive ? "--" : sc.parts.ops.toFixed(0));
    var max = sc.maxParts || { recipe: 20, surface: 20, interface: 20, strain: 15, lambda: 15, ops: 10, evidence: 0 };
    setText("maxRecipe", "/" + max.recipe);
    setText("maxSurface", "/" + max.surface);
    setText("maxInterface", "/" + max.interface);
    setText("maxStrain", "/" + max.strain);
    setText("maxLambda", "/" + max.lambda);
    setText("maxOps", "/" + max.ops);
    var evidenceRow = $("evidenceScoreRow");
    if (evidenceRow) evidenceRow.style.display = sc.parts.evidence == null ? "none" : "grid";
    setText("partEvidence", isExamActive || sc.parts.evidence == null ? "--" : sc.parts.evidence.toFixed(0));

    setText("insbHint", "InSb 等效厚度约 " + M.balanceInSb().toFixed(2) + " nm/界面");
    $("insbRow").style.opacity = P.comp ? "1" : ".45";

    setText("surfaceDiag", rheed.desc); setBadge("rheedState", rheed);

    updateConditionHints();
    updateShutterButtons();
    renderLogs();

    if (!isExamActive && Q < 0.55) logEvent("RHEED 变差：表面粗糙度增加，建议检查温度/V-III/速率");
    if (!isExamActive && M.shutterChemistry().mat === "GaAs") logEvent("快门组合 Ga + As：正在形成 GaAs-like 界面");
    if (!isExamActive && strainM.relaxationIndex > 1) logEvent("应变超过临界线：位错和暗电流风险显著升高");
  }
  M.updateReadout = updateReadout;

  function showSummary() {
    var st = M.st, opt = M.opticalEstimate(), eg = M.effGap(), lam = opt.lambda, avg = st.iCount ? st.iSum / st.iCount : 0, ss = M.strainStatus(st.accStrain), sc = M.score(), risk = M.darkCurrentRisk(), strainM = M.strainMetrics();
    var task = currentTask(), expected = task.expected || {}, intf = M.interfaceMetrics(), app = M.applicability();
    setText("sStruct", P.inas.toFixed(2) + "/" + P.gasb.toFixed(2) + " nm x " + P.nPer + " periods");
    setText("sScore", sc.total + " / 100");
    setText("sThick", st.thickNm.toFixed(0) + " nm");
    setText("sFlat", (avg * 100).toFixed(0) + " %");
    setText("sStrain", strainM.signed.toFixed(1) + " %·nm (" + ss.txt + ")");
    setText("sMode", M.rheedState(M.surfaceQuality()).txt);
    setText("sEg", eg.toFixed(3) + " eV"); setText("sLam", lam.toFixed(1) + " μm"); setText("sBand", M.bandName(lam));
    setText("sRisk", risk.txt);
    setText("sModelBasis", "本报告使用物理启发规则模型进行预判：截止波长采用文献锚点插值 + 置信区间，应变采用相对 GaSb 的晶格失配与临界厚度风险。当前模型适用性：" + app.txt + "，置信度 " + (app.confidence * 100).toFixed(0) + "%；不等同于真实 DXRD/AFM/PL/I-V 或机器学习预测。");
    renderEvalRules(sc, avg, ss, intf, expected, opt, strainM, app);
    renderTruthComparison(task);
    renderTaskEvidence(task);
    $("summary").classList.add("show");
    logEvent("训练完成：生成评分与工艺复盘报告");
  }
  function hideSummary() { $("summary").classList.remove("show"); }

  function renderEvalRules(sc, avg, ss, intf, expected, opt, strainM, app) {
    var box = $("sEvalRules");
    if (!box) return;
    var lamRange = expected.lambdaRange ? expected.lambdaRange[0] + "-" + expected.lambdaRange[1] + " μm" : "目标 λc " + P.targetLam.toFixed(1) + " μm ± 3.0 μm";
    var surfaceMin = expected.surfaceQualityMin != null ? expected.surfaceQualityMin + "%" : "70/85% 分档";
    var max = sc.maxParts || { recipe: 20, surface: 20, interface: 20, strain: 15, lambda: 15, ops: 10, evidence: 0 };
    var weightText = "Recipe " + max.recipe + " / 表面 " + max.surface + " / 界面 " + max.interface + " / 应变 " + max.strain + " / 波长 " + max.lambda + " / 操作 " + max.ops + (max.evidence ? " / 文献真值 " + max.evidence : "");
    var rows = [
      ["总分权重", weightText + "；本次各项为 " + sc.parts.recipe.toFixed(0) + ", " + sc.parts.surface.toFixed(0) + ", " + sc.parts.interface.toFixed(0) + ", " + sc.parts.strain.toFixed(0) + ", " + sc.parts.lambda.toFixed(0) + ", " + sc.parts.ops.toFixed(0) + (sc.parts.evidence == null ? "" : ", " + sc.parts.evidence.toFixed(0)) + "。"],
      ["表面质量", "由温度窗口、As/In、Sb/Ga、源炉束流、RHEED 平均强度和粗糙化惩罚估算；本次平均 RHEED 强度 " + (avg * 100).toFixed(0) + "%，任务目标下限 " + surfaceMin + "。"],
      ["RHEED 分档", "Q >= 85% 为 Streaky 2D，70-85% 为 Weak Streaky，50-70% 为 Mixed Streak/Spot，低于 50% 进入 3D/粗糙风险。"],
      ["界面控制", "由 MEE 双 InSb 界面补偿、Sb soaking、As/Sb 切换延迟和错误快门组合估算；本次 abruptness " + (intf.abruptness * 100).toFixed(0) + "%，GaAs-like 风险 " + (intf.gaAsLike * 100).toFixed(0) + "%。"],
      ["应变判断", "按 GaSb 衬底晶格常数计算 InAs/InSb/GaAs/AlSb 的失配积分，并结合单层临界厚度估算弛豫风险；本次松弛指数 " + strainM.relaxationIndex.toFixed(2) + "，平均失配 " + strainM.avgMismatch.toFixed(3) + "%，状态：" + ss.txt + "。"],
      ["光学目标", "截止波长由 Delmas/Chen/Xie/Jiang 等文献锚点按结构距离插值，并给出置信区间；本次 λc " + opt.lambda.toFixed(2) + " μm，区间 " + opt.low.toFixed(2) + "-" + opt.high.toFixed(2) + " μm，" + opt.extrapolation + "；目标范围：" + lamRange + "。"],
      ["适用范围", app.issues.length ? "当前存在外推项：" + app.issues.join("；") + "。报告分数已按模型置信度轻微折减。" : "当前 recipe、温度、V/III 与界面参数处在模型适用范围内。"],
      ["虚拟表征", "右侧 DXRD/AFM/HRTEM/PL/I-V/光谱黑体为规则模型预估值，对应论文表征链路；真值需要实验测试回填校准。"],
      ["暗电流风险", "由表面质量、界面混合、应变超临界和长波截止偏差综合估算；该项为趋势性风险，不是 A/cm² 数值预测。"]
    ];
    if (sc.truthFit && sc.truthFit.used > 0) {
      rows.push(["文献真值校准", "本任务提供 " + sc.truthFit.used + " 项可校准实验真值；最终总分将 10% 权重分配给预测/真值贴合度，本次贴合度 " + (sc.truthFit.score * 100).toFixed(0) + "%。"]);
    }
    box.innerHTML = rows.map(function (r) {
      return "<div class=\"rule-item\"><b>" + r[0] + "</b><span>" + r[1] + "</span></div>";
    }).join("");
  }
  function fmtTruthValue(v, unit, logScale) {
    if (v == null || !isFinite(v)) return "--";
    if (logScale || Math.abs(v) >= 10000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)) return v.toExponential(2) + " " + unit;
    var digits = unit === "nm" || unit === "μm" ? 2 : unit === "Å" ? 1 : 0;
    return v.toFixed(digits) + " " + unit;
  }
  function renderTruthComparison() {
    var box = $("sTruthTable");
    if (!box) return;
    var cmp = M.truthComparison();
    if (!cmp.rows.length) {
      box.innerHTML = "<div class=\"truth-empty\">当前任务未提供可计算实验真值；报告只输出规则模型趋势判断。</div>";
      return;
    }
    var html = "<div class=\"truth-row truth-head\"><span>指标</span><span>预测</span><span>文献/实验真值</span><span>误差</span><span>置信</span></div>";
    html += cmp.rows.map(function (r) {
      var err = r.logScale ? ((r.error >= 0 ? "+" : "") + r.error.toFixed(2) + " decade") : ((r.error >= 0 ? "+" : "") + r.error.toFixed(r.unit === "nm" || r.unit === "μm" ? 2 : 1) + " " + r.unit);
      var cls = r.fit >= 0.75 ? "good" : r.fit >= 0.45 ? "warn" : "bad";
      var used = r.calibrate ? "校准" : "仅对照";
      var context = r.context ? "<em>" + r.context + "</em>" : "";
      return "<div class=\"truth-row\"><span><b>" + r.label + "</b>" + context + "</span><span>" + fmtTruthValue(r.pred, r.unit, r.logScale) + "</span><span>" + fmtTruthValue(r.truth, r.unit, r.logScale) + "</span><span>" + err + "</span><span><i class=\"truth-fit " + cls + "\">" + used + " " + (r.fit * 100).toFixed(0) + "%</i></span></div>";
    }).join("");
    box.innerHTML = html;
  }
  function renderTaskEvidence(task) {
    var list = $("sTaskEvidence");
    if (!list) return;
    var items = task && task.evidence && task.evidence.length ? task.evidence.slice() : [
      "该任务未提供实验标定 evidence；当前报告仅依据用户设置的目标参数和通用物理启发规则进行预判。",
      "建议为自定义任务补充真实 XRD FWHM、AFM RMS、PL 截止波长、RHEED 振荡曲线、I-V 暗电流等数据，以校准评分阈值。"
    ];
    if (task && task.shutterSequence && task.shutterSequence.basis) items.push(task.shutterSequence.basis);
    list.innerHTML = "";
    for (var i = 0; i < items.length; i++) {
      var li = document.createElement("li");
      li.textContent = items[i];
      list.appendChild(li);
    }
  }

  var lastTs = 0;
  function loop(ts) {
    if (!lastTs) lastTs = ts;
    var dt = Math.min(0.05, (ts - lastTs) / 1000); lastTs = ts;
    var st = M.st, Q = M.surfaceQuality();
    var running = st && st.playing && !st.done;
    if (running) {
      var prevRealT = st.realT;
      st.realT += dt * P.sim * st.speedMul;
      if (st.realT >= st.totalReal) {
        st.realT = st.totalReal; st.done = true; st.playing = false; $("play").textContent = "重新生长";
        M.needsRedraw = true;
      }
      var si = M.locate(st.realT), s = st.steps[si], frac = M.clamp((st.realT - s.t0) / s.dur, 0, 1);
      if (si !== st.si) logEvent("进入步骤：" + s.label);
      st.si = si; st.sp = frac;
      if (!M.manualShutters) M.setShutters(M.stepShutters(s));
      M.integrateGrowth(st, prevRealT, st.realT);
      Q = M.surfaceQuality();
      st.strainHist.push({ t: st.realT, s: st.accStrain });
      if (st.strainHist.length > 5000) st.strainHist.shift();

      var depNow = M.currentDeposition(s);
      var growing = !!depNow.mat;
      var oscVal = growing ? M.rheedClean(st.phase) : 0.5;
      var curI = (0.2 + 0.8 * Q) * (0.6 + 0.4 * oscVal) * (1 - st.oxideOpacity * 0.85);
      var rough = 1 - Q, ph = st.phase * 2 * Math.PI;
      var sensors = [
        M.clamp(0.11 + 0.16 * curI + 0.018 * Math.sin(ph * 0.55 + 0.4), 0, 1),
        M.clamp(0.48 + 0.18 * curI + 0.026 * Math.sin(ph + 1.2) + rough * 0.035 * Math.sin(st.realT * 2.1), 0, 1),
        M.clamp(0.60 + 0.19 * curI + 0.030 * Math.sin(ph + 2.4) + rough * 0.045 * Math.sin(st.realT * 2.8 + 0.7), 0, 1),
        M.clamp(0.72 + 0.17 * curI + 0.034 * Math.sin(ph + 3.1) + rough * 0.055 * Math.sin(st.realT * 3.2 + 1.8), 0, 1)
      ];
      st.rheedHist.push({ t: st.realT, v: curI, sensors: sensors });
      if (st.rheedHist.length > 5000) st.rheedHist.shift();

          // Thermal inertia calculation (actual temperature response)
      var targetT = M.lerp(s.subA, s.subB, frac);
      if (s.stage === "sl") targetT = P.tSub;
      var dtSim = Math.max(0, st.realT - prevRealT);
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
      M.drawScreen(Q);
      M.needsRedraw = false;
    }
    requestAnimationFrame(loop);
  }

  function pullStaticInputs() {
    P.nPer = parseFloat($("nPer").value); P.inas = parseFloat($("inas").value);
    P.gasb = parseFloat($("gasb").value); P.alSb = parseFloat($("alSb").value); P.insb = parseFloat($("insb").value);
    P.targetLam = parseFloat($("targetLam").value);
    P.soak = parseFloat($("soak").value); P.switchDelay = parseFloat($("switchDelay").value);
    P.tempIn = parseFloat($("tempIn").value); P.tempGa = parseFloat($("tempGa").value);
    P.tempAl = parseFloat($("tempAl").value); P.tempAs = parseFloat($("tempAs").value);
    P.tempSb = parseFloat($("tempSb").value);
  }
  function setRheedZoom(nextSec) {
    M.rheedZoomSec = Math.max(0, nextSec || 0);
    M.needsRedraw = true;
    M.drawRheed();
  }
  function doReset() {
    var taskId = $("taskSelect").value;
    pullStaticInputs();
    M.manualShutters = false; M.st = M.freshState(); M.atoms = [];
    M.st.taskId = taskId; M.st.mode = currentMode();
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
    initCollapsibleControls();

    $("taskSelect").addEventListener("change", async function () { await loadTaskData(this.value); doReset(); logEvent("训练任务切换：" + currentTask().title); });
    $("modeSelect").addEventListener("change", function () { doReset(); logEvent("训练模式切换：" + currentMode()); });
    $("play").addEventListener("click", function () { if (M.st.done) doReset(); M.st.playing = !M.st.playing; M.st.speedMul = 1; this.textContent = M.st.playing ? "暂停" : "继续"; M.needsRedraw = true; });
    $("reset").addEventListener("click", doReset);
    $("fast").addEventListener("click", function () { if (M.st.done) doReset(); M.st.playing = true; M.st.speedMul = 10; $("play").textContent = "暂停"; M.needsRedraw = true; });

    bindSlider("nPer", "nPer", null, doReset);
    bindSlider("bufferNm", "bufferNm", function (v) { return v.toFixed(0) + " nm"; }, doReset);
    bindSlider("inas", "inas", function (v) { return v.toFixed(2) + " nm"; }, doReset);
    bindSlider("gasb", "gasb", function (v) { return v.toFixed(2) + " nm"; }, doReset);
    bindSlider("alSb", "alSb", function (v) { return v.toFixed(2) + " nm"; }, doReset);
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
    $("rheedFull").addEventListener("click", function () { setRheedZoom(0); });
    $("rheedZoomIn").addEventListener("click", function () { setRheedZoom(M.rheedZoomSec ? Math.max(15, M.rheedZoomSec / 2) : 120); });
    $("rheedZoomOut").addEventListener("click", function () {
      var total = M.st ? M.st.totalReal : 0;
      if (!M.rheedZoomSec || M.rheedZoomSec >= total) setRheedZoom(0);
      else setRheedZoom(M.rheedZoomSec * 2 >= total ? 0 : M.rheedZoomSec * 2);
    });

    await loadTaskData($("taskSelect").value);
    doReset(); requestAnimationFrame(loop);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(window.MBE);
