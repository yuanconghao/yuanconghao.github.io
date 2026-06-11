/*
 * InAs/GaSb 炉子 · 领域模型 (core.js)
 * 参数 / 训练任务 / 工艺步骤 / 半物理规则模型 / 评分与诊断。
 * 加载顺序：platform/engine.js -> core.js -> render.js -> app.js
 * 通用机制（$/lerp/clamp/C/画布/通用图表/工具）由 engine.js 提供。
 */
(function (M) {
  "use strict";

  var P = {
    nPer: 20,
    bufferNm: 150,
    inas: 5.21,
    gasb: 1.39,
    alSb: 0,
    insb: 0.20,
    comp: true,
    tSub: 380,
    tempIn: 780,
    tempGa: 900,
    tempAl: 1050,
    tempAs: 310,
    tempSb: 430,
    alEnabled: false,
    asIn: 6.25,
    sbGa: 6.00,
    soak: 2.0,
    switchDelay: 0.5,
    targetLam: 12.8,
    sim: 220
  };
  M.P = P;

  var ML_NM = 0.303;
  M.ML_NM = ML_NM;
  M.CELL = {
    In: { temp: 780.0, bep: "4.00E-7", col: "#4E79A7" },
    Ga: { temp: 900.0, bep: "2.00E-7", col: "#F28E2B" },
    Al: { temp: 1050.0, bep: "1.00E-8", col: "#8c564b" },
    As: { temp: 310.0, bep: "2.50E-6", col: "#76B7B2" },
    Sb: { temp: 430.0, bep: "1.20E-6", col: "#B07AA1" },
    rot: 15.0
  };
  M.FLUX = [["In", "4.00E-7"], ["Ga", "2.00E-7"], ["Al", "1.00E-8"], ["As", "2.50E-6"], ["Sb", "1.20E-6"]];
  M.COL = { InAs: "#4E79A7", GaSb: "#F28E2B", InSb: "#B07AA1", AlSb: "#8c564b", GaAs: "#E15759", BUF: "#59A14F", SUB: "#BAB0AC" };
  M.C = {
    canvasBg: "#ffffff", subFill: "#e3ead9", subText: "#5f6b78",
    label: "#5f6b78", beam: "#c2cad3", front: "rgba(23,33,43,.28)",
    axis: "#d4dae1", axisText: "#5f6b78", line: "#245f9d",
    good: "#0f7a68", warn: "#b97818", bad: "#c0392b",
    zone: "rgba(192,57,43,.08)", screenBg: "#06180f"
  };

  var TENSILE_NM = 0.62, COMP_NM = -5.92, STRAIN_CRIT = 60;
  M.TENSILE_NM = TENSILE_NM; M.COMP_NM = COMP_NM; M.STRAIN_CRIT = STRAIN_CRIT;
  M.LATTICE_A = {
    GaSb: 6.0959,
    InAs: 6.0583,
    InSb: 6.4794,
    AlSb: 6.1355,
    GaAs: 5.6533
  };
  M.CRITICAL_NM = {
    GaSb: 10000,
    InAs: 120,
    InSb: 0.45,
    AlSb: 80,
    GaAs: 5
  };
  var OPTICAL_POINTS = [
    { label: "Delmas 2019", inas: 2.12, gasb: 1.21, alSb: 0, lambda: 5.15, uncertainty: 0.35 },
    { label: "Chen 2019", inas: 2.43, gasb: 2.42, alSb: 0, lambda: 5.30, uncertainty: 0.35 },
    { label: "Xie 2020", inas: 2.40, gasb: 2.40, alSb: 0, lambda: 5.00, uncertainty: 0.35 },
    { label: "Jiang 2016 M-type", inas: 5.45, gasb: 3.03, alSb: 1.52, lambda: 14.00, uncertainty: 0.80 }
  ];
  M.OPTICAL_POINTS = OPTICAL_POINTS;
  var DEFAULT_VALID_RANGE = {
    nPer: [2, 120],
    bufferNm: [20, 500],
    inas: [1.5, 8.0],
    gasb: [0.5, 4.0],
    alSb: [0, 3.0],
    insb: [0, 0.35],
    tSub: [320, 500],
    asIn: [2.0, 12.0],
    sbGa: [2.0, 12.0]
  };
  M.STAGES = [
    { id: "load", name: "Load" },
    { id: "deoxide", name: "Deoxide" },
    { id: "buffer", name: "Buffer" },
    { id: "calib", name: "Interface Calib" },
    { id: "sl", name: "SL Growth" },
    { id: "cooldown", name: "Cooldown" },
    { id: "report", name: "Report" }
  ];

  M.TASKS = [
    { id: "standard", title: "标准 InAs/GaSb 超晶格生长", target: "20 周期教学基线，保持二维层状生长", dataUrl: "./data/standard.json" },
    { id: "delmas2019", title: "Delmas 2019 · 界面时序优化", target: "复现 7 ML / 4 ML MWIR 高质量材料", dataUrl: "./data/delmas2019.json" },
    { id: "chen2019_icip", title: "Chen 2019 · 两级 ICIP 高速探测器", target: "复现低失配、高速 MWIR 吸收区", dataUrl: "./data/chen2019_icip.json" },
    { id: "xie2020_icip", title: "Xie 2020 · 五级 ICIP 室温高速器件", target: "复现 50 周期 2.4/2.4 nm 薄吸收区", dataUrl: "./data/xie2020_icip.json" },
    { id: "jiang2016_mtype", title: "蒋洞微 2016 · M 型势垒长波器件", target: "复现 18/5/5/5 ML M 型势垒与 14 μm 长波指标", dataUrl: "./data/jiang2016_mtype.json" }
  ];
  M.activeTaskData = null;
  M.activeShutterSequence = null;

  M.st = null; M.atoms = []; M.dims = {};
  M.chamber = null; M.cx = null; M.screenC = null; M.scx = null;
  M.mapC = null; M.mcx = null;
  M.strain = null; M.sx = null; M.timingC = null; M.tcx = null; M.bandC = null; M.bcx = null;
  M.rheedC = null; M.rx = null; M.rheedZoomSec = 0;
  M.shutters = { In: false, Ga: false, Al: false, As: false, Sb: false };
  M.manualShutters = false;

  M.$ = function (id) { return document.getElementById(id); };
  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  M.lerp = lerp; M.clamp = clamp;

  function cellTempToFlux(element, temp) {
    var config = {
      In: { center: 780, sensitivity: 0.035 },
      Ga: { center: 900, sensitivity: 0.030 },
      Al: { center: 1050, sensitivity: 0.025 },
      As: { center: 310, sensitivity: 0.045 },
      Sb: { center: 430, sensitivity: 0.040 }
    };
    var cfg = config[element];
    if (!cfg) return 1.0;
    return Math.exp((temp - cfg.center) * cfg.sensitivity);
  }
  M.cellTempToFlux = cellTempToFlux;

  var BEP_BASE = {
    In: 4.0e-7,
    Ga: 2.0e-7,
    Al: 1.0e-8,
    As: 2.5e-6,
    Sb: 1.2e-6
  };
  function cellTempToBEP(element, temp) {
    return BEP_BASE[element] * cellTempToFlux(element, temp);
  }
  M.cellTempToBEP = cellTempToBEP;

  function fmtBEP(val) {
    if (val <= 0) return "0.00E-0";
    return val.toExponential(2).toUpperCase();
  }
  M.fmtBEP = fmtBEP;

  function updateCellStates() {
    var keys = ["In", "Ga", "Al", "As", "Sb"];
    M.FLUX = [];
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var temp = P["temp" + k];
      var bepVal = cellTempToBEP(k, temp);
      M.CELL[k].temp = temp;
      var bepStr = fmtBEP(bepVal);
      M.CELL[k].bep = bepStr;
      if (k !== "Al" || P.alEnabled) {
        M.FLUX.push([k, bepStr]);
      }
    }
  }
  M.updateCellStates = updateCellStates;

  function updateRatios() {
    var bepIn = cellTempToBEP("In", P.tempIn);
    var bepGa = cellTempToBEP("Ga", P.tempGa);
    var bepAs = cellTempToBEP("As", P.tempAs);
    var bepSb = cellTempToBEP("Sb", P.tempSb);
    P.asIn = bepIn > 0 ? bepAs / bepIn : 0;
    P.sbGa = bepGa > 0 ? bepSb / bepGa : 0;
  }
  M.updateRatios = updateRatios;

  function bandName(lam) {
    if (lam < 3) return "短波红外 SWIR";
    if (lam < 5) return "中波红外 MWIR";
    if (lam < 8) return "MWIR/LWIR 过渡";
    if (lam < 14) return "长波红外 LWIR";
    return "甚长波红外 VLWIR";
  }
  function taskModel() { return (M.activeTaskData && M.activeTaskData.model) || {}; }
  function modelSection(name) { return taskModel()[name] || {}; }
  function taskMeasurements() { return (M.activeTaskData && M.activeTaskData.measurements) || {}; }
  function soakLayerNm() {
    var itf = modelSection("interface");
    return itf.soakNm != null ? itf.soakNm : 0;
  }
  function balanceInSb() { return Math.abs(P.inas * latticeMismatch("InAs") / Math.max(Math.abs(latticeMismatch("InSb")), 0.01)) / 2; }
  function rheedClean(ph) { var f = ph - Math.floor(ph); return 1 - 4 * f * (1 - f); }
  function latticeMismatch(mat) {
    var a = M.LATTICE_A[mat], sub = M.LATTICE_A.GaSb;
    if (!a) return 0;
    return (sub - a) / a * 100;
  }
  function criticalThickness(mat) { return M.CRITICAL_NM[mat] || 10000; }
  function strainRateForMaterial(mat) {
    return latticeMismatch(mat);
  }
  function growthRateForMaterial(mat) {
    if (mat === "InAs") return 0.1034 * cellTempToFlux("In", P.tempIn);
    if (mat === "GaSb") return 0.1148 * cellTempToFlux("Ga", P.tempGa);
    if (mat === "InSb") return 0.1034 * cellTempToFlux("In", P.tempIn);
    if (mat === "AlSb") return 0.0709 * cellTempToFlux("Al", P.tempAl);
    if (mat === "GaAs") return 0.1148 * cellTempToFlux("Ga", P.tempGa);
    return 0;
  }

  function stepShutters(s) {
    if (!s || !s.mat) return { In: false, Ga: false, Al: false, As: false, Sb: false };
    if (s.label.indexOf("soak") >= 0 || s.label.indexOf("迁移") >= 0) return { In: false, Ga: false, Al: false, As: false, Sb: true };
    if (s.mat === "InAs") return { In: true, Ga: false, Al: false, As: true, Sb: false };
    if (s.mat === "GaSb") return { In: false, Ga: true, Al: false, As: false, Sb: true };
    if (s.mat === "InSb") return { In: true, Ga: false, Al: false, As: false, Sb: true };
    if (s.mat === "AlSb") return { In: false, Ga: false, Al: true, As: false, Sb: true };
    return { In: false, Ga: false, Al: false, As: false, Sb: false };
  }
  function setShutters(next) {
    M.shutters.In = !!next.In; M.shutters.Ga = !!next.Ga; M.shutters.Al = !!next.Al; M.shutters.As = !!next.As; M.shutters.Sb = !!next.Sb;
  }
  function shutterChemistry() {
    var sh = M.shutters, n = (sh.In ? 1 : 0) + (sh.Ga ? 1 : 0) + (sh.Al ? 1 : 0) + (sh.As ? 1 : 0) + (sh.Sb ? 1 : 0);
    if (sh.In && sh.As && !sh.Ga && !sh.Al && !sh.Sb) return { txt: "当前生长：InAs", mat: "InAs", risk: "low", cls: "good" };
    if (sh.Ga && sh.Sb && !sh.In && !sh.Al && !sh.As) return { txt: "当前生长：GaSb", mat: "GaSb", risk: "low", cls: "good" };
    if (sh.Al && sh.Sb && !sh.In && !sh.Ga && !sh.As) return { txt: "当前生长：AlSb", mat: "AlSb", risk: "low", cls: "good" };
    if (sh.In && sh.Sb && !sh.Ga && !sh.Al && !sh.As) return { txt: "当前形成：InSb 界面 (MEE)", mat: "InSb", risk: "low", cls: "good" };
    if (sh.Sb && n === 1) return { txt: "Sb 浸润 (soaking)", mat: "InSb", risk: "low", cls: "good" };
    if ((sh.Ga || sh.Al) && sh.As) return { txt: "警告：GaAs-like 界面", mat: "GaAs", risk: "high", cls: "bad" };
    if (n === 0) return { txt: "快门全关 (AtRunTime 待机)", mat: null, risk: "idle", cls: "warn" };
    return { txt: "混合束流/界面扰动", mat: null, risk: "med", cls: "warn" };
  }
  function currentDeposition(step) {
    step = step || (M.st ? M.curStep() : null);
    if (M.manualShutters) {
      var chem = shutterChemistry();
      return {
        mat: chem.mat,
        label: chem.txt,
        risk: chem.risk,
        cls: chem.cls,
        rate: growthRateForMaterial(chem.mat),
        manual: true
      };
    }
    return {
      mat: step && step.mat ? step.mat : null,
      label: step ? step.label : "Thermal",
      risk: step && step.mat ? "low" : "idle",
      cls: step && step.mat ? "good" : "warn",
      rate: step && step.dur > 0 ? step.nm / step.dur : 0,
      manual: false
    };
  }
  function appendLayerSegment(st, mat, nm, step, manual) {
    if (!st || !mat || nm <= 0) return;
    var segs = st.layerSegments || (st.layerSegments = []);
    var last = segs[segs.length - 1];
    var period = step && step.period != null ? step.period : null;
    var stage = step ? step.stage : null;
    if (last && last.mat === mat && last.stage === stage && last.period === period && last.manual === !!manual) {
      last.nm += nm;
    } else {
      segs.push({
        mat: mat,
        nm: nm,
        stage: stage,
        period: period,
        label: step ? step.label : "",
        manual: !!manual
      });
    }
  }
  function integrateGrowth(st, t0, t1) {
    if (!st || t1 <= t0) return;
    var t = t0, guard = 0;
    while (t < t1 - 1e-6 && guard++ < 1000) {
      var i = locate(t);
      var step = st.steps[i];
      if (!step) break;
      var segEnd = Math.min(t1, step.t0 + step.dur);
      var dt = Math.max(0, segEnd - t);
      var dep = M.manualShutters ? currentDeposition(step) : {
        mat: step.mat,
        label: step.label,
        risk: step.mat ? "low" : "idle",
        cls: step.mat ? "good" : "warn",
        rate: step.dur > 0 ? step.nm / step.dur : 0,
        manual: false
      };
      var nm = dep.mat ? dep.rate * dt : 0;
      if (nm > 0) {
        appendLayerSegment(st, dep.mat, nm, step, dep.manual);
        st.actualThickNm += nm;
        st.accStrain += strainRateForMaterial(dep.mat) * nm;
        if (dep.manual && step.mat && dep.mat !== step.mat) st.offRecipeNm += nm;
        if (dep.mat === "GaAs") st.gaAsNm += nm;
      } else if (dep.manual && step.mat && !dep.mat) {
        st.idleGrowthSec += dt;
      }
      t = segEnd;
    }
    st.thickNm = st.actualThickNm;
    st.phase = st.thickNm / ML_NM;
  }

  function rangeStatus(v, lo, hi, unit) {
    if (v < lo) return { txt: "偏低", cls: "warn", note: "建议 " + lo + "-" + hi + unit };
    if (v > hi) return { txt: "偏高", cls: "warn", note: "建议 " + lo + "-" + hi + unit };
    return { txt: "合适", cls: "good", note: "推荐 " + lo + "-" + hi + unit };
  }

  function interfaceMetrics(optSt) {
    var st = optSt || M.st;
    var sh = shutterChemistry();
    var ideal = balanceInSb();
    var itfModel = modelSection("interface");
    var soakOpt = itfModel.soakOpt != null ? itfModel.soakOpt : 2.0;
    var soakPenalty = itfModel.soakPenalty != null ? itfModel.soakPenalty : 0.12;
    var insbErr = P.comp ? Math.abs(P.insb - ideal) / Math.max(ideal, 0.02) : 1;
    var gaAsHist = st && st.thickNm > 0 ? clamp((st.gaAsNm || 0) / Math.max(st.thickNm, 1) * 2.0, 0, 0.45) : 0;
    var gaAs = clamp(P.switchDelay * 0.18 + (sh.mat === "GaAs" ? 0.32 : 0) + gaAsHist + (!P.comp ? 0.12 : 0), 0, 0.7);
    var inSbLike = clamp((P.comp ? 0.88 : 0.25) - gaAs * 0.55 - clamp(insbErr - 0.8, 0, 2) * 0.08, 0, 1);
    var abrupt = clamp(0.95 - Math.abs(P.soak - soakOpt) * soakPenalty - P.switchDelay * 0.12 - gaAs * 0.35, 0, 1);
    return {
      inSbLike: inSbLike,
      gaAsLike: clamp(1 - inSbLike, 0, 1),
      abruptness: abrupt,
      cls: abrupt > 0.75 && gaAs < 0.25 ? "good" : abrupt > 0.55 ? "warn" : "bad",
      text: abrupt > 0.75 ? "界面 abruptness 良好" : abrupt > 0.55 ? "界面有轻微混合" : "界面切换失控",
      gaAsRisk: gaAs
    };
  }

  function recipeStrainSegments() {
    var segs = [];
    if (P.bufferNm > 0) segs.push({ mat: "GaSb", nm: P.bufferNm, stage: "buffer" });
    for (var p = 0; p < Math.max(1, P.nPer); p++) {
      if (P.comp && P.insb > 0) segs.push({ mat: "InSb", nm: P.insb, stage: "sl" });
      if (P.inas > 0) segs.push({ mat: "InAs", nm: P.inas, stage: "sl" });
      if (P.comp && P.insb > 0) segs.push({ mat: "InSb", nm: P.insb, stage: "sl" });
      if (P.gasb > 0) segs.push({ mat: "GaSb", nm: P.gasb, stage: "sl" });
      if (P.alEnabled && P.alSb > 0) segs.push({ mat: "AlSb", nm: P.alSb, stage: "sl" });
    }
    return segs;
  }
  function strainMetrics(optSt) {
    var st = optSt || M.st;
    var segs = st && st.layerSegments && st.layerSegments.length ? st.layerSegments : recipeStrainSegments();
    var total = 0, slTotal = 0, signed = 0, absIntegral = 0, localCritical = 0, worst = null;
    for (var i = 0; i < segs.length; i++) {
      var mat = segs[i].mat, nm = Math.max(0, segs[i].nm || 0);
      if (!mat || nm <= 0) continue;
      var mis = latticeMismatch(mat), hc = criticalThickness(mat);
      total += nm;
      if (segs[i].stage !== "buffer") slTotal += nm;
      signed += mis * nm;
      absIntegral += Math.abs(mis) * nm;
      var ratio = hc > 0 ? nm / hc : 0;
      if (ratio > localCritical) { localCritical = ratio; worst = { mat: mat, nm: nm, hc: hc, mismatch: mis }; }
    }
    var avgMismatch = total > 0 ? signed / total : 0;
    var relaxationIndex = Math.max(Math.abs(signed) / STRAIN_CRIT, localCritical);
    var cls = relaxationIndex < 0.5 ? "good" : relaxationIndex < 1 ? "warn" : "bad";
    var txt = cls === "good" ? "赝晶生长，应变受控" : cls === "warn" ? "接近临界，需关注弛豫" : "超临界，弛豫/位错风险高";
    return {
      signed: signed,
      totalNm: total,
      slNm: slTotal,
      avgMismatch: avgMismatch,
      absIntegral: absIntegral,
      localCritical: localCritical,
      relaxationIndex: relaxationIndex,
      worst: worst,
      cls: cls,
      txt: txt
    };
  }
  function strainStatus() {
    var m = strainMetrics();
    return { txt: m.txt, cls: m.cls };
  }

  function opticalCalibrationPoints() {
    var optical = modelSection("optical"), pts = optical.points && optical.points.length ? optical.points.slice() : OPTICAL_POINTS.slice();
    var meas = taskMeasurements(), lam = null;
    if (meas.cutoffUm && meas.cutoffUm.value != null) lam = numericValue(meas.cutoffUm.value);
    else if (meas.plPeakUm && meas.plPeakUm.value != null) lam = numericValue(meas.plPeakUm.value);
    else if (optical.cutoffUm != null) lam = optical.cutoffUm;
    if (lam != null) {
      pts.push({
        label: (M.activeTaskData && M.activeTaskData.title) || "当前任务锚点",
        inas: M.activeTaskData && M.activeTaskData.recipe && M.activeTaskData.recipe.inas != null ? M.activeTaskData.recipe.inas : P.inas,
        gasb: M.activeTaskData && M.activeTaskData.recipe && M.activeTaskData.recipe.gasb != null ? M.activeTaskData.recipe.gasb : P.gasb,
        alSb: M.activeTaskData && M.activeTaskData.recipe && M.activeTaskData.recipe.alSb != null ? M.activeTaskData.recipe.alSb : 0,
        lambda: lam,
        uncertainty: optical.uncertaintyUm != null ? optical.uncertaintyUm : 0.45
      });
    }
    return pts;
  }
  function opticalEstimate() {
    var pts = opticalCalibrationPoints(), al = P.alEnabled ? P.alSb : 0;
    var items = pts.map(function (pt) {
      var d = Math.sqrt(Math.pow((P.inas - pt.inas) / 1.6, 2) + Math.pow((P.gasb - pt.gasb) / 1.1, 2) + Math.pow((al - (pt.alSb || 0)) / 1.0, 2));
      return { point: pt, d: d, w: 1 / (d * d + 0.06) };
    }).sort(function (a, b) { return a.d - b.d; }).slice(0, Math.min(4, pts.length));
    var sw = 0, lam = 0, dist = 0;
    for (var i = 0; i < items.length; i++) { sw += items[i].w; lam += items[i].point.lambda * items[i].w; dist += items[i].d * items[i].w; }
    lam = sw > 0 ? lam / sw : P.targetLam;
    dist = sw > 0 ? dist / sw : 2.5;
    var spread = 0, uncert = 0;
    for (var j = 0; j < items.length; j++) {
      spread += Math.pow(items[j].point.lambda - lam, 2) * items[j].w;
      uncert += (items[j].point.uncertainty || 0.5) * items[j].w;
    }
    spread = sw > 0 ? Math.sqrt(spread / sw) : 1.0;
    uncert = sw > 0 ? uncert / sw : 0.8;
    var intf = interfaceMetrics(), strain = strainMetrics();
    lam += intf.abruptness < 0.65 ? 0.15 : 0;
    lam += clamp(strain.relaxationIndex, 0, 1.4) * 0.08;
    var ci = clamp(uncert + dist * 0.45 + spread * 0.35 + (1 - intf.abruptness) * 0.15, 0.25, 3.5);
    var confidence = clamp(1 - dist / 3.0 - spread / 6.0 - Math.max(0, strain.relaxationIndex - 0.8) * 0.15, 0.15, 0.95);
    return {
      lambda: clamp(lam, 2.5, 18),
      ci: ci,
      low: clamp(lam - ci, 2.5, 18),
      high: clamp(lam + ci, 2.5, 18),
      confidence: confidence,
      nearest: items.map(function (it) { return it.point.label; }),
      extrapolation: confidence < 0.45 ? "强外推" : confidence < 0.68 ? "弱外推" : "插值可信"
    };
  }
  function effGap() {
    return 1.24 / opticalEstimate().lambda;
  }
  function cutoff(eg) { return 1.24 / eg; }
  function targetMatch() {
    var opt = opticalEstimate();
    return clamp(1 - Math.abs(opt.lambda - P.targetLam) / Math.max(3.0, opt.ci * 2.2), 0, 1) * (0.82 + opt.confidence * 0.18);
  }
  function mergedValidRange() {
    var out = {}, custom = modelSection("validRange");
    Object.keys(DEFAULT_VALID_RANGE).forEach(function (k) { out[k] = DEFAULT_VALID_RANGE[k].slice(); });
    Object.keys(custom || {}).forEach(function (k) { if (custom[k] && custom[k].length === 2) out[k] = custom[k].slice(); });
    return out;
  }
  function applicability() {
    var ranges = mergedValidRange(), values = {
      nPer: P.nPer,
      bufferNm: P.bufferNm,
      inas: P.inas,
      gasb: P.gasb,
      alSb: P.alEnabled ? P.alSb : 0,
      insb: P.insb,
      tSub: P.tSub,
      asIn: P.asIn,
      sbGa: P.sbGa
    };
    var labels = {
      nPer: "周期数",
      bufferNm: "GaSb buffer",
      inas: "InAs 厚度",
      gasb: "GaSb 厚度",
      alSb: "AlSb 厚度",
      insb: "InSb 界面",
      tSub: "生长温度",
      asIn: "As/In",
      sbGa: "Sb/Ga"
    };
    var issues = [], penalty = 0;
    Object.keys(ranges).forEach(function (k) {
      var r = ranges[k], v = values[k], span = Math.max(1e-6, r[1] - r[0]), sev = 0;
      if (v < r[0]) sev = (r[0] - v) / span;
      if (v > r[1]) sev = (v - r[1]) / span;
      if (sev > 0) {
        penalty += Math.min(0.25, 0.10 + sev * 0.35);
        issues.push(labels[k] + "超出 " + r[0] + "-" + r[1]);
      }
    });
    var opt = opticalEstimate();
    if (opt.confidence < 0.55) {
      penalty += (0.55 - opt.confidence) * 0.35;
      issues.push("截止波长处于" + opt.extrapolation + "区");
    }
    var confidence = clamp(1 - penalty, 0.2, 1);
    return {
      confidence: confidence,
      cls: confidence >= 0.75 ? "good" : confidence >= 0.5 ? "warn" : "bad",
      txt: confidence >= 0.75 ? "模型适用" : confidence >= 0.5 ? "弱外推" : "超出范围",
      issues: issues,
      optical: opt,
      ranges: ranges
    };
  }

  function surfaceQuality(optSt) {
    var st = optSt || M.st;
    var tSubVal = st ? st.actTSub : P.tSub;
    var quality = modelSection("quality");
    var tOpt = quality.tSubOpt != null ? quality.tSubOpt : 390;
    var tSigma = quality.tSubSigma != null ? quality.tSubSigma : 48;
    var optAs = quality.asInOpt != null ? quality.asInOpt : 6.25;
    var asSigma = quality.asInSigma != null ? quality.asInSigma : 3.0;
    var optSb = quality.sbGaOpt != null ? quality.sbGaOpt : 6.0;
    var sbSigma = quality.sbGaSigma != null ? quality.sbGaSigma : 2.0;
    var qT = Math.exp(-Math.pow((tSubVal - tOpt) / tSigma, 2));
    var qV_As = Math.exp(-Math.pow((P.asIn - optAs) / asSigma, 2));
    var qV_Sb = Math.exp(-Math.pow((P.sbGa - optSb) / sbSigma, 2));
    var qV = qV_As * qV_Sb;
    
    // Growth penalty from high fluxes
    var flux_In = cellTempToFlux("In", P.tempIn);
    var flux_Ga = cellTempToFlux("Ga", P.tempGa);
    var maxFlux = Math.max(flux_In, flux_Ga);
    var growthPenalty = maxFlux > 1.2 ? clamp((maxFlux - 1.2) * 0.28, 0, 0.28) : 0;
    
    var intf = interfaceMetrics(st);
    var idealInSb = balanceInSb();
    var insbPenalty = P.comp ? clamp(Math.abs(P.insb - idealInSb) / Math.max(idealInSb, 0.02) * 0.08, 0, 0.18) : 0.08;
    var strainPenalty = st ? clamp(Math.abs(st.accStrain) / STRAIN_CRIT * 0.18, 0, 0.24) : 0;
    var shutterPenalty = shutterChemistry().risk === "high" ? 0.25 : shutterChemistry().risk === "med" ? 0.12 : 0;
    var offRecipePenalty = st && st.thickNm > 0 ? clamp((st.offRecipeNm || 0) / Math.max(st.thickNm, 1) * 0.42, 0, 0.32) : 0;
    var idlePenalty = st && st.realT > 0 ? clamp((st.idleGrowthSec || 0) / Math.max(st.realT, 1) * 0.20, 0, 0.12) : 0;
    return clamp(qT * qV - growthPenalty - insbPenalty - (1 - intf.abruptness) * 0.20 - strainPenalty - shutterPenalty - offRecipePenalty - idlePenalty, 0, 1);
  }

  function rheedState(Q) {
    if (Q >= 0.85) return { txt: "Streaky 2D", cls: "good", desc: "清晰长条纹，二维层状生长稳定" };
    if (Q >= 0.70) return { txt: "Weak Streaky", cls: "good", desc: "条纹略暗变宽，表面有轻微粗糙" };
    if (Q >= 0.50) return { txt: "Mixed Streak/Spot", cls: "warn", desc: "条纹断裂并出现斑点，进入 2D/3D 过渡" };
    if (Q >= 0.30) return { txt: "Spotty 3D", cls: "bad", desc: "斑点增强，三维岛状生长明显" };
    return { txt: "Ring-like", cls: "bad", desc: "散射背景增强，工艺严重失控" };
  }
  function growthMode(Q) {
    var r = rheedState(Q);
    if (Q >= 0.85) return { txt: "2D 逐层 (FvdM)", cls: r.cls };
    if (Q >= 0.70) return { txt: "弱条纹 2D", cls: r.cls };
    if (Q >= 0.50) return { txt: "2D/3D 过渡", cls: r.cls };
    return { txt: "3D 岛状/粗化", cls: r.cls };
  }

  function darkCurrentRisk() {
    var q = surfaceQuality(), intf = interfaceMetrics(), lam = cutoff(effGap());
    var strain = strainMetrics().relaxationIndex;
    var risk = (1 - q) * 0.35 + (1 - intf.abruptness) * 0.25 + clamp(strain, 0, 1.4) * 0.22 + clamp((lam - 12.8) / 5, 0, 0.25);
    if (risk < 0.25) return { txt: "低", cls: "good", why: "表面质量、界面与应变均处在训练目标内" };
    if (risk < 0.48) return { txt: "中低", cls: "good", why: "材料质量可接受，需继续关注界面扰动" };
    if (risk < 0.72) return { txt: "中高", cls: "warn", why: "粗糙度、界面混合或应变开始推高暗电流" };
    return { txt: "高", cls: "bad", why: "缺陷/应变风险高，器件暗电流会明显上升" };
  }

  function characterizationMetrics() {
    var st = M.st, q = surfaceQuality(), intf = interfaceMetrics(), lam = cutoff(effGap());
    var strain = strainMetrics(st), strainRatio = strain.relaxationIndex;
    var gaAs = intf.gaAsLike;
    var fwhm = 18 + (1 - q) * 55 + (1 - intf.abruptness) * 42 + clamp(strainRatio - 0.45, 0, 1.2) * 35;
    var rmsA = 1.5 + (1 - q) * 8.5 + gaAs * 5.0 + clamp(strainRatio - 0.6, 0, 1.2) * 4.0;
    var hUniform = clamp(intf.abruptness * 0.62 + q * 0.26 + (1 - clamp(strainRatio, 0, 1)) * 0.12, 0, 1);
    var plFwhm = 12 + (1 - hUniform) * 38 + Math.abs(lam - P.targetLam) * 2.0;
    var dark = darkCurrentRisk();
    var r0a = Math.pow(10, 2.5 + q * 1.2 + intf.abruptness * 0.8 - clamp(strainRatio, 0, 1.4) * 0.9 - (lam > 14 ? 0.45 : 0));
    var qe = clamp(0.20 + q * 0.28 + intf.abruptness * 0.18 - gaAs * 0.12 - clamp(strainRatio - 0.7, 0, 1) * 0.12, 0.05, 0.72);
    var dstar = Math.pow(10, 9.6 + q * 0.55 + intf.abruptness * 0.35 - clamp(strainRatio, 0, 1.3) * 0.45 - (dark.cls === "bad" ? 0.5 : 0));
    return {
      xrdFwhm: fwhm,
      xrdPeriod: P.inas + P.gasb + (P.alEnabled ? P.alSb : 0) + (P.comp ? P.insb * 2 + soakLayerNm() * 2 : 0),
      afmRmsA: rmsA,
      hUniform: hUniform,
      plPeak: lam,
      plFwhm: plFwhm,
      r0a: r0a,
      qe: qe,
      dstar: dstar,
      cls: fwhm < 25 && rmsA < 2.5 && hUniform > 0.78 && dark.cls !== "bad" ? "good" : fwhm < 60 && rmsA < 6 && hUniform > 0.55 ? "warn" : "bad"
    };
  }
  function valueOfMeasurement(keys) {
    var meas = taskMeasurements();
    for (var i = 0; i < keys.length; i++) {
      var raw = meas[keys[i]];
      if (raw == null) continue;
      if (typeof raw === "number") return { value: raw, calibrate: true };
      if (raw.value != null) return raw;
    }
    return null;
  }
  function numericValue(v) {
    if (v == null) return null;
    if (typeof v === "number" && isFinite(v)) return v;
    if (typeof v === "string") {
      var m = v.replace(/×10\^/g, "e").replace(/\s/g, "").match(/[-+]?\d*\.?\d+(?:e[-+]?\d+)?/i);
      return m ? parseFloat(m[0]) : null;
    }
    return null;
  }
  function truthComparison() {
    var task = M.activeTaskData || {}, expected = task.expected || {}, ch = characterizationMetrics(), rows = [];
    function add(label, pred, keys, unit, tol, logScale, fallback, calibrateDefault) {
      var m = valueOfMeasurement(keys), truth = m ? numericValue(m.value) : numericValue(fallback);
      if (truth == null) return;
      if (keys.indexOf("afmRmsNm") >= 0 && !m && expected.afmRmsNm != null) truth = truth * 10;
      var calibrate = m && m.calibrate === false ? false : calibrateDefault !== false;
      var err = logScale ? Math.log10(Math.max(pred, 1e-30) / Math.max(truth, 1e-30)) : pred - truth;
      var fit = clamp(1 - Math.abs(err) / Math.max(tol, 1e-9), 0, 1);
      rows.push({
        label: label,
        pred: pred,
        truth: truth,
        unit: unit,
        error: err,
        tolerance: tol,
        logScale: !!logScale,
        fit: fit,
        calibrate: calibrate,
        context: m && m.context ? m.context : "",
        source: m && m.source ? m.source : ""
      });
    }
    add("DXRD FWHM", ch.xrdFwhm, ["xrdFwhmArcsec"], "arcsec", 30, false, expected.xrdFwhmArcsec);
    add("DXRD 周期", ch.xrdPeriod, ["periodNm"], "nm", 0.35, false, expected.periodNm);
    add("AFM RMS", ch.afmRmsA, ["afmRmsA", "afmRmsNm"], "Å", 1.2, false, expected.afmRmsA != null ? expected.afmRmsA : expected.afmRmsNm);
    add("PL 峰/截止", ch.plPeak, ["plPeakUm", "cutoffUm"], "μm", 0.7, false, null);
    add("R0A", ch.r0a, ["r0aOhmCm2"], "Ω·cm²", 1.4, true, expected.r0aOhmCm2, false);
    add("D*", ch.dstar, ["detectivity"], "cm·Hz^1/2/W", 1.2, true, null, false);
    var used = rows.filter(function (r) { return r.calibrate; });
    var score = used.length ? used.reduce(function (sum, r) { return sum + r.fit; }, 0) / used.length : null;
    return { rows: rows, used: used.length, score: score };
  }

  function score() {
    var st = M.st;
    var q, abrupt, inSbLike, gaAsLike, ops;
    if (st && st.history && st.history.surfaceQualities.length > 0) {
      var avg = function (arr) { return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length; };
      q = avg(st.history.surfaceQualities);
      abrupt = avg(st.history.abruptnessValues);
      inSbLike = avg(st.history.inSbLikeValues);
      gaAsLike = avg(st.history.gaAsLikeValues);
      ops = avg(st.history.shutterRisks);
    } else {
      var intf = interfaceMetrics();
      q = surfaceQuality();
      abrupt = intf.abruptness;
      inSbLike = intf.inSbLike;
      gaAsLike = intf.gaAsLike;
      ops = shutterChemistry().risk === "high" ? 0.45 : shutterChemistry().risk === "med" ? 0.68 : 0.92;
    }
    var lamMatch = targetMatch();
    var strainIdx = strainMetrics(st).relaxationIndex;
    var strainSafe = strainIdx < 0.5 ? 1 : clamp(1 - (strainIdx - 0.5) / 0.8, 0, 1);
    var recipe = clamp(0.55 + lamMatch * 0.35 + (P.nPer >= 50 ? 0.10 : 0), 0, 1);
    var truth = truthComparison(), hasTruth = truth.used > 0;
    var maxParts = hasTruth ?
      { recipe: 18, surface: 18, interface: 18, strain: 14, lambda: 14, ops: 8, evidence: 10 } :
      { recipe: 20, surface: 20, interface: 20, strain: 15, lambda: 15, ops: 10, evidence: 0 };
    var parts = {
      recipe: recipe * maxParts.recipe,
      surface: q * maxParts.surface,
      interface: abrupt * maxParts.interface,
      strain: strainSafe * maxParts.strain,
      lambda: lamMatch * maxParts.lambda,
      ops: ops * maxParts.ops,
      evidence: hasTruth ? truth.score * maxParts.evidence : null
    };
    var app = applicability();
    var total = Math.round((parts.recipe + parts.surface + parts.interface + parts.strain + parts.lambda + parts.ops + (parts.evidence || 0)) * (0.92 + app.confidence * 0.08));
    return {
      total: total,
      grade: total >= 88 ? "优秀" : total >= 75 ? "合格" : total >= 60 ? "需优化" : "失败训练",
      cls: total >= 75 ? "good" : total >= 60 ? "warn" : "bad",
      parts: parts,
      maxParts: maxParts,
      truthFit: truth,
      applicability: app
    };
  }

  function buildSteps() {
    var s = [], p;
    var In_RATE = 0.1034, Ga_RATE = 0.1148, Al_RATE = 0.0709;
    var flux_In = cellTempToFlux("In", P.tempIn);
    var flux_Ga = cellTempToFlux("Ga", P.tempGa);
    var flux_Al = cellTempToFlux("Al", P.tempAl);
    
    var rate_InAs = In_RATE * flux_In;
    var rate_GaSb = Ga_RATE * flux_Ga;
    var rate_AlSb = Al_RATE * flux_Al;

    s.push({ stage: "load", label: "Load / AtRunTime 待机", mat: null, nm: 0, dur: 1, subA: 100, subB: 100 });
    s.push({ stage: "deoxide", label: "Deoxide 脱氧 (升温去氧化层)", mat: null, nm: 0, dur: 1400, subA: 300, subB: 530 });
    s.push({ stage: "deoxide", label: "Deoxide 脱氧 (降至生长温度)", mat: null, nm: 0, dur: 700, subA: 530, subB: 440 });
    var bufferNm = Math.max(0, P.bufferNm || 0);
    var soakNm = Math.max(0, soakLayerNm());
    s.push({ stage: "buffer", label: "GaSb Buffer 生长 " + bufferNm.toFixed(0) + " nm", mat: "GaSb", nm: bufferNm, dur: Math.max(0.1, bufferNm / rate_GaSb), subA: 440, subB: 380 });
    s.push({ stage: "calib", label: "Interface Calib / Sb soaking 标定", mat: "InSb", nm: P.comp ? 0.08 : 0, dur: P.comp ? 0.08 / rate_InAs : 25, subA: 380, subB: 380 });
    for (p = 0; p < P.nPer; p++) {
      s.push({ stage: "sl", period: p, label: "GaSb→InAs: InSb 界面 (MEE)", mat: "InSb", nm: P.comp ? P.insb : 0.0, dur: P.comp ? Math.max(0.01, P.insb / rate_InAs) : 0.01, subA: 380, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "Sb 迁移 / soak", mat: "InSb", nm: P.comp ? soakNm : 0.0, dur: P.comp ? P.soak : 0.01, subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "InAs 主层", mat: "InAs", nm: P.inas, dur: Math.max(0.1, P.inas / rate_InAs), subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "Sb 迁移 / soak", mat: "InSb", nm: P.comp ? soakNm : 0.0, dur: P.comp ? P.soak : 0.01, subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "InAs→GaSb: InSb 界面 (MEE)", mat: "InSb", nm: P.comp ? P.insb : 0.0, dur: P.comp ? Math.max(0.01, P.insb / rate_InAs) : 0.01, subA: P.tSub, subB: P.tSub });
      if (P.alEnabled && P.alSb > 0) {
        var gasbHalf = Math.max(0, P.gasb / 2);
        s.push({ stage: "sl", period: p, label: "GaSb 半垒层", mat: "GaSb", nm: gasbHalf, dur: Math.max(0.1, gasbHalf / rate_GaSb), subA: P.tSub, subB: P.tSub });
        s.push({ stage: "sl", period: p, label: "AlSb 势垒层 (M-type)", mat: "AlSb", nm: P.alSb, dur: Math.max(0.1, P.alSb / rate_AlSb), subA: P.tSub, subB: P.tSub });
        s.push({ stage: "sl", period: p, label: "GaSb 半垒层", mat: "GaSb", nm: gasbHalf, dur: Math.max(0.1, gasbHalf / rate_GaSb), subA: P.tSub, subB: P.tSub });
      } else {
        s.push({ stage: "sl", period: p, label: "GaSb 垒层", mat: "GaSb", nm: P.gasb, dur: Math.max(0.1, P.gasb / rate_GaSb), subA: P.tSub, subB: P.tSub });
      }
    }
    s.push({ stage: "cooldown", label: "Cooldown / Sb overpressure 冷却", mat: null, nm: 0, dur: 900, subA: P.tSub, subB: 220 });
    s.push({ stage: "report", label: "Report / 训练复盘", mat: null, nm: 0, dur: 1, subA: 220, subB: 220 });
    var t = 0, cn = 0, cs = 0;
    for (var i = 0; i < s.length; i++) {
      s[i].t0 = t; t += s[i].dur;
      s[i].rate = strainRateForMaterial(s[i].mat);
      s[i].cumNm0 = cn; s[i].cumStrain0 = cs;
      cn += s[i].nm; cs += s[i].rate * s[i].nm;
    }
    s.totalReal = t; s.totalNm = cn;
    return s;
  }

  function freshState() {
    updateRatios();
    updateCellStates();
    var steps = buildSteps();
    return {
      taskId: "standard", mode: "引导模式", steps: steps, si: 0, sp: 0, realT: 0, totalReal: steps.totalReal,
      playing: false, done: false, speedMul: 1, thickNm: 0, phase: 0, accStrain: 0,
      actualThickNm: 0, layerSegments: [], offRecipeNm: 0, gaAsNm: 0, idleGrowthSec: 0,
      strainHist: [{ t: 0, s: 0 }], rheedHist: [{ t: 0, v: 0.5, sensors: [0.18, 0.52, 0.66, 0.78] }], iSum: 0, iCount: 0,
      logs: [{ t: 0, msg: "训练任务载入：标准 InAs/GaSb 超晶格生长" }],
      actTSub: steps[0] ? steps[0].subA : 100,
      oxideOpacity: 1.0,
      deoxideStartLogged: false,
      deoxideDoneLogged: false,
      history: {
        surfaceQualities: [],
        abruptnessValues: [],
        inSbLikeValues: [],
        gaAsLikeValues: [],
        shutterRisks: []
      }
    };
  }
  function curStep() { var st = M.st; return st.steps[Math.min(st.si, st.steps.length - 1)]; }
  function subTemp() { var s = curStep(); return lerp(s.subA, s.subB, M.st.sp); }
  function stageOf() { return curStep().stage; }
  function curPeriod() { var s = curStep(); return s.period != null ? s.period + 1 : 0; }
  function stageIdx(id) { var S = M.STAGES; for (var i = 0; i < S.length; i++) if (S[i].id === id) return i; return 0; }
  function locate(rt) {
    var s = M.st.steps, i = 0;
    while (i < s.length - 1 && rt >= s[i].t0 + s[i].dur) i++;
    return i;
  }

  function recordRHEED(Q) {
    var st = M.st;
    st.iSum += rheedClean(st.phase) * Q + (1 - Q) * .15;
    st.iCount++;
  }

  M.balanceInSb = balanceInSb; M.bandName = bandName; M.rheedClean = rheedClean; M.soakLayerNm = soakLayerNm;
  M.strainRateForMaterial = strainRateForMaterial; M.growthRateForMaterial = growthRateForMaterial;
  M.stepShutters = stepShutters; M.setShutters = setShutters; M.shutterChemistry = shutterChemistry;
  M.currentDeposition = currentDeposition; M.integrateGrowth = integrateGrowth;
  M.rangeStatus = rangeStatus; M.interfaceMetrics = interfaceMetrics; M.strainStatus = strainStatus;
  M.effGap = effGap; M.cutoff = cutoff; M.targetMatch = targetMatch;
  M.surfaceQuality = surfaceQuality; M.rheedState = rheedState; M.growthMode = growthMode;
  M.latticeMismatch = latticeMismatch; M.criticalThickness = criticalThickness; M.strainMetrics = strainMetrics;
  M.opticalEstimate = opticalEstimate; M.applicability = applicability; M.truthComparison = truthComparison;
  M.darkCurrentRisk = darkCurrentRisk; M.characterizationMetrics = characterizationMetrics; M.score = score; M.buildSteps = buildSteps;
  M.freshState = freshState; M.curStep = curStep; M.subTemp = subTemp;
  M.stageOf = stageOf; M.curPeriod = curPeriod; M.stageIdx = stageIdx; M.locate = locate;
  M.recordRHEED = recordRHEED;

})(window.MBE);
