/*
 * InAs/GaSb 炉子 · 领域模型 (core.js)
 * 参数 / 训练任务 / 工艺步骤 / 半物理规则模型 / 评分与诊断。
 * 加载顺序：platform/engine.js -> core.js -> render.js -> app.js
 * 通用机制（$/lerp/clamp/C/画布/通用图表/工具）由 engine.js 提供。
 */
(function (M) {
  "use strict";

  var P = {
    nPer: 100,
    inas: 5.21,
    gasb: 1.39,
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

  var TENSILE_NM = 1.0, COMP_NM = 13.0, STRAIN_CRIT = 30;
  M.TENSILE_NM = TENSILE_NM; M.COMP_NM = COMP_NM; M.STRAIN_CRIT = STRAIN_CRIT;
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
    { id: "lwir", title: "复现 12.6 μm LWIR 超晶格", target: "λc 12-13 μm, Q > 80, 应变可控", dataUrl: "./data/lwir.json" },
    { id: "standard", title: "标准 InAs/GaSb 超晶格生长", target: "按所选周期数完成生长并保持二维生长", dataUrl: "./data/standard.json" },
    { id: "rheed", title: "RHEED 诊断训练", target: "通过参数调节保持 streaky 图像", dataUrl: "./data/rheed.json" },
    { id: "strain", title: "应变控制训练", target: "避免失配位错与压应变过量", dataUrl: "./data/strain.json" }
  ];
  M.activeTaskData = null;

  M.st = null; M.atoms = []; M.dims = {};
  M.chamber = null; M.cx = null; M.screenC = null; M.scx = null;
  M.mapC = null; M.mcx = null;
  M.strain = null; M.sx = null; M.timingC = null; M.tcx = null; M.bandC = null; M.bcx = null;
  M.rheedC = null; M.rx = null;
  M.shutters = { In: false, Ga: false, As: false, Sb: false };
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
  function balanceInSb() { return (P.inas * TENSILE_NM / COMP_NM) / 2; }
  function rheedClean(ph) { var f = ph - Math.floor(ph); return 1 - 4 * f * (1 - f); }

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
    if (sh.In && sh.Sb && !sh.Ga && !sh.Al && !sh.As) return { txt: "当前形成：InSb-like 界面", mat: "InSb", risk: "low", cls: "good" };
    if (sh.Sb && n === 1) return { txt: "Sb 浸润 (soaking)", mat: "InSb", risk: "low", cls: "good" };
    if ((sh.Ga || sh.Al) && sh.As) return { txt: "警告：GaAs-like 界面", mat: "GaAs", risk: "high", cls: "bad" };
    if (n === 0) return { txt: "快门全关 (AtRunTime 待机)", mat: null, risk: "idle", cls: "warn" };
    return { txt: "混合束流/界面扰动", mat: null, risk: "med", cls: "warn" };
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
    var insbErr = P.comp ? Math.abs(P.insb - ideal) / Math.max(ideal, 0.02) : 1;
    var gaAs = clamp(P.switchDelay * 0.18 + (sh.mat === "GaAs" ? 0.32 : 0) + (!P.comp ? 0.12 : 0), 0, 0.7);
    var inSbLike = clamp((P.comp ? 0.88 : 0.25) - gaAs * 0.55 - clamp(insbErr - 0.8, 0, 2) * 0.08, 0, 1);
    var abrupt = clamp(0.95 - Math.abs(P.soak - 2.0) * 0.12 - P.switchDelay * 0.12 - gaAs * 0.35, 0, 1);
    return {
      inSbLike: inSbLike,
      gaAsLike: clamp(1 - inSbLike, 0, 1),
      abruptness: abrupt,
      cls: abrupt > 0.75 && gaAs < 0.25 ? "good" : abrupt > 0.55 ? "warn" : "bad",
      text: abrupt > 0.75 ? "界面 abruptness 良好" : abrupt > 0.55 ? "界面有轻微混合" : "界面切换失控",
      gaAsRisk: gaAs
    };
  }

  function strainStatus(a) {
    a = Math.abs(a);
    if (a < STRAIN_CRIT * 0.5) return { txt: "赝晶生长，应变受控", cls: "good" };
    if (a < STRAIN_CRIT) return { txt: "应变累积中", cls: "warn" };
    return { txt: "超临界，弛豫/位错风险高", cls: "bad" };
  }

  function effGap() {
    var nML = P.inas / ML_NM;
    var eg = 0.62 / (1 + 0.30 * nML);
    eg += 0.0015 * (P.gasb / ML_NM - 7);
    eg += interfaceMetrics().abruptness < 0.65 ? 0.006 : 0;
    eg += Math.min(Math.abs(M.st ? M.st.accStrain : 0) / STRAIN_CRIT, 1) * 0.003;
    return Math.max(0.045, eg);
  }
  function cutoff(eg) { return 1.24 / eg; }
  function targetMatch() {
    var lam = cutoff(effGap());
    return clamp(1 - Math.abs(lam - P.targetLam) / 3.0, 0, 1);
  }

  function surfaceQuality(optSt) {
    var st = optSt || M.st;
    var tSubVal = st ? st.actTSub : P.tSub;
    var qT = Math.exp(-Math.pow((tSubVal - 390) / 48, 2));
    var optAs = 6.25;
    var optSb = 6.0;
    var qV_As = Math.exp(-Math.pow((P.asIn - optAs) / 3.0, 2));
    var qV_Sb = Math.exp(-Math.pow((P.sbGa - optSb) / 2.0, 2));
    var qV = qV_As * qV_Sb;
    
    // Growth penalty from high fluxes
    var flux_In = cellTempToFlux("In", P.tempIn);
    var flux_Ga = cellTempToFlux("Ga", P.tempGa);
    var maxFlux = Math.max(flux_In, flux_Ga);
    var growthPenalty = maxFlux > 1.2 ? clamp((maxFlux - 1.2) * 0.28, 0, 0.28) : 0;
    
    var intf = interfaceMetrics(st);
    var insbPenalty = P.insb > 0.08 ? clamp((P.insb - 0.08) * 1.6, 0, 0.26) : (P.insb < 0.01 ? 0.08 : 0);
    var strainPenalty = st ? clamp(Math.abs(st.accStrain) / STRAIN_CRIT * 0.18, 0, 0.24) : 0;
    var shutterPenalty = shutterChemistry().risk === "high" ? 0.25 : shutterChemistry().risk === "med" ? 0.12 : 0;
    return clamp(qT * qV - growthPenalty - insbPenalty - (1 - intf.abruptness) * 0.20 - strainPenalty - shutterPenalty, 0, 1);
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
    var strain = M.st ? Math.abs(M.st.accStrain) / STRAIN_CRIT : 0;
    var risk = (1 - q) * 0.35 + (1 - intf.abruptness) * 0.25 + clamp(strain, 0, 1.4) * 0.22 + clamp((lam - 12.8) / 5, 0, 0.25);
    if (risk < 0.25) return { txt: "低", cls: "good", why: "表面质量、界面与应变均处在训练目标内" };
    if (risk < 0.48) return { txt: "中低", cls: "good", why: "材料质量可接受，需继续关注界面扰动" };
    if (risk < 0.72) return { txt: "中高", cls: "warn", why: "粗糙度、界面混合或应变开始推高暗电流" };
    return { txt: "高", cls: "bad", why: "缺陷/应变风险高，器件暗电流会明显上升" };
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
    var strainSafe = st ? clamp(1 - Math.abs(st.accStrain) / STRAIN_CRIT, 0, 1) : 1;
    var recipe = clamp(0.55 + lamMatch * 0.35 + (P.nPer >= 50 ? 0.10 : 0), 0, 1);
    var parts = {
      recipe: recipe * 20,
      surface: q * 20,
      interface: abrupt * 20,
      strain: strainSafe * 15,
      lambda: lamMatch * 15,
      ops: ops * 10
    };
    var total = Math.round(parts.recipe + parts.surface + parts.interface + parts.strain + parts.lambda + parts.ops);
    return {
      total: total,
      grade: total >= 88 ? "优秀" : total >= 75 ? "合格" : total >= 60 ? "需优化" : "失败训练",
      cls: total >= 75 ? "good" : total >= 60 ? "warn" : "bad",
      parts: parts
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
    s.push({ stage: "buffer", label: "GaSb Buffer 生长 500 nm", mat: "GaSb", nm: 500, dur: 500 / rate_GaSb, subA: 440, subB: 380 });
    s.push({ stage: "calib", label: "Interface Calib / Sb soaking 标定", mat: "InSb", nm: P.comp ? 0.08 : 0, dur: P.comp ? 0.08 / rate_InAs : 25, subA: 380, subB: 380 });
    for (p = 0; p < P.nPer; p++) {
      s.push({ stage: "sl", period: p, label: "InSb-like 界面 1 (MEE)", mat: "InSb", nm: P.comp ? P.insb : 0.0, dur: P.comp ? Math.max(0.01, P.insb / rate_InAs) : 0.01, subA: 380, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "Sb 迁移 / soak", mat: "InSb", nm: P.comp ? 0.18 : 0.0, dur: P.comp ? P.soak : 0.01, subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "InAs 主层", mat: "InAs", nm: P.inas, dur: Math.max(0.1, P.inas / rate_InAs), subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "Sb 迁移 / soak", mat: "InSb", nm: P.comp ? 0.18 : 0.0, dur: P.comp ? P.soak : 0.01, subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "InSb-like 界面 2 (MEE)", mat: "InSb", nm: P.comp ? P.insb : 0.0, dur: P.comp ? Math.max(0.01, P.insb / rate_InAs) : 0.01, subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "GaSb 垒层", mat: "GaSb", nm: P.gasb, dur: Math.max(0.1, P.gasb / rate_GaSb), subA: P.tSub, subB: P.tSub });
    }
    s.push({ stage: "cooldown", label: "Cooldown / Sb overpressure 冷却", mat: null, nm: 0, dur: 900, subA: P.tSub, subB: 220 });
    s.push({ stage: "report", label: "Report / 训练复盘", mat: null, nm: 0, dur: 1, subA: 220, subB: 220 });
    var t = 0, cn = 0, cs = 0;
    for (var i = 0; i < s.length; i++) {
      s[i].t0 = t; t += s[i].dur;
      s[i].rate = s[i].mat === "InAs" ? TENSILE_NM : s[i].mat === "InSb" ? -COMP_NM : s[i].mat === "AlSb" ? -0.1 : 0;
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
      taskId: "lwir", mode: "引导模式", steps: steps, si: 0, sp: 0, realT: 0, totalReal: steps.totalReal,
      playing: false, done: false, speedMul: 1, thickNm: 0, phase: 0, accStrain: 0,
      strainHist: [{ t: 0, s: 0 }], rheedHist: [{ t: 0, v: 0.5 }], iSum: 0, iCount: 0,
      logs: [{ t: 0, msg: "训练任务载入：复现 12.6 μm LWIR 超晶格" }],
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
  function locate(rt) { var s = M.st.steps, i = M.st.si; while (i < s.length - 1 && rt >= s[i].t0 + s[i].dur) i++; return i; }

  function recordRHEED(Q) {
    var st = M.st;
    st.iSum += rheedClean(st.phase) * Q + (1 - Q) * .15;
    st.iCount++;
  }

  M.balanceInSb = balanceInSb; M.bandName = bandName; M.rheedClean = rheedClean;
  M.stepShutters = stepShutters; M.setShutters = setShutters; M.shutterChemistry = shutterChemistry;
  M.rangeStatus = rangeStatus; M.interfaceMetrics = interfaceMetrics; M.strainStatus = strainStatus;
  M.effGap = effGap; M.cutoff = cutoff; M.targetMatch = targetMatch;
  M.surfaceQuality = surfaceQuality; M.rheedState = rheedState; M.growthMode = growthMode;
  M.darkCurrentRisk = darkCurrentRisk; M.score = score; M.buildSteps = buildSteps;
  M.freshState = freshState; M.curStep = curStep; M.subTemp = subTemp;
  M.stageOf = stageOf; M.curPeriod = curPeriod; M.stageIdx = stageIdx; M.locate = locate;
  M.recordRHEED = recordRHEED;

})(window.MBE);
