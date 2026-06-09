/*
 * Si/Ge (IV 族) 炉子 · 领域模型 (core.js)
 * 设备：Octoplus 500 · 3" 晶圆 · 8 热蒸发 + 2 电子束源 · 离子泵+冷凝泵 · RHEED · QMS · 晶振膜厚仪
 * 本体源：Si、Ge（电子束）；掺杂源：B（热蒸发）。
 * 加载顺序：platform/engine.js -> core.js -> render.js -> app.js
 * IV 族为共价、单元素束流（无 III-V 阴阳离子配对）；关键工艺概念为 Ge 组分、压应变与临界厚度。
 */
(function (M) {
  "use strict";
  var clamp = M.clamp, lerp = M.lerp;

  var P = {
    nPer: 15,          // Si/SiGe 超晶格周期数
    siThick: 8.0,      // Si 隔垒厚 (nm)
    sigeThick: 6.0,    // SiGe 阱厚 (nm)
    geFrac: 0.30,      // SiGe 阱中 Ge 组分 x (0-1)
    tSub: 550,         // 衬底温度 (℃)，Si/Ge MBE 典型 450-650
    powSi: 80,         // Si 电子束功率 (%)
    powGe: 55,         // Ge 电子束功率 (%)
    tempB: 1150,       // B 掺杂源温 (℃)
    dope: false,       // 是否 B 掺杂
    sim: 220
  };
  M.P = P;

  var ML_NM = 0.272;   // Si(001) 单原子层 ≈ a/4 = 0.136nm？取双层台阶 ~0.272nm 作相位刻度
  M.ML_NM = ML_NM;

  // 源：Si/Ge 电子束(ebeam)，B 热蒸发(thermal)。temp 字段对 ebeam 表示功率%
  M.CELL = {
    Si: { temp: 80, bep: "e-beam", col: "#4E79A7", kind: "ebeam", unit: " %" },
    Ge: { temp: 55, bep: "e-beam", col: "#59A14F", kind: "ebeam", unit: " %" },
    B: { temp: 1150, bep: "1.0E-9", col: "#E15759", kind: "thermal", unit: " ℃" },
    rot: 12.0
  };
  M.ELEMENTS = ["Si", "Ge", "B"];
  M.FLUX = [["Si", "e-beam 80%"], ["Ge", "e-beam 55%"], ["B", "1.0E-9"]];
  M.COL = { Si: "#4E79A7", Ge: "#59A14F", SiGe: "#8CB369", B: "#E15759", BUF: "#9CA3AF", SUB: "#BAB0AC" };

  // 应变模型：SiGe/Si 失配 ≈ 0.0418·x（x=Ge 组分），压应变；超过临界厚度→弛豫
  var STRAIN_CRIT = 30;
  M.STRAIN_CRIT = STRAIN_CRIT;

  M.STAGES = [
    { id: "load", name: "Load" },
    { id: "deoxide", name: "Si Deoxide" },
    { id: "buffer", name: "Si Buffer" },
    { id: "sl", name: "Si/SiGe SL" },
    { id: "cooldown", name: "Cooldown" },
    { id: "report", name: "Report" }
  ];

  M.TASKS = [
    { id: "qw", title: "应变 SiGe 量子阱叠层", target: "Ge 组分稳定、应变受控、保持二维生长", dataUrl: "./data/qw.json" },
    { id: "strain", title: "临界厚度 / 应变弛豫训练", target: "在不超过临界厚度的前提下最大化 Ge 组分", dataUrl: "./data/strain.json" },
    { id: "dope", title: "B 掺杂 SiGe 训练", target: "在掺 B 条件下保持表面与应变质量", dataUrl: "./data/dope.json" }
  ];
  M.activeTaskData = null;

  M.shutters = { Si: false, Ge: false, B: false };
  M.manualShutters = false;

  // ---------- 派生量 ----------
  function siRate() { return 0.10 * Math.exp((P.powSi - 80) * 0.020); }   // nm/s
  function geRate() { return 0.07 * Math.exp((P.powGe - 55) * 0.022); }
  M.siRate = siRate; M.geRate = geRate;

  function updateCellStates() {
    M.CELL.Si.temp = P.powSi; M.CELL.Ge.temp = P.powGe; M.CELL.B.temp = P.tempB;
    M.CELL.Si.bep = "e-beam " + P.powSi.toFixed(0) + "%";
    M.CELL.Ge.bep = "e-beam " + P.powGe.toFixed(0) + "%";
    M.FLUX = [["Si", "e-beam " + P.powSi.toFixed(0) + "%"], ["Ge", "e-beam " + P.powGe.toFixed(0) + "%"], ["B", P.dope ? "1.0E-9" : "OFF"]];
  }
  M.updateCellStates = updateCellStates;
  M.updateRatios = function () { };  // IV 族无 V/III 比，占位以兼容引擎接口

  // 实际生长出的 SiGe 阱 Ge 组分（受 Si/Ge 束流比影响，目标值 geFrac 为指令值）
  function geFraction() {
    var rs = siRate(), rg = geRate();
    return clamp(rg / Math.max(rs + rg, 1e-6), 0, 1);
  }
  M.geFraction = geFraction;

  // SiGe 弛豫态带隙(eV)：Si 1.12，Ge 0.66；线性+bowing（定性）
  function sigeBandgap(x) { return 1.12 - 0.41 * x - 0.008 * x * (1 - x) * 12; }
  M.sigeBandgap = sigeBandgap;

  function balanceNote() { return P.geFrac; } // 兼容占位

  function rangeStatus(v, lo, hi, unit) {
    if (v < lo) return { txt: "偏低", cls: "warn", note: "建议 " + lo + "-" + hi + unit };
    if (v > hi) return { txt: "偏高", cls: "warn", note: "建议 " + lo + "-" + hi + unit };
    return { txt: "合适", cls: "good", note: "推荐 " + lo + "-" + hi + unit };
  }
  M.rangeStatus = rangeStatus;

  function rheedClean(ph) { var f = ph - Math.floor(ph); return 1 - 4 * f * (1 - f); }
  M.rheedClean = rheedClean;

  // ---------- 快门（单元素，无配对化学）----------
  function stepShutters(s) {
    if (!s || !s.mat) return { Si: false, Ge: false, B: false };
    if (s.mat === "Si") return { Si: true, Ge: false, B: !!P.dope };
    if (s.mat === "SiGe") return { Si: true, Ge: true, B: !!P.dope };
    if (s.mat === "Ge") return { Si: false, Ge: true, B: !!P.dope };
    return { Si: false, Ge: false, B: false };
  }
  function setShutters(n) { M.shutters.Si = !!n.Si; M.shutters.Ge = !!n.Ge; M.shutters.B = !!n.B; }
  function shutterChemistry() {
    var sh = M.shutters, n = (sh.Si ? 1 : 0) + (sh.Ge ? 1 : 0);
    var dop = sh.B ? "（B 掺杂）" : "";
    if (sh.Si && sh.Ge) return { txt: "当前生长：SiGe 合金" + dop, mat: "SiGe", risk: "low", cls: "good" };
    if (sh.Si && !sh.Ge) return { txt: "当前生长：Si" + dop, mat: "Si", risk: "low", cls: "good" };
    if (sh.Ge && !sh.Si) return { txt: "当前生长：纯 Ge（高失配，注意弛豫）" + dop, mat: "Ge", risk: "med", cls: "warn" };
    if (sh.B && n === 0) return { txt: "仅 B 开启（无本体源，异常）", mat: null, risk: "high", cls: "bad" };
    return { txt: "快门全关 / 待机", mat: null, risk: "idle", cls: "warn" };
  }
  M.stepShutters = stepShutters; M.setShutters = setShutters; M.shutterChemistry = shutterChemistry;

  function strainStatus(a) {
    a = Math.abs(a);
    if (a < STRAIN_CRIT * 0.5) return { txt: "赝晶生长，应变受控", cls: "good" };
    if (a < STRAIN_CRIT) return { txt: "接近临界厚度", cls: "warn" };
    return { txt: "超临界，应变弛豫/失配位错", cls: "bad" };
  }
  M.strainStatus = strainStatus;

  function surfaceQuality(optSt) {
    var st = optSt || M.st;
    var tSubVal = st ? st.actTSub : P.tSub;
    var qT = Math.exp(-Math.pow((tSubVal - 550) / 80, 2));          // Si/Ge 最优 ~550℃
    var x = geFraction();
    var geRough = clamp((x - 0.4) * 0.5, 0, 0.3);                    // Ge 越高越易粗化/3D
    var strainPenalty = st ? clamp(Math.abs(st.accStrain) / STRAIN_CRIT * 0.22, 0, 0.3) : 0;
    var fastPenalty = clamp((siRate() + geRate() - 0.25) * 0.4, 0, 0.2);
    return clamp(qT - geRough - strainPenalty - fastPenalty, 0, 1);
  }
  M.surfaceQuality = surfaceQuality;

  function rheedState(Q) {
    if (Q >= 0.85) return { txt: "Streaky 2×1", cls: "good", desc: "Si(001) 2×1 重构长条纹，二维层状生长" };
    if (Q >= 0.70) return { txt: "Weak Streaky", cls: "good", desc: "条纹略宽，表面轻微粗糙" };
    if (Q >= 0.50) return { txt: "Streak + Spot", cls: "warn", desc: "条纹断裂出现斑点，进入 2D/3D 过渡" };
    if (Q >= 0.30) return { txt: "Spotty 3D", cls: "bad", desc: "斑点增强，SiGe 应变弛豫致岛状生长" };
    return { txt: "Ring/Poly", cls: "bad", desc: "多晶/非晶散射，工艺失控" };
  }
  M.rheedState = rheedState;
  function growthMode(Q) {
    var r = rheedState(Q);
    if (Q >= 0.85) return { txt: "2D 逐层 (FvdM)", cls: r.cls };
    if (Q >= 0.70) return { txt: "弱条纹 2D", cls: r.cls };
    if (Q >= 0.50) return { txt: "2D/3D 过渡", cls: r.cls };
    return { txt: "3D 岛状/弛豫", cls: r.cls };
  }
  M.growthMode = growthMode;

  function relaxRisk() {
    var q = surfaceQuality(), strain = M.st ? Math.abs(M.st.accStrain) / STRAIN_CRIT : 0;
    var risk = (1 - q) * 0.4 + clamp(strain, 0, 1.4) * 0.45 + clamp((geFraction() - 0.4) * 0.6, 0, 0.25);
    if (risk < 0.25) return { txt: "低", cls: "good", why: "应变与表面均在赝晶生长窗口内" };
    if (risk < 0.5) return { txt: "中低", cls: "good", why: "可接受，注意 Ge 组分与厚度" };
    if (risk < 0.72) return { txt: "中高", cls: "warn", why: "接近临界厚度，弛豫风险升高" };
    return { txt: "高", cls: "bad", why: "超临界，失配位错穿透阱区" };
  }
  M.relaxRisk = relaxRisk;

  function targetMatch() { return clamp(1 - Math.abs(geFraction() - P.geFrac) / 0.3, 0, 1); }
  M.targetMatch = targetMatch;

  function score() {
    var st = M.st, q, ops;
    if (st && st.history && st.history.surfaceQualities.length > 0) {
      var avg = function (a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; };
      q = avg(st.history.surfaceQualities); ops = avg(st.history.shutterRisks);
    } else { q = surfaceQuality(); ops = shutterChemistry().risk === "high" ? 0.45 : shutterChemistry().risk === "med" ? 0.7 : 0.92; }
    var strainSafe = st ? clamp(1 - Math.abs(st.accStrain) / STRAIN_CRIT, 0, 1) : 1;
    var geMatch = targetMatch();
    var recipe = clamp(0.6 + geMatch * 0.3 + (P.nPer >= 10 ? 0.1 : 0), 0, 1);
    var parts = { recipe: recipe * 22, surface: q * 24, strain: strainSafe * 22, gecomp: geMatch * 22, ops: ops * 10 };
    var total = Math.round(parts.recipe + parts.surface + parts.strain + parts.gecomp + parts.ops);
    return { total: total, grade: total >= 88 ? "优秀" : total >= 75 ? "合格" : total >= 60 ? "需优化" : "失败训练", cls: total >= 75 ? "good" : total >= 60 ? "warn" : "bad", parts: parts };
  }
  M.score = score;

  // ---------- 工艺步骤 ----------
  function buildSteps() {
    var s = [], p, rSi = siRate(), rGe = geRate(), rSiGe = rSi + rGe;
    s.push({ stage: "load", label: "Load / 3\" Si 晶圆传入", mat: null, nm: 0, dur: 1, subA: 100, subB: 100 });
    s.push({ stage: "deoxide", label: "Si 衬底脱氧 (升温去 SiO2)", mat: null, nm: 0, dur: 1500, subA: 300, subB: 880 });
    s.push({ stage: "deoxide", label: "降至生长温度", mat: null, nm: 0, dur: 600, subA: 880, subB: P.tSub });
    s.push({ stage: "buffer", label: "Si Buffer 生长 100 nm", mat: "Si", nm: 100, dur: 100 / rSi, subA: P.tSub, subB: P.tSub });
    for (p = 0; p < P.nPer; p++) {
      s.push({ stage: "sl", period: p, label: "SiGe 阱层 (Si+Ge" + (P.dope ? "+B" : "") + ")", mat: "SiGe", nm: P.sigeThick, dur: Math.max(0.1, P.sigeThick / rSiGe), subA: P.tSub, subB: P.tSub });
      s.push({ stage: "sl", period: p, label: "Si 隔垒层", mat: "Si", nm: P.siThick, dur: Math.max(0.1, P.siThick / rSi), subA: P.tSub, subB: P.tSub });
    }
    s.push({ stage: "cooldown", label: "Cooldown 冷却", mat: null, nm: 0, dur: 800, subA: P.tSub, subB: 200 });
    s.push({ stage: "report", label: "Report / 训练复盘", mat: null, nm: 0, dur: 1, subA: 200, subB: 200 });
    var t = 0, cn = 0, cs = 0, mis = 0.0418 * P.geFrac;   // SiGe 失配
    for (var i = 0; i < s.length; i++) {
      s[i].t0 = t; t += s[i].dur;
      // SiGe 压应变（负），Si 隔垒释放部分（趋零）
      s[i].rate = s[i].mat === "SiGe" ? -(mis * 220) : s[i].mat === "Si" ? 0 : 0;
      s[i].cumNm0 = cn; s[i].cumStrain0 = cs;
      cn += s[i].nm; cs += s[i].rate * s[i].nm;
    }
    s.totalReal = t; s.totalNm = cn;
    return s;
  }
  M.buildSteps = buildSteps;

  function freshState() {
    updateCellStates();
    var steps = buildSteps();
    return {
      taskId: "qw", mode: "引导模式", steps: steps, si: 0, sp: 0, realT: 0, totalReal: steps.totalReal,
      playing: false, done: false, speedMul: 1, thickNm: 0, phase: 0, accStrain: 0,
      strainHist: [{ t: 0, s: 0 }], rheedHist: [{ t: 0, v: 0.5 }], iSum: 0, iCount: 0,
      logs: [{ t: 0, msg: "训练任务载入：应变 SiGe 量子阱叠层" }],
      actTSub: steps[0] ? steps[0].subA : 100, oxideOpacity: 1.0,
      deoxideStartLogged: false, deoxideDoneLogged: false,
      history: { surfaceQualities: [], shutterRisks: [] }
    };
  }
  M.freshState = freshState;
  M.curStep = function () { var st = M.st; return st.steps[Math.min(st.si, st.steps.length - 1)]; };
  M.subTemp = function () { var s = M.curStep(); return lerp(s.subA, s.subB, M.st.sp); };
  M.stageOf = function () { return M.curStep().stage; };
  M.curPeriod = function () { var s = M.curStep(); return s.period != null ? s.period + 1 : 0; };
  M.stageIdx = function (id) { var S = M.STAGES; for (var i = 0; i < S.length; i++) if (S[i].id === id) return i; return 0; };
  M.locate = function (rt) { var s = M.st.steps, i = M.st.si; while (i < s.length - 1 && rt >= s[i].t0 + s[i].dur) i++; return i; };
  M.recordRHEED = function (Q) { var st = M.st; st.iSum += rheedClean(st.phase) * Q + (1 - Q) * .15; st.iCount++; };

})(window.MBE);
