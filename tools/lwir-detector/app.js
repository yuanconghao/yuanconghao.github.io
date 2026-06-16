(function () {
  "use strict";

  var P = {
    inas: 3.63,
    gasb: 1.22,
    periods: 100,
    interfaceNm: 0.18,
    interfaceQ: 0.84,
    mBarrier: true,
    detT: 77,
    bias: 0.05,
    pixel: 25,
    tint: 5,
    targetT: 310,
    bgT: 293,
    xrd: 64,
    afm: 2.4,
    passivation: 0.78,
    scene: "human"
  };

  var scenes = {
    human: {
      label: "LWIR · 行人",
      target: "行人目标",
      note: "8-14 μm 常温热辐射峰值",
      band: "LWIR 8-14 μm",
      lambdaMin: 8,
      lambdaMax: 14,
      resolution: "128 × 128"
    },
    plume: {
      label: "MWIR · 尾焰",
      target: "发动机尾焰",
      note: "3-5 μm 高温目标强辐射",
      band: "MWIR 3-5 μm",
      lambdaMin: 3,
      lambdaMax: 5,
      resolution: "128 × 128"
    },
    gas: {
      label: "BBIR · 气体",
      target: "气体泄漏场景",
      note: "2-14 μm 宽带气体/宽域侦测",
      band: "BBIR 2-14 μm",
      lambdaMin: 2,
      lambdaMax: 14,
      resolution: "128 × 128"
    }
  };

  var anchors = [
    { name: "Delmas 5.15 μm", inas: 2.12, gasb: 1.21, al: 0, lambda: 5.15 },
    { name: "Chen 5.30 μm", inas: 2.43, gasb: 2.42, al: 0, lambda: 5.30 },
    { name: "Xie 5.00 μm", inas: 2.40, gasb: 2.40, al: 0, lambda: 5.00 },
    { name: "Delmas 10.0 μm", inas: 3.63, gasb: 1.22, al: 0, lambda: 10.0 },
    { name: "Jiang M-type 14 μm", inas: 5.45, gasb: 3.03, al: 1.52, lambda: 14.00 }
  ];

  var els = {};
  var latestModel = null;
  var activeScenario = "";
  var viewMode = "experiment";
  var mechanismFocus = "band";
  var referenceData = null;
  var referenceDataSets = {};
  var activeDataKey = "lwir";
  var activeReferenceId = "";
  var REFERENCE_DATA_FILES = {
    lwir: "./data/lwir-scenarios.json",
    mwir: "./data/mwir-scenarios.json",
    bbir: "./data/bbir-scenarios.json"
  };
  var ids = [
    "inas", "gasb", "periods", "interfaceNm", "interfaceQ", "mBarrier", "detT", "bias", "pixel", "tint", "targetT", "bgT", "xrd", "afm", "passivation",
    "inasV", "gasbV", "periodsV", "interfaceNmV", "interfaceQV", "detTV", "biasV", "pixelV", "tintV", "targetTV", "bgTV", "xrdV", "afmV", "passivationV",
    "lambdaHero", "bandHero", "snrHero", "egOut", "absOut", "qeOut", "iphOut", "jdarkOut", "r0aOut", "dstarOut", "netdOut",
    "absBadge", "darkBadge", "readBadge", "absDiag", "darkDiag", "readDiag",
    "impactLead", "impactBand", "impactTemp", "impactTint", "impactQuality", "dataSourceText",
    "mechanismTitle", "mechanismText", "mechanismMetricA", "mechanismMetricB", "mechanismMetricC",
    "chainRad", "chainAbs", "chainSep", "chainNoise", "chainRoi",
    "presetLwir", "presetMwir", "presetBbir",
    "issueBand", "issueHot", "issueSat", "issueDark", "issueQe", "issueReset",
    "modePrinciple", "modeExperiment", "focusBand", "focusDark", "focusReadout"
  ];

  var VIS = {
    photon: "#b86b1d",
    photonSoft: "rgba(184,107,29,.44)",
    photonGlow: "rgba(184,107,29,.42)",
    electron: "#2166d1",
    electronSoft: "rgba(33,102,209,.34)",
    hole: "#b83f7a",
    holeSoft: "rgba(184,63,122,.36)",
    field: "#0f8a72",
    fieldSoft: "rgba(15,138,114,.64)",
    current: "#2f8fac",
    dark: "#b8322a",
    readout: "#687787"
  };

  function $(id) { return document.getElementById(id); }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function pct(v) { return Math.round(v * 100) + "%"; }
  function fmtExp(v) { return v.toExponential(1).replace("e", "E"); }
  function rr(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function setText(id, value) { if (els[id]) els[id].textContent = value; }
  function setBadge(id, text, cls) { if (els[id]) { els[id].textContent = text; els[id].className = "badge " + cls; } }

  function syncControls() {
    Object.keys(P).forEach(function (k) {
      if (!els[k]) return;
      if (els[k].type === "checkbox") els[k].checked = !!P[k];
      else els[k].value = P[k];
    });
  }

  function setActiveScenario(kind) {
    activeScenario = kind || "";
    [
      ["issueBand", "band"],
      ["issueHot", "hot"],
      ["issueSat", "sat"],
      ["issueDark", "dark"],
      ["issueQe", "qe"],
      ["issueReset", "reset"]
    ].forEach(function (it) {
      if (els[it[0]]) els[it[0]].classList.toggle("active", activeScenario === it[1]);
    });
  }

  function getReferenceDevice(id) {
    if (!id || !referenceData || !referenceData.literatureDevices) return null;
    return referenceData.literatureDevices[id] || null;
  }

  function getActiveReference() {
    return getReferenceDevice(activeReferenceId);
  }

  function dataKeyForScene(scene) {
    if (scene === "plume") return "mwir";
    if (scene === "gas") return "bbir";
    return "lwir";
  }

  function dataKeyForPreset(kind) {
    if (kind === "mwir") return "mwir";
    if (kind === "bbir") return "bbir";
    return "lwir";
  }

  function setReferenceDataKey(key) {
    activeDataKey = key || "lwir";
    referenceData = referenceDataSets[activeDataKey] || null;
    return !!referenceData;
  }

  function measuredQe(ref, lambda) {
    if (!ref || !ref.measured) return null;
    if (typeof ref.measured.estimatedQE === "number") return ref.measured.estimatedQE;
    if (typeof ref.measured.quantumEfficiency === "number") return ref.measured.quantumEfficiency;
    if (typeof ref.measured.responsivity_AW === "number" && lambda > 0) {
      return clamp(ref.measured.responsivity_AW * 1.24 / lambda, 0.02, 0.95);
    }
    return null;
  }

  function measuredDarkCurrent(ref, opt) {
    if (!ref || !ref.measured || typeof ref.measured.darkCurrentDensity_Acm2 !== "number") return null;
    var m = ref.measured;
    var tref = Math.max(m.temperature_K || P.detT || 77, 1);
    var vref = Math.max(Math.abs(m.bias_V || P.bias || 0.05), 0.001);
    var t = Math.max(P.detT || tref, 1);
    var v = Math.max(Math.abs(P.bias || vref), 0.001);
    var k = 8.617e-5;
    var activationScale = Math.exp(-opt.eg / (2 * k) * (1 / t - 1 / tref));
    var biasScale = (1 + Math.pow(v / 0.08, 1.7)) / (1 + Math.pow(vref / 0.08, 1.7));
    return clamp(m.darkCurrentDensity_Acm2 * activationScale * biasScale, 1e-10, 40);
  }

  function applyScenarioData(key, activeKey) {
    if (!referenceData || !referenceData.scenarioPresets || !referenceData.scenarioPresets[key]) return false;
    var preset = referenceData.scenarioPresets[key];
    if (preset.aliasOf && referenceData.scenarioPresets[preset.aliasOf]) {
      preset = referenceData.scenarioPresets[preset.aliasOf];
    }
    Object.assign(P, preset.params || {});
    activeReferenceId = preset.referenceDeviceId || "";
    setActiveScenario(activeKey || "");
    syncControls();
    render();
    return true;
  }

  function setViewMode(mode) {
    viewMode = mode === "principle" ? "principle" : "experiment";
    document.body.classList.toggle("mode-principle", viewMode === "principle");
    document.body.classList.toggle("mode-experiment", viewMode === "experiment");
    if (els.modePrinciple) els.modePrinciple.classList.toggle("active", viewMode === "principle");
    if (els.modeExperiment) els.modeExperiment.classList.toggle("active", viewMode === "experiment");
  }

  function loadReferenceData() {
    if (!window.fetch) return Promise.resolve(false);
    var keys = Object.keys(REFERENCE_DATA_FILES);
    return Promise.all(keys.map(function (key) {
      return fetch(REFERENCE_DATA_FILES[key], { cache: "no-cache" })
        .then(function (res) {
          if (!res.ok) throw new Error("reference data http " + res.status);
          return res.json();
        })
        .then(function (json) {
          referenceDataSets[key] = json;
          return true;
        })
        .catch(function () {
          referenceDataSets[key] = null;
          return false;
        });
    }))
      .then(function (results) {
        setReferenceDataKey("lwir");
        return results.some(Boolean);
      })
      .catch(function () {
        referenceDataSets = {};
        referenceData = null;
        return false;
      });
  }

  function setMechanismFocus(kind) {
    mechanismFocus = kind === "dark" || kind === "readout" ? kind : "band";
    [
      ["focusBand", "band"],
      ["focusDark", "dark"],
      ["focusReadout", "readout"]
    ].forEach(function (it) {
      if (els[it[0]]) els[it[0]].classList.toggle("active", mechanismFocus === it[1]);
    });
    if (latestModel) updateMechanism(latestModel);
  }

  function readInputs() {
    ["inas", "gasb", "periods", "interfaceNm", "interfaceQ", "detT", "bias", "pixel", "tint", "targetT", "bgT", "xrd", "afm", "passivation"].forEach(function (id) {
      if (!els[id]) return;
      P[id] = parseFloat(els[id].value);
    });
    if (els.mBarrier) P.mBarrier = els.mBarrier.checked;
  }

  function updateLabels() {
    setText("inasV", P.inas.toFixed(2) + " nm");
    setText("gasbV", P.gasb.toFixed(2) + " nm");
    setText("periodsV", P.periods.toFixed(0));
    setText("interfaceNmV", P.interfaceNm.toFixed(3) + " nm");
    setText("interfaceQV", pct(P.interfaceQ));
    setText("detTV", P.detT.toFixed(0) + " K");
    setText("biasV", P.bias.toFixed(2) + " V");
    setText("pixelV", P.pixel.toFixed(0) + " μm");
    setText("tintV", P.tint.toFixed(1) + " ms");
    setText("targetTV", P.targetT.toFixed(0) + " K");
    setText("bgTV", P.bgT.toFixed(0) + " K");
    setText("xrdV", P.xrd.toFixed(0) + " arcsec");
    setText("afmV", P.afm.toFixed(1) + " Å");
    setText("passivationV", pct(P.passivation));
  }

  function opticalEstimate() {
    var items = anchors.map(function (a) {
      var al = P.mBarrier ? 1.52 : 0;
      var d = Math.sqrt(Math.pow((P.inas - a.inas) / 1.6, 2) + Math.pow((P.gasb - a.gasb) / 1.1, 2) + Math.pow((al - a.al) / 1.2, 2));
      return { a: a, d: d, w: 1 / (d * d + 0.08) };
    }).sort(function (a, b) { return a.d - b.d; }).slice(0, 4);
    var sw = 0, lam = 0, dist = 0;
    items.forEach(function (it) { sw += it.w; lam += it.a.lambda * it.w; dist += it.d * it.w; });
    lam = sw ? lam / sw : 10;
    dist = sw ? dist / sw : 2;
    lam += (1 - P.interfaceQ) * 0.25;
    lam = clamp(lam, 3.0, 18);
    return {
      lambda: lam,
      eg: 1.24 / lam,
      confidence: clamp(1 - dist / 3.2 - (1 - P.interfaceQ) * 0.18, 0.18, 0.95),
      nearest: items[0].a.name
    };
  }

  function bandName(lambda) {
    if (lambda < 5) return "MWIR";
    if (lambda < 8) return "MWIR/LWIR";
    if (lambda < 14) return "LWIR";
    return "VLWIR";
  }

  function bandCoverage(lambda) {
    var s = sceneMeta();
    var min = s.lambdaMin || 3, max = s.lambdaMax || 14;
    var mid = (min + max) / 2;
    if (lambda <= min) return 0;
    if (lambda >= max) return 1;
    if (lambda <= mid) return 0.75 * (lambda - min) / Math.max(mid - min, 0.5);
    return 0.75 + 0.25 * (lambda - mid) / Math.max(max - mid, 0.5);
  }

  function model() {
    var opt = opticalEstimate();
    var ref = getActiveReference();
    if (ref && ref.measured && typeof ref.measured.cutoff_um === "number") {
      opt.lambda = ref.measured.cutoff_um;
      opt.eg = 1.24 / opt.lambda;
      opt.confidence = 0.92;
    }
    var bandResponse = bandCoverage(opt.lambda);
    var periodNm = P.inas + P.gasb + P.interfaceNm * 2;
    var absNm = periodNm * P.periods;
    var absUm = absNm / 1000;
    var materialQ = clamp(1 - (P.xrd - 18) / 170 - (P.afm - 1.2) / 25, 0.25, 1);
    var overlap = clamp(0.30 + P.interfaceQ * 0.52 + (P.mBarrier ? 0.05 : 0) - Math.max(0, opt.lambda - 14) * 0.025, 0.15, 0.88);
    var absorption = 1 - Math.exp(-1.15 * absUm * clamp(opt.lambda / 10, 0.55, 1.35));
    if (ref && ref.measured && typeof ref.measured.wavefunctionOverlap === "number") {
      overlap = clamp(ref.measured.wavefunctionOverlap, 0.10, 0.92);
    }
    var qe = clamp(absorption * overlap * materialQ * (0.78 + P.passivation * 0.22) * (0.18 + bandResponse * 0.82), 0.03, 0.78);
    var qeAnchor = measuredQe(ref, opt.lambda);
    if (qeAnchor != null) qe = clamp(qe * 0.35 + qeAnchor * 0.65, 0.03, 0.78);
    var photonSignal = Math.max(0, (Math.pow(P.targetT / 300, 4) - Math.pow(P.bgT / 300, 4)));
    var pixelAreaCm2 = Math.pow(P.pixel * 1e-4, 2);
    var iph = 2.2e-7 * qe * photonSignal * pixelAreaCm2 * 1e6;
    var tempTerm = Math.exp(-opt.eg / (2 * 8.617e-5 * Math.max(P.detT, 1)));
    var defect = 0.20 + (1 - materialQ) * 1.4 + (1 - P.interfaceQ) * 1.2 + (1 - P.passivation) * 1.0;
    var biasTerm = 1 + Math.pow(P.bias / 0.08, 1.7);
    var barrierTerm = P.mBarrier ? 0.34 : 1.0;
    var jdark = clamp(8e1 * tempTerm * defect * biasTerm * barrierTerm, 1e-10, 4e-1);
    var jdarkAnchor = measuredDarkCurrent(ref, opt);
    if (jdarkAnchor != null) jdark = jdarkAnchor;
    var idark = jdark * pixelAreaCm2;
    var noise = Math.sqrt(Math.max(idark, 1e-18) * Math.max(P.tint, 0.1) / 5) * 6.5e-6 + 1.5e-12;
    var snr = iph / noise;
    var r0a = clamp(0.025 / Math.max(jdark, 1e-12), 1e-2, 1e9);
    var dstar = clamp(1.0e10 * qe / Math.sqrt(Math.max(jdark, 1e-10) / 1e-6) * (P.pixel / 25), 1e7, 8e12);
    var netd = clamp(90 / Math.sqrt(Math.max(snr, 0.05)) + (1 - qe) * 18 + Math.max(0, P.detT - 120) * 0.45, 5, 500);
    var darkRatio = idark / Math.max(iph, 1e-18);
    var fieldStrength = clamp(0.25 + P.bias / 0.16 * 0.45 + overlap * 0.45, 0.25, 1.15);
    var darkActivity = clamp(Math.log10(Math.max(darkRatio, 0.001)) / 2 + 0.35, 0.12, 1);
    var integrationFill = clamp((snr / 18) * Math.sqrt(Math.max(P.tint, 0.2) / 5), 0.04, 1);
    var saturationRisk = clamp((P.tint - 12) / 8 + Math.max(0, integrationFill - 0.88) * 1.6, 0, 1);
    return {
      opt: opt,
      periodNm: periodNm,
      absNm: absNm,
      materialQ: materialQ,
      overlap: overlap,
      absorption: absorption,
      qe: qe,
      bandResponse: bandResponse,
      photonSignal: photonSignal,
      iph: iph,
      jdark: jdark,
      idark: idark,
      noise: noise,
      snr: snr,
      r0a: r0a,
      dstar: dstar,
      netd: netd,
      darkRatio: darkRatio,
      fieldStrength: fieldStrength,
      darkActivity: darkActivity,
      integrationFill: integrationFill,
      saturationRisk: saturationRisk,
      band: bandName(opt.lambda)
    };
  }

  function updateReadout(m) {
    var s = sceneMeta();
    setText("lambdaHero", m.opt.lambda.toFixed(1) + " μm");
    setText("bandHero", (s.band || m.band) + " · " + s.label);
    setText("snrHero", m.snr > 10 ? "可分辨" : m.snr > 2 ? "临界" : "被噪声淹没");
    setText("egOut", m.opt.eg.toFixed(3) + " eV");
    setText("absOut", m.absNm.toFixed(0) + " nm");
    setText("qeOut", pct(m.qe));
    setText("iphOut", fmtExp(m.iph) + " A");
    setText("jdarkOut", fmtExp(m.jdark) + " A/cm²");
    setText("r0aOut", fmtExp(m.r0a) + " Ω·cm²");
    setText("dstarOut", fmtExp(m.dstar));
    setText("netdOut", m.netd.toFixed(0) + " mK");

    var absCls = m.qe > 0.35 ? "good" : m.qe > 0.18 ? "warn" : "bad";
    setBadge("absBadge", m.qe > 0.35 ? "吸收/收集较好" : m.qe > 0.18 ? "吸收偏弱" : "吸收不足", absCls);
    var coverText = m.bandResponse > 0.85 ? "基本覆盖目标波段" : m.bandResponse > 0.35 ? "只能覆盖目标波段的一部分" : "与目标波段不匹配";
    setText("absDiag", "当前观察对象为" + s.label + "，截止波长约 " + m.opt.lambda.toFixed(1) + " μm，" + coverText + "。红外光子进入 T2SL 吸收区后产生电子/空穴对；QE 表示入射光子最终变成可收集电荷的效率。");

    var darkCls = m.darkRatio < 0.25 ? "good" : m.darkRatio < 2 ? "warn" : "bad";
    setBadge("darkBadge", darkCls === "good" ? "暗电流受控" : darkCls === "warn" ? "暗电流接近信号" : "暗电流主导", darkCls);
    setText("darkDiag", "探测器温度升高或反向偏压过大，会让没有光照也产生的暗电流变强；暗电流越接近光电流，图像越容易被噪声淹没。");

    var readCls = m.snr > 10 ? "good" : m.snr > 2 ? "warn" : "bad";
    setBadge("readBadge", readCls === "good" ? "ROIC 可稳定积分" : readCls === "warn" ? "读出临界" : "信号偏弱", readCls);
    var satText = m.saturationRisk > 0.65 ? "当前积分时间偏长，ROIC 有接近饱和的风险。" : "当前积分时间仍在可解释范围内。";
    setText("readDiag", "目标与背景热辐射差形成光电流；积分时间越长信号越多，但暗电流和饱和风险也会增加。" + satText + " 当前 SNR 趋势约 " + m.snr.toFixed(1) + "。");

    [["chainRad", m.photonSignal > 0.01], ["chainAbs", m.qe > 0.18], ["chainSep", m.overlap > 0.45], ["chainNoise", m.darkRatio < 2], ["chainRoi", m.snr > 2]].forEach(function (it) {
      if (els[it[0]]) els[it[0]].classList.toggle("on", !!it[1]);
    });

    updateImpact(m, s);
    updateMechanism(m);
  }

  function updateImpact(m, s) {
    var lead;
    if (m.bandResponse < 0.35) {
      lead = "主要限制来自波段不匹配：目标发出的红外光没有被当前截止波长充分覆盖。";
    } else if (m.darkRatio > 2) {
      lead = "主要限制来自暗电流：没有光照也产生的电流已经压过了有效光电流。";
    } else if (m.saturationRisk > 0.65) {
      lead = "主要风险来自读出饱和：积分时间太长，Vint 曲线接近 ROIC 上限。";
    } else if (m.qe < 0.18) {
      lead = "主要限制来自吸收/收集效率：界面、周期厚度或表面钝化让可收集电荷变少。";
    } else if (m.snr < 2) {
      lead = "当前信号偏弱：热辐射差、吸收效率和噪声共同限制了输出热图。";
    } else {
      lead = "当前链路比较顺畅：目标辐射、T2SL 吸收、载流子分离和 ROIC 读出能形成可分辨输出。";
    }

    var bandText = m.bandResponse > 0.85
      ? "λc 覆盖 " + (s.band || m.band) + "，入射光和光生载流子动效较强。"
      : m.bandResponse > 0.35
        ? "λc 只覆盖部分目标波段，热图仍有响应，但 QE 和光电流会下降。"
        : "λc 与 " + (s.band || m.band) + " 不匹配，光路会变暗，吸收区粒子明显减少。";

    var tempText = m.darkRatio < 0.25
      ? "探测器温度和偏压较稳，暗电流低于光电流，红色漏电动效较弱。"
      : m.darkRatio < 2
        ? "暗电流已经接近光电流，温度或偏压继续升高会让 SNR 快速变差。"
        : "暗电流主导输出，红色漏电动效增强，热图会被噪声淹没。";

    var tintText = m.saturationRisk > 0.65
      ? "积分时间偏长，Vint 曲线接近饱和线，继续增加会丢失亮暗差异。"
      : m.integrationFill > 0.72
        ? "积分电荷较多，输出更亮，但需要留意暗电流和饱和余量。"
        : "积分电荷仍有余量，ROIC 曲线处在较安全的动态范围。";

    var qualityText = m.materialQ > 0.75 && P.interfaceQ > 0.80
      ? "界面质量和表征指标较好，电子/空穴收集路径清晰。"
      : m.materialQ > 0.50
        ? "材料质量处于临界区，XRD/AFM 或界面质量会拖低 QE。"
        : "材料与界面质量偏差较大，吸收区产生的载流子难以有效收集。";

    setText("impactLead", lead);
    setText("impactBand", bandText);
    setText("impactTemp", tempText);
    setText("impactTint", tintText);
    setText("impactQuality", qualityText);

    var ref = getActiveReference();
    if (ref && ref.source) {
      var sourceText = "数据依据：" + activeDataKey.toUpperCase() + " · " + ref.source.authors + "，" + ref.source.year + "，《" + ref.source.title + "》";
      if (ref.measured && typeof ref.measured.darkCurrentDensity_Acm2 === "number") {
        sourceText += "；Jdark=" + fmtExp(ref.measured.darkCurrentDensity_Acm2) + " A/cm²";
      }
      if (ref.measured && typeof ref.measured.cutoff_um === "number") {
        sourceText += "，λc≈" + ref.measured.cutoff_um + " μm";
      }
      setText("dataSourceText", sourceText);
    } else {
      setText("dataSourceText", "数据依据：当前为教学规则插值；LWIR/MWIR/BBIR 分别读取独立 JSON，避免场景参数互相覆盖。");
    }
  }

  function updateMechanism(m) {
    var s = sceneMeta();
    if (mechanismFocus === "dark") {
      setText("mechanismTitle", "暗电流把真实光信号淹没");
      setText("mechanismText", "理想情况下，ROIC 主要积分目标辐射产生的光电流；但探测器温度、偏压、界面缺陷和表面漏电会带来无光照电流。暗电流越接近或超过光电流，热图中的目标边界就越不稳定。");
      setText("mechanismMetricA", "Jdark " + fmtExp(m.jdark) + " A/cm²");
      setText("mechanismMetricB", "Idark / Iphoto " + m.darkRatio.toFixed(2));
      setText("mechanismMetricC", "探测器温度 " + P.detT.toFixed(0) + " K");
    } else if (mechanismFocus === "readout") {
      setText("mechanismTitle", "ROIC 把电荷积分成像素灰度");
      setText("mechanismText", "CTIA 读出链路不是直接读一个瞬时电流，而是在积分时间内把 Iphoto - Idark 转成电容电压 Vint。积分时间太短会信号弱，太长又可能饱和，亮暗差异会被压扁。");
      setText("mechanismMetricA", "积分时间 " + P.tint.toFixed(1) + " ms");
      setText("mechanismMetricB", "Vint 占比 " + pct(m.integrationFill));
      setText("mechanismMetricC", "饱和风险 " + pct(m.saturationRisk));
    } else {
      setText("mechanismTitle", "Type-II 能带让电子和空穴分居两侧");
      setText("mechanismText", "InAs/GaSb 二类超晶格的关键不是简单叠层，而是错位能带：电子更容易落在 InAs 势阱，空穴更容易落在 GaSb 势阱。这样可以通过周期厚度调节有效带隙和截止波长，但界面质量会直接影响波函数重叠和收集效率。");
      setText("mechanismMetricA", "λc " + m.opt.lambda.toFixed(1) + " μm / " + (s.band || m.band));
      setText("mechanismMetricB", "波段覆盖 " + pct(m.bandResponse));
      setText("mechanismMetricC", "波函数重叠 " + pct(m.overlap));
    }
  }

  function canvasCtx(id) {
    var c = $(id), dpr = window.devicePixelRatio || 1, rect = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(rect.width * dpr));
    c.height = Math.max(1, Math.round(rect.height * dpr));
    var ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: rect.width, h: rect.height };
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, width) {
    var a = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width || 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - Math.cos(a - 0.45) * 10, y2 - Math.sin(a - 0.45) * 10);
    ctx.lineTo(x2 - Math.cos(a + 0.45) * 10, y2 - Math.sin(a + 0.45) * 10);
    ctx.closePath();
    ctx.fill();
  }

  function cubicPoint(p0, p1, p2, p3, t) {
    var u = 1 - t;
    return {
      x: u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
      y: u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y
    };
  }

  function glowDot(ctx, x, y, color, radius, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha == null ? 1 : alpha;
    ctx.shadowColor = color;
    ctx.shadowBlur = radius * 2.6;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function stepBadge(ctx, n, x, y, title, desc, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 13, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "800 12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(n), x, y + 4);
    ctx.fillStyle = "#17212b";
    ctx.font = "800 12px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(title, x + 18, y - 2);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText(desc, x + 18, y + 13);
  }

  function drawBandInset(ctx, x, y, w, h, m, tAnim) {
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,.90)";
    rr(ctx, x, y, w, h, 7);
    ctx.fill();
    ctx.strokeStyle = "rgba(214,224,232,.88)";
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Type-II 能带", x + 8, y + 13);

    var bx = x + 12, bw = w - 24;
    var eY = y + h * 0.38, hY = y + h * 0.66;
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = VIS.electron;
    ctx.beginPath();
    ctx.moveTo(bx, eY - 5);
    ctx.lineTo(bx + bw * 0.34, eY - 5);
    ctx.lineTo(bx + bw * 0.34, eY + 5);
    ctx.lineTo(bx + bw * 0.67, eY + 5);
    ctx.lineTo(bx + bw * 0.67, eY - 4);
    ctx.lineTo(bx + bw, eY - 4);
    ctx.stroke();

    ctx.strokeStyle = VIS.hole;
    ctx.beginPath();
    ctx.moveTo(bx, hY + 4);
    ctx.lineTo(bx + bw * 0.32, hY + 4);
    ctx.lineTo(bx + bw * 0.32, hY - 5);
    ctx.lineTo(bx + bw * 0.66, hY - 5);
    ctx.lineTo(bx + bw * 0.66, hY + 5);
    ctx.lineTo(bx + bw, hY + 5);
    ctx.stroke();

    var p = (tAnim * 0.35) % 1;
    var photonX = x + w * 0.50;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = VIS.photon;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(photonX, y + 18);
    ctx.lineTo(photonX, y + h - 12);
    ctx.stroke();
    ctx.setLineDash([]);
    glowDot(ctx, photonX, lerp(y + 20, y + h - 14, p), VIS.photon, 2.2, 0.75);
    glowDot(ctx, x + w * 0.40, eY + 5, VIS.electron, 2.5, 0.86);
    glowDot(ctx, x + w * 0.62, hY - 5, VIS.hole, 2.5, 0.86);

    ctx.fillStyle = VIS.electron;
    ctx.font = "800 8px sans-serif";
    ctx.fillText("e- / InAs", x + 8, y + h - 18);
    ctx.fillStyle = VIS.hole;
    ctx.textAlign = "right";
    ctx.fillText("h+ / GaSb", x + w - 8, y + h - 18);
    ctx.fillStyle = "#66727f";
    ctx.textAlign = "center";
    ctx.fillText("Eg " + m.opt.eg.toFixed(3) + " eV", x + w * 0.50, y + h - 6);
    ctx.restore();
  }

  function drawIntegrationCurve(ctx, x, y, w, h, m, tAnim) {
    ctx.fillStyle = "#ffffff";
    rr(ctx, x, y, w, h, 7);
    ctx.fill();
    ctx.strokeStyle = VIS.readout;
    ctx.stroke();

    ctx.fillStyle = "#17212b";
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("CTIA 积分电容 · Vint(t)", x + w / 2, y + 13);

    var gx = x + 14, gy = y + 22, gw = w - 28, gh = h - 36;
    ctx.fillStyle = "#edf4f8";
    rr(ctx, gx, gy, gw, gh, 5);
    ctx.fill();
    ctx.strokeStyle = "rgba(104,119,135,.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gx, gy + gh);
    ctx.lineTo(gx + gw, gy + gh);
    ctx.stroke();

    var darkBase = clamp(m.darkActivity * 0.34, 0.04, 0.46);
    ctx.strokeStyle = "rgba(184,50,42,.45)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(gx, gy + gh - gh * darkBase);
    ctx.lineTo(gx + gw, gy + gh - gh * darkBase);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = VIS.current;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i <= 52; i++) {
      var u = i / 52;
      var level = (1 - Math.exp(-u * (2.2 + m.bandResponse * 1.5))) * m.integrationFill;
      var px = gx + gw * u;
      var py = gy + gh - gh * clamp(level, 0, 1);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    if (m.saturationRisk > 0.35) {
      ctx.strokeStyle = "rgba(169,103,22,.62)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gx, gy + gh * 0.14);
      ctx.lineTo(gx + gw, gy + gh * 0.14);
      ctx.stroke();
      ctx.fillStyle = "#a96716";
      ctx.font = "800 8px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText("接近饱和", gx + gw - 4, gy + gh * 0.14 - 3);
    }

    var phase = (tAnim * 0.18) % 1;
    var dotLevel = (1 - Math.exp(-phase * (2.2 + m.bandResponse * 1.5))) * m.integrationFill;
    var dotX = gx + gw * phase, dotY = gy + gh - gh * clamp(dotLevel, 0, 1);
    ctx.strokeStyle = "rgba(47,143,172,.45)";
    ctx.beginPath();
    ctx.moveTo(dotX, gy);
    ctx.lineTo(dotX, gy + gh);
    ctx.stroke();
    glowDot(ctx, dotX, dotY, VIS.current, 3, 0.86);

    ctx.fillStyle = "#66727f";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("dark", gx + 4, gy + gh - gh * darkBase - 3);
    ctx.textAlign = "center";
    ctx.fillText("Iphoto - Idark → Vint", x + w / 2, y + h - 5);
  }

  function sceneMeta() {
    return scenes[P.scene] || scenes.human;
  }

  function blob(v, cx, cy, rx, ry) {
    var dx = (v.x - cx) / rx, dy = (v.y - cy) / ry;
    return Math.exp(-(dx * dx + dy * dy));
  }

  function inBox(v, x, y, w, h) {
    return v.x >= x && v.x <= x + w && v.y >= y && v.y <= y + h ? 1 : 0;
  }

  function objectHeat(sceneKey, x, y) {
    var v = { x: x, y: y };
    if (sceneKey === "plume" || sceneKey === "engine") {
      var nozzle = inBox(v, 0.15, 0.46, 0.24, 0.16) * 0.50;
      var hotCore = blob(v, 0.43, 0.53, 0.16, 0.11) * 1.00;
      var plume1 = blob(v, 0.58, 0.50, 0.22, 0.13) * 0.86;
      var plume2 = blob(v, 0.75, 0.47, 0.24, 0.18) * 0.66;
      var plume3 = blob(v, 0.68, 0.62, 0.18, 0.12) * 0.46;
      return Math.max(nozzle, hotCore, plume1, plume2, plume3);
    }
    if (sceneKey === "gas" || sceneKey === "drone") {
      var tank = inBox(v, 0.16, 0.45, 0.18, 0.30) * 0.34;
      var pipe = inBox(v, 0.30, 0.58, 0.48, 0.055) * 0.26;
      var valve = blob(v, 0.46, 0.58, 0.035, 0.035) * 0.45;
      var leak = blob(v, 0.56, 0.42, 0.15, 0.11) * 0.72;
      var cloud1 = blob(v, 0.66, 0.35, 0.18, 0.13) * 0.62;
      var cloud2 = blob(v, 0.75, 0.50, 0.20, 0.15) * 0.50;
      var cloud3 = blob(v, 0.63, 0.63, 0.15, 0.10) * 0.38;
      return Math.max(tank, pipe, valve, leak, cloud1, cloud2, cloud3);
    }
    var head = blob(v, 0.50, 0.18, 0.090, 0.095) * 1.00;
    var torso = blob(v, 0.50, 0.43, 0.14, 0.22) * 0.96;
    var leftArm = blob(v, 0.34, 0.43, 0.060, 0.20) * 0.68;
    var rightArm = blob(v, 0.66, 0.43, 0.060, 0.20) * 0.68;
    var leftLeg = blob(v, 0.43, 0.75, 0.062, 0.23) * 0.70;
    var rightLeg = blob(v, 0.57, 0.75, 0.062, 0.23) * 0.70;
    return Math.max(head, torso, leftArm, rightArm, leftLeg, rightLeg);
  }

  function heatColor(v) {
    v = clamp(v, 0, 1);
    var r, g, b;
    if (v < 0.34) {
      var a = v / 0.34;
      r = lerp(22, 40, a); g = lerp(42, 76, a); b = lerp(58, 96, a);
    } else if (v < 0.72) {
      var c = (v - 0.34) / 0.38;
      r = lerp(40, 190, c); g = lerp(76, 116, c); b = lerp(96, 44, c);
    } else {
      var d = (v - 0.72) / 0.28;
      r = lerp(190, 252, d); g = lerp(116, 222, d); b = lerp(44, 108, d);
    }
    return [Math.round(r), Math.round(g), Math.round(b)];
  }

  var thermalCanvas = null;

  function drawDetectionBox(ctx, x, y, w, h, label, color, confidence, dashed) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    if (dashed) ctx.setLineDash([5, 4]);
    rr(ctx, x, y, w, h, 6);
    ctx.stroke();
    ctx.setLineDash([]);

    var text = label + " " + Math.round(confidence * 100) + "%";
    ctx.font = "800 10px sans-serif";
    var tw = ctx.measureText(text).width + 14;
    ctx.fillStyle = "rgba(255,255,255,.90)";
    rr(ctx, x, Math.max(2, y - 22), tw, 18, 5);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.fillText(text, x + 7, Math.max(14, y - 9));
    ctx.restore();
  }

  function drawThermalDetection(ctx, x, y, size, m) {
    var sceneKey = P.scene || "human";
    var conf = clamp(0.36 + m.snr / 35, 0.32, 0.96);
    var color = m.snr > 6 ? VIS.field : m.snr > 2 ? VIS.photon : VIS.dark;
    var dashed = m.snr <= 2;
    if (sceneKey === "plume" || sceneKey === "engine") {
      drawDetectionBox(ctx, x + size * 0.26, y + size * 0.33, size * 0.60, size * 0.35, "尾焰高温源", color, conf, dashed);
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,.50)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + size * 0.18, y + size * 0.56);
      ctx.lineTo(x + size * 0.86, y + size * 0.48);
      ctx.stroke();
      ctx.restore();
    } else if (sceneKey === "gas" || sceneKey === "drone") {
      drawDetectionBox(ctx, x + size * 0.46, y + size * 0.22, size * 0.40, size * 0.48, "气体云团", color, conf, true);
      drawDetectionBox(ctx, x + size * 0.13, y + size * 0.43, size * 0.36, size * 0.28, "泄漏源", VIS.photon, clamp(conf - 0.10, 0.25, 0.90), dashed);
    } else {
      drawDetectionBox(ctx, x + size * 0.31, y + size * 0.08, size * 0.38, size * 0.82, "行人目标", color, conf, dashed);
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,.42)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + size * 0.42, y + size * 0.18);
      ctx.lineTo(x + size * 0.58, y + size * 0.18);
      ctx.moveTo(x + size * 0.37, y + size * 0.74);
      ctx.lineTo(x + size * 0.63, y + size * 0.74);
      ctx.stroke();
      ctx.restore();
    }
    if (m.snr <= 2) {
      ctx.save();
      ctx.fillStyle = "rgba(184,50,42,.88)";
      ctx.font = "800 10px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText("低置信 / 噪声干扰", x + size - 8, y + size - 10);
      ctx.restore();
    }
  }

  function drawThermalOutput(ctx, x, y, size, m, heat, tAnim) {
    var res = 128;
    if (!thermalCanvas) thermalCanvas = document.createElement("canvas");
    thermalCanvas.width = res;
    thermalCanvas.height = res;
    var tctx = thermalCanvas.getContext("2d");
    var img = tctx.createImageData(res, res);
    var sceneKey = P.scene || "human";
    var snrVisual = clamp(m.snr / 15, 0, 1);
    var bandGate = m.bandResponse > 0.05 ? 1.0 : clamp(0.12 + m.bandResponse * 17.6, 0.12, 1.0);
    var responseVisual = clamp((0.15 + m.bandResponse * 0.35 + m.qe * 0.45 + snrVisual * 0.55) * bandGate, 0.06, 1.15);
    var readoutVisual = clamp(0.38 + Math.sqrt(m.integrationFill) * 0.72, 0.25, 1.12);
    var darkVisual = clamp(0.08 + m.darkActivity * 0.22 + Math.min(m.darkRatio, 6) * 0.045 + (1 - m.materialQ) * 0.09, 0.08, 0.52);
    var saturationVisual = clamp(m.saturationRisk, 0, 1);
    var sceneSignal = sceneKey === "gas"
      ? clamp(responseVisual * readoutVisual, 0.05, 0.92)
      : clamp((0.50 + heat * 0.50) * responseVisual * readoutVisual * 1.55, 0.08, 1.15);
    var noiseScale = clamp(1 - snrVisual * 0.55, 0.30, 1.0);
    var sceneNoise = ((sceneKey === "gas" ? 0.035 : 0.018) + darkVisual * (sceneKey === "gas" ? 0.55 : 0.34) + (1 - m.bandResponse) * 0.035) * noiseScale;
    var contrastLoss = 1 - saturationVisual * 0.38;
    for (var py = 0; py < res; py++) {
      for (var px = 0; px < res; px++) {
        var nx = px / (res - 1), ny = py / (res - 1);
        var shape = Math.pow(clamp(objectHeat(sceneKey, nx, ny), 0, 1), 0.72);
        var vignette = 0.08 * (1 - Math.min(1, Math.hypot(nx - 0.5, ny - 0.5) / 0.72));
        var fixedNoise = Math.sin(px * 12.9898 + py * 78.233 + tAnim * 2.1) * 43758.5453;
        fixedNoise = fixedNoise - Math.floor(fixedNoise);
        var columnNoise = Math.sin(px * 0.47 + tAnim * 0.65) * darkVisual * 0.045;
        var rowNoise = Math.sin(py * 0.55 + tAnim * 1.4) * (0.012 + darkVisual * 0.075);
        var badPixel = fixedNoise > 0.986 ? darkVisual * 0.42 : 0;
        var val = 0.10 + vignette + shape * (0.62 + heat * 0.30) * sceneSignal * contrastLoss + (fixedNoise - 0.5) * sceneNoise + rowNoise + columnNoise + badPixel;
        if (sceneKey === "plume") val += blob({ x: nx, y: ny }, 0.46, 0.52, 0.14, 0.09) * 0.30 * sceneSignal * contrastLoss;
        if (sceneKey === "gas") {
          var plumeBand = Math.max(blob({ x: nx, y: ny }, 0.64, 0.39, 0.24, 0.17), blob({ x: nx, y: ny }, 0.74, 0.52, 0.22, 0.15));
          var gasStructure = shape * 0.46 + plumeBand * 0.34;
          var bandLoss = clamp(1 - m.bandResponse, 0, 1);
          val = 0.11 + vignette * 0.62 + gasStructure * sceneSignal * contrastLoss + bandLoss * 0.085 + (fixedNoise - 0.5) * sceneNoise + rowNoise + columnNoise + badPixel;
          if (bandLoss > 0.35) {
            val = lerp(val, 0.15 + vignette * 0.35 + (fixedNoise - 0.5) * sceneNoise * 0.75, (bandLoss - 0.35) * 0.55);
          }
        }
        if (saturationVisual > 0.05) {
          var flatLevel = sceneKey === "gas" ? 0.52 + shape * 0.12 : 0.58 + shape * 0.18;
          val = lerp(val, flatLevel + (fixedNoise - 0.5) * 0.035, saturationVisual * 0.50);
          val += saturationVisual * 0.08;
        }
        var col = heatColor(val);
        var idx = (py * res + px) * 4;
        img.data[idx] = col[0];
        img.data[idx + 1] = col[1];
        img.data[idx + 2] = col[2];
        img.data[idx + 3] = 255;
      }
    }
    tctx.putImageData(img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(thermalCanvas, x, y, size, size);
    ctx.restore();
    drawThermalDetection(ctx, x, y, size, m);
    ctx.strokeStyle = "#c7d1da";
    ctx.strokeRect(x, y, size, size);
  }

  function drawTargetGlyph(ctx, sceneKey, cx, cy, s) {
    ctx.save();
    ctx.fillStyle = "rgba(23,33,43,.72)";
    ctx.strokeStyle = "rgba(23,33,43,.72)";
    ctx.lineWidth = Math.max(2, s * 0.04);
    if (sceneKey === "plume" || sceneKey === "engine") {
      rr(ctx, cx - s * 0.46, cy - s * 0.08, s * 0.32, s * 0.18, s * 0.03);
      ctx.fill();
      ctx.fillStyle = "rgba(196,123,56,.85)";
      [0.00, 0.15, 0.29].forEach(function (p, idx) {
        ctx.beginPath();
        ctx.ellipse(cx - s * 0.06 + p * s, cy - s * (0.02 + idx * 0.03), s * (0.18 + idx * 0.05), s * (0.10 + idx * 0.03), -0.16, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = "rgba(184,50,42,.78)";
      ctx.beginPath(); ctx.arc(cx - s * 0.07, cy - s * 0.02, s * 0.12, 0, Math.PI * 2); ctx.fill();
    } else if (sceneKey === "gas" || sceneKey === "drone") {
      rr(ctx, cx - s * 0.40, cy - s * 0.05, s * 0.22, s * 0.42, s * 0.04);
      ctx.fill();
      ctx.beginPath(); ctx.moveTo(cx - s * 0.18, cy + s * 0.18); ctx.lineTo(cx + s * 0.30, cy + s * 0.18); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx + s * 0.02, cy + s * 0.18, s * 0.05, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = "rgba(196,123,56,.70)";
      ctx.beginPath();
      ctx.ellipse(cx + s * 0.23, cy - s * 0.02, s * 0.18, s * 0.10, -0.20, 0, Math.PI * 2);
      ctx.ellipse(cx + s * 0.36, cy + s * 0.08, s * 0.15, s * 0.09, 0.10, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(cx, cy - s * 0.28, s * 0.11, 0, Math.PI * 2); ctx.fill();
      rr(ctx, cx - s * 0.12, cy - s * 0.14, s * 0.24, s * 0.34, s * 0.08); ctx.fill();
      ctx.beginPath(); ctx.moveTo(cx - s * 0.12, cy - s * 0.05); ctx.lineTo(cx - s * 0.30, cy + s * 0.12); ctx.moveTo(cx + s * 0.12, cy - s * 0.05); ctx.lineTo(cx + s * 0.30, cy + s * 0.12); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - s * 0.07, cy + s * 0.18); ctx.lineTo(cx - s * 0.16, cy + s * 0.42); ctx.moveTo(cx + s * 0.07, cy + s * 0.18); ctx.lineTo(cx + s * 0.16, cy + s * 0.42); ctx.stroke();
    }
    ctx.restore();
  }

  function drawDeviceLegacy(m, now) {
    var o = canvasCtx("deviceCanvas"), ctx = o.ctx, W = o.w, H = o.h;
    now = now || (performance.now ? performance.now() : Date.now());
    var tAnim = now / 1000;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    var flowY = 30;
    stepBadge(ctx, 1, W * 0.05, flowY, "热辐射", "目标发出 8-14 μm 光子", "#a96716");
    stepBadge(ctx, 2, W * 0.25, flowY, "光学聚焦", "镜头把红外聚到像元", "#7a5c24");
    stepBadge(ctx, 3, W * 0.45, flowY, "T2SL 吸收", "hν ≥ Eg 产生 e-/h+", "#1f6f8b");
    stepBadge(ctx, 4, W * 0.65, flowY, "结区分离", "内建电场收集载流子", "#0e7665");
    stepBadge(ctx, 5, W * 0.82, flowY, "ROIC 输出", "积分电荷形成灰度", "#687787");

    drawArrow(ctx, W * 0.18, flowY, W * 0.23, flowY, "#c47b38", 1.6);
    drawArrow(ctx, W * 0.38, flowY, W * 0.43, flowY, "#1f6f8b", 1.6);
    drawArrow(ctx, W * 0.58, flowY, W * 0.63, flowY, "#0e7665", 1.6);
    drawArrow(ctx, W * 0.76, flowY, W * 0.80, flowY, "#687787", 1.6);

    ctx.strokeStyle = "#eef2f5";
    ctx.beginPath();
    ctx.moveTo(24, 84);
    ctx.lineTo(W - 24, 84);
    ctx.stroke();

    var sceneY = Math.max(112, H * 0.18);
    var targetX = W * 0.08, targetY = sceneY + 126;
    var lensX = W * 0.25, lensY = sceneY + 124;
    var pixX = W * 0.38, pixY = sceneY + 18, pixW = W * 0.28, pixH = Math.min(340, Math.max(310, H - sceneY - 112));
    var roicX = W * 0.70, roicW = W * 0.24, roicH = Math.min(300, Math.max(248, H - sceneY - 132));

    // Target object and blackbody cue
    var heat = clamp((P.targetT - 250) / 170, 0, 1);
    var currentScene = sceneMeta();
    var tg = ctx.createLinearGradient(targetX - 42, targetY - 55, targetX + 42, targetY + 55);
    tg.addColorStop(0, "rgba(123, 92, 36, .25)");
    tg.addColorStop(1, "rgba(196, 123, 56, " + (0.35 + heat * 0.45).toFixed(2) + ")");
    ctx.fillStyle = tg;
    rr(ctx, targetX - 48, targetY - 54, 96, 108, 12);
    ctx.fill();
    ctx.strokeStyle = "#c47b38";
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("目标物体", targetX, targetY - 12);
    ctx.fillStyle = "#7a5c24";
    ctx.font = "12px sans-serif";
    ctx.fillText(P.targetT.toFixed(0) + " K", targetX, targetY + 8);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("热辐射源", targetX, targetY + 28);

    // Lens
    ctx.strokeStyle = "#7a5c24";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(lensX, lensY, 15, 66, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "rgba(122,92,36,.10)";
    ctx.fill();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 12px sans-serif";
    ctx.fillText("红外镜头", lensX, lensY + 82);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("Ge / ZnSe", lensX, lensY + 97);

    // Thermal photon paths and animated photon dots
    ctx.strokeStyle = "rgba(196,123,56,.55)";
    ctx.save();
    ctx.shadowColor = "rgba(212,138,56,.42)";
    ctx.shadowBlur = 8;
    ctx.lineWidth = 1.5;
    for (var pr = 0; pr < 7; pr++) {
      var off = (pr - 3) * 16;
      ctx.beginPath();
      ctx.moveTo(targetX + 48, targetY + off * 0.7);
      ctx.bezierCurveTo(lensX - 55, lensY + off, lensX + 48, lensY + off * 0.4, pixX, pixY + pixH * 0.38 + off * 0.2);
      ctx.stroke();
      var p = (tAnim * (0.18 + m.photonSignal * 0.18) + pr * 0.13) % 1;
      var px = lerp(targetX + 52, pixX, p);
      var py = lerp(targetY + off * 0.7, pixY + pixH * 0.38 + off * 0.2, p) + Math.sin(p * Math.PI) * -24;
      ctx.fillStyle = "#c47b38";
      ctx.beginPath();
      ctx.arc(px, py, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // Pixel stack
    ctx.fillStyle = "#f8fafb";
    rr(ctx, pixX, pixY, pixW, pixH, 9);
    ctx.fill();
    ctx.strokeStyle = "#d9e0e7";
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("一个探测器像元截面", pixX + pixW / 2, pixY - 12);

    var winY = pixY + 18, contactTopY = pixY + 58, absY = pixY + 96, absH = 92, contactBotY = absY + absH + 34, subY = contactBotY + 44;
    var roicY = clamp(contactBotY - 73, sceneY + 58, H - roicH - 58);
    ctx.fillStyle = "#dfeaf0"; rr(ctx, pixX + 24, winY, pixW - 48, 18, 5); ctx.fill();
    ctx.fillStyle = "#66727f"; ctx.font = "10px sans-serif"; ctx.fillText("光学窗口 / passivation", pixX + pixW / 2, winY + 13);
    ctx.fillStyle = "#687787"; rr(ctx, pixX + 40, contactTopY, pixW - 80, 18, 4); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "10px sans-serif"; ctx.fillText("p contact", pixX + pixW / 2, contactTopY + 13);

    ctx.strokeStyle = "#d9e0e7";
    ctx.strokeRect(pixX + 36, absY, pixW - 72, absH);
    var n = Math.min(40, Math.max(10, Math.round(P.periods / 2)));
    var layerW = (pixW - 72) / n;
    for (var i = 0; i < n; i++) {
      var lx = pixX + 36 + i * layerW;
      var inW = layerW * P.inas / Math.max(P.inas + P.gasb, 0.01);
      ctx.fillStyle = "#3f6f9f";
      ctx.fillRect(lx, absY, Math.max(1, inW), absH);
      ctx.fillStyle = "#c47b38";
      ctx.fillRect(lx + inW, absY, Math.max(1, layerW - inW), absH);
    }
    ctx.fillStyle = "rgba(138,106,168,.45)";
    for (var j = 0; j < n; j += 4) ctx.fillRect(pixX + 36 + j * layerW, absY, Math.max(1, layerW * 0.16), absH);
    ctx.fillStyle = "#17212b";
    ctx.font = "800 11px sans-serif";
    ctx.fillText("T2SL 吸收区", pixX + pixW / 2, absY - 10);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("光子吸收产生 e-/h+", pixX + pixW / 2, absY + absH + 14);

    ctx.fillStyle = "#687787"; rr(ctx, pixX + 40, contactBotY, pixW - 80, 18, 4); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "10px sans-serif"; ctx.fillText("n contact", pixX + pixW / 2, contactBotY + 13);
    ctx.fillStyle = "#edf2f6"; rr(ctx, pixX + 20, subY, pixW - 40, 28, 5); ctx.fill();
    ctx.fillStyle = "#66727f"; ctx.font = "10px sans-serif"; ctx.fillText("GaSb substrate / bump bonds", pixX + pixW / 2, subY + 18);

    // Electric field and carrier separation
    ctx.strokeStyle = "rgba(14,118,101,.65)";
    ctx.setLineDash([5, 4]);
    drawArrow(ctx, pixX + pixW * 0.50, contactBotY + 2, pixX + pixW * 0.50, contactTopY + 20, "#0e7665", 1.5);
    ctx.setLineDash([]);
    var fieldLabelX = pixX + pixW * 0.56, fieldLabelY = absY + absH + 22;
    ctx.fillStyle = "rgba(255,255,255,.92)";
    rr(ctx, fieldLabelX, fieldLabelY - 14, 86, 20, 4);
    ctx.fill();
    ctx.fillStyle = "#0e7665";
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("内建电场 / 反偏", fieldLabelX + 6, fieldLabelY);

    var carrierCount = Math.max(3, Math.round(3 + m.qe * 8));
    for (var c = 0; c < carrierCount; c++) {
      var phase = (tAnim * 0.45 + c / carrierCount) % 1;
      var cx = pixX + 64 + ((c * 41) % Math.max(40, pixW - 128));
      var cy = absY + 12 + ((c * 23) % Math.max(20, absH - 24));
      ctx.fillStyle = "#1f6f8b";
      ctx.beginPath();
      ctx.arc(cx + Math.sin(tAnim + c) * 3, lerp(cy, contactBotY + 9, phase), 3.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#c47b38";
      ctx.beginPath();
      ctx.arc(cx + 10 + Math.cos(tAnim + c) * 3, lerp(cy, contactTopY + 9, phase), 3.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#1f6f8b"; ctx.fillText("e-", pixX + 44, contactBotY + 34);
    ctx.fillStyle = "#c47b38"; ctx.fillText("h+", pixX + 44, contactTopY - 8);

    // Dark current leakage path
    var leakAlpha = clamp(Math.log10(Math.max(m.darkRatio, 0.001)) / 2 + 0.35, 0.18, 0.82);
    ctx.strokeStyle = "rgba(184,50,42," + leakAlpha.toFixed(2) + ")";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pixX + pixW - 38, contactBotY + 8);
    ctx.bezierCurveTo(pixX + pixW + 22, absY + absH + 8, pixX + pixW + 22, absY + 10, pixX + pixW - 38, contactTopY + 8);
    ctx.stroke();
    ctx.setLineDash([]);
    var leakLabelX = pixX + pixW - 112, leakLabelY = subY + 47;
    ctx.fillStyle = "rgba(255,255,255,.94)";
    rr(ctx, leakLabelX, leakLabelY - 15, 104, 21, 4);
    ctx.fill();
    ctx.fillStyle = VIS.dark;
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("暗电流 / 表面漏电", leakLabelX + 6, leakLabelY);

    // ROIC and output image
    ctx.fillStyle = "#13202b";
    rr(ctx, roicX, roicY, roicW, roicH, 10);
    ctx.fill();
    ctx.strokeStyle = "#405366";
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("ROIC 读出电路", roicX + roicW / 2, roicY + 22);

    var fill = clamp(m.snr / 25, 0.05, 1);
    var blockX = roicX + 20, blockW = roicW - 40;
    var capX = blockX, capY = roicY + 46, capW = blockW, capH = 54;
    ctx.fillStyle = "#0f1822";
    rr(ctx, capX, capY, capW, capH, 6);
    ctx.fill();
    ctx.strokeStyle = "#5d748a";
    ctx.stroke();
    ctx.fillStyle = "#edf4f8";
    rr(ctx, capX + 12, capY + 14, capW - 24, 18, 4);
    ctx.fill();
    ctx.fillStyle = VIS.current;
    rr(ctx, capX + 12, capY + 14, (capW - 24) * fill, 18, 4);
    ctx.fill();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 11px sans-serif";
    ctx.fillText("积分电容 CTIA", roicX + roicW / 2, capY + 12);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("Iphoto - Idark 积分为电压", roicX + roicW / 2, capY + 46);

    var rowY = capY + capH + 18, unitGap = 12, unitW = (blockW - unitGap * 2) / 3, unitH = 42;
    var roicUnits = [
      ["S/H", "采样保持"],
      ["AMP", "低噪声放大"],
      ["ADC", "数字化"]
    ];
    roicUnits.forEach(function (u, idx) {
      var ux = blockX + idx * (unitW + unitGap);
      ctx.fillStyle = "#ffffff";
      rr(ctx, ux, rowY, unitW, unitH, 6);
      ctx.fill();
      ctx.strokeStyle = "#c7d1da";
      ctx.stroke();
      ctx.fillStyle = idx === 1 ? VIS.current : VIS.readout;
      ctx.font = "800 11px sans-serif";
      ctx.fillText(u[0], ux + unitW / 2, rowY + 17);
      ctx.fillStyle = "#66727f";
      ctx.font = "9px sans-serif";
      ctx.fillText(u[1], ux + unitW / 2, rowY + 31);
      if (idx < 2) drawArrow(ctx, ux + unitW + 2, rowY + unitH / 2, ux + unitW + unitGap - 3, rowY + unitH / 2, "#687787", 1.2);
    });

    var gridX = blockX + 8, gridY = rowY + unitH + 22, cols = 6, rows = 4;
    var cellGap = 4, cell = Math.min(18, (blockW - 16 - (cols - 1) * cellGap) / cols);
    for (var gy = 0; gy < rows; gy++) {
      for (var gx = 0; gx < cols; gx++) {
        var centerBoost = 1 - Math.min(1, Math.hypot(gx - 2.7, gy - 1.6) / 3.2);
        var hot = clamp(fill * 0.28 + centerBoost * 0.42 + heat * 0.22 + ((gx + gy) % 2) * 0.04, 0, 1);
        ctx.fillStyle = "rgb(" + Math.round(38 + hot * 205) + "," + Math.round(72 + hot * 112) + "," + Math.round(96 - hot * 62) + ")";
        ctx.fillRect(gridX + gx * (cell + cellGap), gridY + gy * (cell + cellGap), cell, cell);
      }
    }
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("输出灰度 / 热图像素", roicX + roicW / 2, gridY + rows * (cell + cellGap) + 12);

    drawArrow(ctx, pixX + pixW + 8, contactBotY + 9, capX - 8, capY + capH / 2, "#1f6f8b", 2);
    ctx.fillStyle = "#1f6f8b";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("光电流", pixX + pixW + 16, contactBotY - 2);

    // Legend and current state
    var legendY = H - 35;
    var legendX = W * 0.06;
    var legend = [
      ["#c47b38", "红外光子"],
      ["#1f6f8b", "电子 e-"],
      ["#c47b38", "空穴 h+"],
      ["#b8322a", "暗电流"],
      ["#0e7665", "电场收集"]
    ];
    ctx.textAlign = "left";
    legend.forEach(function (it, idx) {
      var x0 = legendX + idx * 105;
      ctx.fillStyle = it[0];
      ctx.beginPath();
      ctx.arc(x0, legendY, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#66727f";
      ctx.font = "10px sans-serif";
      ctx.fillText(it[1], x0 + 10, legendY + 4);
    });
    ctx.fillStyle = "#17212b";
    ctx.font = "800 12px sans-serif";
    ctx.fillText("当前: λc " + m.opt.lambda.toFixed(1) + " μm · QE " + pct(m.qe) + " · SNR " + m.snr.toFixed(1) + " · Jdark " + fmtExp(m.jdark) + " A/cm²", W * 0.06, H - 12);
  }

  function drawDevice(m, now) {
    var o = canvasCtx("deviceCanvas"), ctx = o.ctx, W = o.w, H = o.h;
    now = now || (performance.now ? performance.now() : Date.now());
    var tAnim = now / 1000;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    var currentScene = sceneMeta();
    var sceneBand = currentScene.band || "8-14 μm";

    var flowY = 30;
    stepBadge(ctx, 1, W * 0.05, flowY, "热辐射", "目标发出 " + sceneBand + " 光子", VIS.photon);
    stepBadge(ctx, 2, W * 0.25, flowY, "光学聚焦", "镜头把红外聚到像元", "#7a5c24");
    stepBadge(ctx, 3, W * 0.45, flowY, "T2SL 吸收", "hν ≥ Eg 产生 e-/h+", VIS.electron);
    stepBadge(ctx, 4, W * 0.65, flowY, "结区分离", "内建电场收集载流子", VIS.field);
    stepBadge(ctx, 5, W * 0.82, flowY, "ROIC 输出", "积分电荷形成灰度", VIS.readout);
    drawArrow(ctx, W * 0.18, flowY, W * 0.23, flowY, VIS.photon, 1.5);
    drawArrow(ctx, W * 0.38, flowY, W * 0.43, flowY, VIS.electron, 1.5);
    drawArrow(ctx, W * 0.58, flowY, W * 0.63, flowY, VIS.field, 1.5);
    drawArrow(ctx, W * 0.76, flowY, W * 0.80, flowY, VIS.readout, 1.5);

    ctx.strokeStyle = "#e8eef3";
    ctx.beginPath();
    ctx.moveTo(24, 68);
    ctx.lineTo(W - 24, 68);
    ctx.stroke();

    var mainTop = 76, footerTop = H - 58, mainH = footerTop - mainTop;
    var opticalX = W * 0.030, opticalW = W * 0.175;
    var pixelX = W * 0.235, pixelW = W * 0.385;
    var roicX = W * 0.650, roicW = W * 0.315;
    var panelY = mainTop + 4, panelH = mainH - 8;

    function lane(x, w, title, subtitle) {
      ctx.fillStyle = "#17212b";
      ctx.font = "800 13px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(title, x + 8, panelY + 19);
      ctx.fillStyle = "#66727f";
      ctx.font = "10px sans-serif";
      ctx.fillText(subtitle, x + 8, panelY + 35);
    }
    ctx.strokeStyle = "#edf2f6";
    ctx.lineWidth = 1;
    [pixelX - W * 0.018, roicX - W * 0.018].forEach(function (x) {
      ctx.beginPath();
      ctx.moveTo(x, panelY + 8);
      ctx.lineTo(x, panelY + panelH - 8);
      ctx.stroke();
    });
    lane(opticalX, opticalW, "光学输入", "目标热辐射被镜头聚焦到窗口");
    lane(pixelX, pixelW, "像元剖面", "II 类超晶格吸收并分离载流子");
    lane(roicX, roicW, "读出链路", "光电流积分、放大、数字化");

    var heat = clamp((P.targetT - 250) / 170, 0, 1);
    var targetX = opticalX + opticalW * 0.27, targetY = panelY + panelH * 0.45;
    var lensX = opticalX + opticalW * 0.70, lensY = targetY;
    var focusX = pixelX + pixelW * 0.50, focusY = panelY + 92;
    var targetW = clamp(opticalW * 0.62, 78, 108);
    var targetH = clamp(targetW * 1.08, 88, 118);
    var glyphSize = clamp(targetW * 0.48, 40, 54);
    var lensRx = clamp(opticalW * 0.075, 10, 15);
    var lensRy = clamp(panelH * 0.17, 54, 72);
    var tg = ctx.createLinearGradient(targetX - targetW * 0.42, targetY - targetH * 0.46, targetX + targetW * 0.42, targetY + targetH * 0.46);
    tg.addColorStop(0, "rgba(123,92,36,.16)");
    tg.addColorStop(1, "rgba(184,107,29," + (0.34 + heat * 0.44).toFixed(2) + ")");
    ctx.fillStyle = tg;
    rr(ctx, targetX - targetW / 2, targetY - targetH / 2, targetW, targetH, Math.min(14, targetW * 0.13));
    ctx.fill();
    ctx.strokeStyle = VIS.photon;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(currentScene.target, targetX, targetY - targetH * 0.26);
    try {
      drawTargetGlyph(ctx, P.scene, targetX, targetY + 2, glyphSize);
    } catch (err) {
      ctx.fillStyle = "rgba(23,33,43,.72)";
      ctx.font = "800 18px sans-serif";
      ctx.fillText("IR", targetX, targetY + 8);
    }
    ctx.fillStyle = "#7a5c24";
    ctx.font = "12px sans-serif";
    ctx.fillText(P.targetT.toFixed(0) + " K", targetX, targetY + targetH * 0.35);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText(sceneBand, targetX, targetY + targetH * 0.49);

    ctx.strokeStyle = "#7a5c24";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(lensX, lensY, lensRx, lensRy, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "rgba(122,92,36,.10)";
    ctx.fill();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 12px sans-serif";
    ctx.fillText("红外镜头", lensX, lensY + lensRy + 20);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("Ge / ZnSe", lensX, lensY + lensRy + 36);

    ctx.save();
    ctx.shadowColor = VIS.photonGlow;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 1.5;
    for (var pr = 0; pr < 7; pr++) {
      var off = (pr - 3) * 16;
      ctx.strokeStyle = "rgba(184,107,29," + (0.14 + m.bandResponse * 0.36).toFixed(2) + ")";
      ctx.beginPath();
      ctx.moveTo(targetX + targetW / 2, targetY + off * 0.72);
      ctx.bezierCurveTo(lensX - lensRx * 3.6, lensY + off, lensX + lensRx * 3.6, lensY + off * 0.28, focusX, focusY + off * 0.15);
      ctx.stroke();
      var p = (tAnim * (0.14 + m.photonSignal * 0.14 + m.bandResponse * 0.10) + pr * 0.13) % 1;
      var px = lerp(targetX + targetW * 0.54, focusX, p);
      var py = lerp(targetY + off * 0.72, focusY + off * 0.15, p) - Math.sin(p * Math.PI) * 20;
      glowDot(ctx, px, py, VIS.photon, 2.0 + m.bandResponse * 1.4, 0.38 + m.bandResponse * 0.52);
    }
    ctx.restore();

    var pixOuterX = pixelX + pixelW * 0.045, pixOuterY = panelY + 50, pixOuterW = pixelW * 0.91, pixOuterH = panelH - 86;

    var winY = pixOuterY + 24, contactTopY = winY + 44, absY = contactTopY + 50;
    var nContactH = 22, subH = 30, gapAbsToN = 34, gapNToSub = 40, bottomPad = 24;
    var maxAbsH = pixOuterY + pixOuterH - bottomPad - subH - gapNToSub - nContactH - gapAbsToN - absY;
    var absH = Math.max(136, Math.min(220, pixOuterH * 0.42, maxAbsH));
    var contactBotY = absY + absH + gapAbsToN, subY = contactBotY + nContactH + gapNToSub;
    ctx.fillStyle = "#dfeaf0";
    rr(ctx, pixOuterX + 30, winY, pixOuterW - 60, 20, 5);
    ctx.fill();
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("光学窗口 / passivation", pixOuterX + pixOuterW / 2, winY + 14);

    var pPadW = Math.max(44, pixOuterW * 0.24);
    ctx.fillStyle = "#718397";
    rr(ctx, pixOuterX + 38, contactTopY, pPadW, 22, 5);
    ctx.fill();
    rr(ctx, pixOuterX + pixOuterW - 38 - pPadW, contactTopY, pPadW, 22, 5);
    ctx.fill();
    ctx.fillStyle = "#66727f";
    ctx.font = "800 10px sans-serif";
    ctx.fillText("p contact", pixOuterX + pixOuterW / 2, contactTopY - 7);

    var absX = pixOuterX + 42, absW = pixOuterW - 84;
    ctx.fillStyle = "#f8fafb";
    ctx.fillRect(absX, absY, absW, absH);
    var fieldGrad = ctx.createLinearGradient(absX, absY, absX, absY + absH);
    fieldGrad.addColorStop(0, "rgba(184,63,122,.10)");
    fieldGrad.addColorStop(0.5, "rgba(15,138,114,.05)");
    fieldGrad.addColorStop(1, "rgba(33,102,209,.10)");
    ctx.fillStyle = fieldGrad;
    ctx.fillRect(absX, absY, absW, absH);
    ctx.strokeStyle = "#d6e0e8";
    ctx.strokeRect(absX, absY, absW, absH);
    var n = Math.min(28, Math.max(10, Math.round(P.periods / 3)));
    var layerH = absH / n;
    var edgeW = Math.min(26, absW * 0.13);
    for (var i = 0; i < n; i++) {
      var ly = absY + i * layerH;
      var inH = layerH * P.inas / Math.max(P.inas + P.gasb, 0.01);
      ctx.fillStyle = "rgba(74,160,216,.62)";
      ctx.fillRect(absX, ly, edgeW, Math.max(1, inH));
      ctx.fillRect(absX + absW - edgeW, ly, edgeW, Math.max(1, inH));
      ctx.fillStyle = "rgba(212,138,56,.62)";
      ctx.fillRect(absX, ly + inH, edgeW, Math.max(1, layerH - inH));
      ctx.fillRect(absX + absW - edgeW, ly + inH, edgeW, Math.max(1, layerH - inH));
      ctx.strokeStyle = i % 2 === 0 ? "rgba(74,160,216,.16)" : "rgba(212,138,56,.16)";
      ctx.beginPath();
      ctx.moveTo(absX + edgeW + 4, ly + layerH * 0.5);
      ctx.lineTo(absX + absW - edgeW - 4, ly + layerH * 0.5);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(169,139,214,.26)";
    for (var j = 0; j < n; j += 5) {
      ctx.fillRect(absX + edgeW, absY + j * layerH, absW - edgeW * 2, Math.max(1, layerH * 0.12));
    }
    var bandInsetH = Math.min(72, absH * 0.42);
    drawBandInset(ctx, absX + edgeW + 8, absY + 12, absW - edgeW * 2 - 16, bandInsetH, m, tAnim);
    ctx.save();
    ctx.shadowColor = VIS.photonGlow;
    ctx.shadowBlur = 8;
    var photonCount = Math.max(1, Math.min(3, Math.round(1 + m.bandResponse * 2)));
    var photonStartY = absY - 28;
    var photonEndY = absY + absH * 0.28;
    for (var ph = 0; ph < photonCount; ph++) {
      var spread = photonCount === 1 ? 0 : (ph - (photonCount - 1) / 2) * 0.105;
      var rx = absX + absW * (0.50 + spread);
      drawArrow(ctx, rx, photonStartY, rx, photonEndY, VIS.photon, 1.35);
      var rp = (tAnim * (0.26 + m.photonSignal * 0.18) + ph * 0.17) % 1;
      glowDot(ctx, rx, lerp(photonStartY + 2, photonEndY - 4, rp), VIS.photon, 2.1 + m.bandResponse * 0.9, 0.42 + m.bandResponse * 0.44);
    }
    ctx.restore();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 11px sans-serif";
    ctx.fillText("T2SL 吸收区", pixOuterX + pixOuterW / 2, absY - 12);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("hν ≥ Eg 产生电子/空穴对", pixOuterX + pixOuterW / 2, absY + absH + 18);

    ctx.fillStyle = "#718397";
    rr(ctx, pixOuterX + 46, contactBotY, pixOuterW - 92, nContactH, 5);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "800 11px sans-serif";
    ctx.fillText("n contact", pixOuterX + pixOuterW / 2, contactBotY + 15);
    ctx.fillStyle = "#edf2f6";
    rr(ctx, pixOuterX + 28, subY, pixOuterW - 56, subH, 6);
    ctx.fill();
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("GaSb substrate / bump bonds", pixOuterX + pixOuterW / 2, subY + 19);

    var fieldX = pixOuterX + pixOuterW - 36;
    ctx.save();
    ctx.strokeStyle = "rgba(15,138,114,.18)";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(fieldX, contactBotY - 2);
    ctx.lineTo(fieldX, contactTopY + 25);
    ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.lineDashOffset = -tAnim * 22;
    drawArrow(ctx, fieldX, contactBotY - 2, fieldX, contactTopY + 25, VIS.field, 1.6);
    ctx.restore();
    ctx.save();
    var fieldPulseCount = Math.max(2, Math.round(2 + m.fieldStrength * 3));
    for (var fp = 0; fp < fieldPulseCount; fp++) {
      var ft = (tAnim * (0.42 + m.fieldStrength * 0.35) + fp / fieldPulseCount) % 1;
      var fy = lerp(contactBotY - 4, contactTopY + 28, ft);
      var fr = 2.4 + Math.sin((ft + fp) * Math.PI) * 0.9;
      glowDot(ctx, fieldX, fy, VIS.field, fr, 0.42 + m.fieldStrength * 0.38);
    }
    ctx.restore();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,.92)";
    rr(ctx, fieldX - 106, contactBotY + 27, 98, 21, 4);
    ctx.fill();
    ctx.fillStyle = VIS.field;
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("内建电场 / 反偏", fieldX - 99, contactBotY + 42);

    var leftPContactX = pixOuterX + 38 + pPadW * 0.56;
    var rightPContactX = pixOuterX + pixOuterW - 38 - pPadW * 0.56;
    var carrierBaseY = absY + Math.max(bandInsetH + 22, absH * 0.54);
    var holeBendY = absY + Math.max(bandInsetH + 8, absH * 0.40);
    var carrierSeeds = [
      { x: absX + absW * 0.40, y: carrierBaseY - absH * 0.05, p: leftPContactX, n: absX + absW * 0.42 },
      { x: absX + absW * 0.47, y: carrierBaseY + absH * 0.02, p: leftPContactX + pPadW * 0.18, n: absX + absW * 0.47 },
      { x: absX + absW * 0.54, y: carrierBaseY - absH * 0.02, p: rightPContactX - pPadW * 0.18, n: absX + absW * 0.54 },
      { x: absX + absW * 0.61, y: carrierBaseY + absH * 0.05, p: rightPContactX, n: absX + absW * 0.60 }
    ];
    var electronPaths = [];
    var holePaths = [];
    carrierSeeds.forEach(function (seed) {
      electronPaths.push({
        p0: { x: seed.x, y: seed.y },
        p1: { x: seed.n - 10, y: absY + absH * 0.70 },
        p2: { x: seed.n - 4, y: contactBotY - 10 },
        p3: { x: seed.n, y: contactBotY + 9 }
      });
      holePaths.push({
        p0: { x: seed.x + 8, y: seed.y - 4 },
        p1: { x: seed.p, y: holeBendY },
        p2: { x: seed.p, y: contactTopY + 32 },
        p3: { x: seed.p, y: contactTopY + 10 }
      });
    });
    ctx.save();
    ctx.lineWidth = 1.35;
    ctx.strokeStyle = VIS.electronSoft;
    electronPaths.forEach(function (path) {
      ctx.beginPath();
      ctx.moveTo(path.p0.x, path.p0.y);
      ctx.bezierCurveTo(path.p1.x, path.p1.y, path.p2.x, path.p2.y, path.p3.x, path.p3.y);
      ctx.stroke();
    });
    ctx.strokeStyle = VIS.holeSoft;
    holePaths.forEach(function (path) {
      ctx.beginPath();
      ctx.moveTo(path.p0.x, path.p0.y);
      ctx.bezierCurveTo(path.p1.x, path.p1.y, path.p2.x, path.p2.y, path.p3.x, path.p3.y);
      ctx.stroke();
    });
    ctx.restore();

    var carrierCount = Math.max(2, Math.round(2 + m.qe * m.bandResponse * 8));
    for (var c = 0; c < carrierCount; c++) {
      var pathIndex = c % carrierSeeds.length;
      var phase = (tAnim * 0.46 + c / carrierCount) % 1;
      var electronPoint = cubicPoint(electronPaths[pathIndex].p0, electronPaths[pathIndex].p1, electronPaths[pathIndex].p2, electronPaths[pathIndex].p3, phase);
      var holePoint = cubicPoint(holePaths[pathIndex].p0, holePaths[pathIndex].p1, holePaths[pathIndex].p2, holePaths[pathIndex].p3, phase);
      ctx.fillStyle = VIS.electron;
      ctx.beginPath();
      ctx.arc(electronPoint.x + Math.sin(tAnim + c) * 1.6, electronPoint.y, 3.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = VIS.hole;
      ctx.beginPath();
      ctx.arc(holePoint.x + Math.cos(tAnim + c) * 1.6, holePoint.y, 3.3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = VIS.hole;
    ctx.font = "800 10px sans-serif";
    ctx.fillText("h+ → p", pixOuterX + 30, contactTopY - 10);
    ctx.fillStyle = VIS.electron;
    ctx.fillText("e- → n", pixOuterX + 30, contactBotY + 38);

    var leakAlpha = clamp(Math.log10(Math.max(m.darkRatio, 0.001)) / 2 + 0.35, 0.16, 0.80);
    var leakX = pixOuterX + pixOuterW - 24;
    var leakStart = { x: leakX, y: contactBotY + 10 };
    var leakCtrl1 = { x: leakX + 18, y: absY + absH + 18 };
    var leakCtrl2 = { x: leakX + 18, y: absY - 8 };
    var leakEnd = { x: leakX, y: contactTopY + 10 };
    ctx.strokeStyle = "rgba(184,50,42," + leakAlpha.toFixed(2) + ")";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    ctx.lineDashOffset = -tAnim * 24;
    ctx.beginPath();
    ctx.moveTo(leakStart.x, leakStart.y);
    ctx.bezierCurveTo(leakCtrl1.x, leakCtrl1.y, leakCtrl2.x, leakCtrl2.y, leakEnd.x, leakEnd.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;
    var leakPulseCount = Math.max(1, Math.round(1 + m.darkActivity * 4));
    for (var lp = 0; lp < leakPulseCount; lp++) {
      var lt = (tAnim * (0.22 + m.darkActivity * 0.48) + lp / leakPulseCount) % 1;
      var leakPt = cubicPoint(leakStart, leakCtrl1, leakCtrl2, leakEnd, lt);
      glowDot(ctx, leakPt.x, leakPt.y, VIS.dark, 2.2 + m.darkActivity * 2.4, 0.32 + m.darkActivity * 0.55);
    }
    ctx.fillStyle = "rgba(255,255,255,.94)";
    rr(ctx, leakX - 116, subY + 38, 112, 22, 4);
    ctx.fill();
    ctx.fillStyle = VIS.dark;
    ctx.font = "800 10px sans-serif";
    ctx.fillText("暗电流 / 表面漏电", leakX - 109, subY + 53);

    var roicOuterX = roicX + roicW * 0.035, roicOuterY = panelY + 50, roicOuterW = roicW * 0.93, roicOuterH = panelH - 86;
    ctx.fillStyle = "#17212b";
    ctx.font = "800 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("ROIC 读出电路", roicOuterX + roicOuterW / 2, roicOuterY + 16);

    var blockX = roicOuterX + 12, blockW = roicOuterW - 24;
    var capY = roicOuterY + 28, capH = 70;
    drawIntegrationCurve(ctx, blockX, capY, blockW, capH, m, tAnim);

    var chainY = capY + capH + 10, unitGap = 8, unitW = (blockW - unitGap * 2) / 3, unitH = 34;
    [["S/H", "采样保持"], ["AMP", "低噪声放大"], ["ADC", "数字化"]].forEach(function (u, idx) {
      var ux = blockX + idx * (unitW + unitGap);
      ctx.fillStyle = "#ffffff";
      rr(ctx, ux, chainY, unitW, unitH, 7);
      ctx.fill();
      ctx.strokeStyle = "#c7d1da";
      ctx.stroke();
      ctx.fillStyle = idx === 1 ? VIS.current : VIS.readout;
      ctx.font = "800 10px sans-serif";
      ctx.fillText(u[0], ux + unitW / 2, chainY + 13);
      ctx.fillStyle = "#66727f";
      ctx.font = "9px sans-serif";
      ctx.fillText(u[1], ux + unitW / 2, chainY + 25);
      ctx.fillStyle = idx === 1 ? VIS.field : VIS.photon;
      ctx.beginPath();
      ctx.arc(ux + 10, chainY + 10, 3, 0, Math.PI * 2);
      ctx.fill();
      if (idx < 2) drawArrow(ctx, ux + unitW + 2, chainY + unitH / 2, ux + unitW + unitGap - 3, chainY + unitH / 2, VIS.readout, 1.2);
    });

    var gridY = chainY + unitH + 12;
    var mapSpace = roicOuterY + roicOuterH - gridY - 24;
    var mapSize = Math.max(96, Math.min(blockW, mapSpace));
    var gridX = roicOuterX + roicOuterW / 2 - mapSize / 2;
    try {
      drawThermalOutput(ctx, gridX, gridY, mapSize, m, heat, tAnim);
    } catch (err2) {
      ctx.fillStyle = "#223642";
      ctx.fillRect(gridX, gridY, mapSize, mapSize);
      ctx.fillStyle = "#f4c46d";
      ctx.beginPath();
      ctx.arc(gridX + mapSize * 0.5, gridY + mapSize * 0.42, mapSize * 0.10, 0, Math.PI * 2);
      ctx.fill();
      rr(ctx, gridX + mapSize * 0.40, gridY + mapSize * 0.52, mapSize * 0.20, mapSize * 0.26, 8);
      ctx.fill();
    }
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText(currentScene.resolution + " 输出热图 · " + currentScene.label, roicOuterX + roicOuterW / 2, gridY + mapSize + 16);

    var currentY = capY + capH / 2;
    ctx.save();
    ctx.shadowColor = "rgba(47,143,172,.65)";
    ctx.shadowBlur = 9;
    drawArrow(ctx, pixOuterX + pixOuterW + 8, currentY, roicOuterX - 10, currentY, VIS.current, 2.2);
    ctx.restore();
    ctx.fillStyle = VIS.current;
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("光电流", pixOuterX + pixOuterW + 16, currentY - 8);

    var legendY = footerTop + 28;
    var legendX = W * 0.06;
    var legend = [
      [VIS.photon, "红外光子"],
      [VIS.electron, "电子 e-"],
      [VIS.hole, "空穴 h+"],
      [VIS.dark, "暗电流"],
      [VIS.field, "电场收集"]
    ];
    ctx.textAlign = "left";
    legend.forEach(function (it, idx) {
      var x0 = legendX + idx * 110;
      ctx.fillStyle = it[0];
      ctx.beginPath();
      ctx.arc(x0, legendY, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#66727f";
      ctx.font = "10px sans-serif";
      ctx.fillText(it[1], x0 + 10, legendY + 4);
    });
    ctx.fillStyle = "#17212b";
    ctx.font = "800 12px sans-serif";
    ctx.fillText("当前: λc " + m.opt.lambda.toFixed(1) + " μm · QE " + pct(m.qe) + " · SNR " + m.snr.toFixed(1) + " · Jdark " + fmtExp(m.jdark) + " A/cm²", W * 0.06, footerTop + 56);
  }

  function drawBand(m) {
    var o = canvasCtx("bandCanvas"), ctx = o.ctx, W = o.w, H = o.h;
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    var pad = 34, mid = H * 0.52;
    ctx.strokeStyle = "#3f6f9f"; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(pad, mid - 40); ctx.bezierCurveTo(W * .28, mid - 72, W * .42, mid - 18, W * .54, mid - 42); ctx.bezierCurveTo(W * .68, mid - 66, W * .78, mid - 16, W - pad, mid - 38); ctx.stroke();
    ctx.strokeStyle = "#c47b38";
    ctx.beginPath(); ctx.moveTo(pad, mid + 36); ctx.bezierCurveTo(W * .30, mid + 16, W * .44, mid + 66, W * .58, mid + 36); ctx.bezierCurveTo(W * .72, mid + 8, W * .80, mid + 62, W - pad, mid + 32); ctx.stroke();
    ctx.fillStyle = "rgba(63,111,159,.14)"; ctx.fillRect(W * .18, 36, W * .23, H - 78);
    ctx.fillStyle = "rgba(196,123,56,.14)"; ctx.fillRect(W * .56, 36, W * .23, H - 78);
    ctx.fillStyle = "#66727f"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("electron well: InAs", W * .30, H - 18);
    ctx.fillText("hole well: GaSb", W * .68, H - 18);
    ctx.fillStyle = "#17212b"; ctx.font = "700 12px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("Eg_eff " + m.opt.eg.toFixed(3) + " eV", pad, 20);
    ctx.fillText("wavefunction overlap " + pct(m.overlap), pad, 38);
    var cx = W * .48, cy = mid;
    ctx.strokeStyle = "#7a5c24"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(cx, cy - 42); ctx.lineTo(cx, cy + 36); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#7a5c24"; ctx.font = "10px sans-serif"; ctx.fillText("hν ≥ Eg", cx + 8, cy - 4);
  }

  function drawSignal(m, now) {
    var o = canvasCtx("signalCanvas"), ctx = o.ctx, W = o.w, H = o.h;
    var tAnim = (now || (performance.now ? performance.now() : Date.now())) / 1000;
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    var padL = 34, padT = 34, padB = 36, leftW = W * 0.36, ph = H - padT - padB;
    ctx.strokeStyle = "#d9e0e7"; ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(padL + leftW, padT + ph); ctx.stroke();
    var sig = clamp(Math.log10(Math.max(m.iph, 1e-14)) + 13, 0, 7) / 7;
    var dark = clamp(Math.log10(Math.max(m.idark, 1e-14)) + 13, 0, 7) / 7;
    function bar(x, val, color, label) {
      var h = val * ph;
      ctx.fillStyle = color; rr(ctx, x, padT + ph - h, leftW * .22, h, 5); ctx.fill();
      ctx.fillStyle = "#66727f"; ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.fillText(label, x + leftW * .11, padT + ph + 18);
    }
    bar(padL + leftW * .18, sig, VIS.current, "photo");
    bar(padL + leftW * .55, dark, VIS.dark, "dark");
    ctx.strokeStyle = VIS.field; ctx.lineWidth = 2;
    var y = padT + ph * (1 - clamp(m.snr / 25, 0, 1));
    ctx.beginPath(); ctx.moveTo(padL + leftW * .06, y); ctx.lineTo(padL + leftW * .92, y); ctx.stroke();
    ctx.fillStyle = VIS.field; ctx.font = "700 11px sans-serif"; ctx.textAlign = "left"; ctx.fillText("SNR " + m.snr.toFixed(1), padL + 4, y - 6);
    ctx.fillStyle = "#17212b"; ctx.font = "800 12px sans-serif"; ctx.fillText("光电流 / 暗电流竞争", padL, 18);
    drawIntegrationCurve(ctx, W * 0.45, 24, W * 0.50, H - 48, m, tAnim);
  }

  function drawTrend(m) {
    var o = canvasCtx("trendCanvas"), ctx = o.ctx, W = o.w, H = o.h;
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    var padL = 48, padR = 18, padT = 22, padB = 32, pw = W - padL - padR, ph = H - padT - padB;
    ctx.strokeStyle = "#d9e0e7"; ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(padL + pw, padT + ph); ctx.stroke();
    ctx.fillStyle = "#66727f"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    [80, 120, 180, 240, 300].forEach(function (t) {
      var x = padL + (t - 60) / 240 * pw;
      ctx.fillText(t + "K", x, padT + ph + 18);
      ctx.strokeStyle = "#eef2f5"; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke();
    });
    ctx.strokeStyle = "#b8322a"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i <= 120; i++) {
      var T = 60 + i * 2;
      var tempTerm = Math.exp(-m.opt.eg / (2 * 8.617e-5 * T));
      var yv = clamp((Math.log10(tempTerm + 1e-18) + 16) / 12, 0, 1);
      var x = padL + (T - 60) / 240 * pw, y = padT + (1 - yv) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    var curX = padL + (P.detT - 60) / 240 * pw;
    ctx.strokeStyle = "#1f6f8b"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(curX, padT); ctx.lineTo(curX, padT + ph); ctx.stroke();
    ctx.fillStyle = "#17212b"; ctx.font = "700 12px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("dark-current thermal activation trend", padL, 16);
    ctx.fillStyle = "#1f6f8b"; ctx.fillText("current " + P.detT.toFixed(0) + " K", curX + 8, padT + 18);
  }

  function render() {
    readInputs();
    updateLabels();
    var m = model();
    latestModel = m;
    updateReadout(m);
    drawDevice(m);
    if (viewMode === "experiment") {
      drawBand(m);
      drawSignal(m);
      drawTrend(m);
    }
  }

  function animationLoop(ts) {
    if (latestModel) {
      drawDevice(latestModel, ts);
      if (viewMode === "experiment") drawSignal(latestModel, ts);
    }
    requestAnimationFrame(animationLoop);
  }

  function applyPreset(kind) {
    setActiveScenario("");
    setReferenceDataKey(dataKeyForPreset(kind));
    if (applyScenarioData("preset", "")) return;
    activeReferenceId = "";
    if (kind === "mwir") {
      Object.assign(P, { scene: "plume", inas: 2.42, gasb: 2.44, periods: 100, interfaceNm: 0.09, interfaceQ: 0.90, mBarrier: false, detT: 77, bias: 0.05, xrd: 56, afm: 1.9, passivation: 0.82, targetT: 760, bgT: 300, tint: 2.5 });
    } else if (kind === "bbir") {
      Object.assign(P, { scene: "gas", inas: 4.24, gasb: 2.13, periods: 100, interfaceNm: 0.18, interfaceQ: 0.72, mBarrier: true, detT: 77, bias: 0.05, xrd: 58, afm: 2.8, passivation: 0.72, targetT: 325, bgT: 300, tint: 8.0 });
    } else {
      Object.assign(P, { scene: "human", inas: 3.63, gasb: 1.22, periods: 100, interfaceNm: 0.18, interfaceQ: 0.84, mBarrier: true, detT: 77, bias: 0.05, xrd: 64, afm: 2.4, passivation: 0.78, targetT: 310, bgT: 293, tint: 5 });
    }
    syncControls();
    render();
  }

  function applyIssuePreset(kind) {
    setReferenceDataKey(dataKeyForScene(P.scene));
    if (kind === "reset") {
      if (applyScenarioData("preset", "reset")) return;
    }
    if (applyScenarioData(kind, kind)) return;
    activeReferenceId = "";
    if (kind === "band") {
      Object.assign(P, { scene: "human", inas: 2.42, gasb: 2.44, periods: 100, interfaceNm: 0.09, interfaceQ: 0.90, mBarrier: false, detT: 77, bias: 0.05, xrd: 56, afm: 1.9, passivation: 0.82, targetT: 310, bgT: 293, tint: 5.0 });
    } else if (kind === "hot") {
      Object.assign(P, { scene: "human", inas: 3.63, gasb: 1.22, periods: 100, interfaceNm: 0.18, interfaceQ: 0.84, mBarrier: true, detT: 150, bias: 0.05, xrd: 64, afm: 2.4, passivation: 0.78, targetT: 310, bgT: 293, tint: 5.0 });
    } else if (kind === "sat") {
      Object.assign(P, { scene: "human", inas: 3.63, gasb: 1.22, periods: 100, interfaceNm: 0.18, interfaceQ: 0.84, mBarrier: true, detT: 77, bias: 0.05, xrd: 64, afm: 2.4, passivation: 0.78, targetT: 315, bgT: 293, tint: 18.0 });
    } else if (kind === "dark") {
      Object.assign(P, { scene: "human", inas: 4.24, gasb: 2.13, periods: 100, interfaceNm: 0.28, interfaceQ: 0.52, mBarrier: false, detT: 150, bias: 0.12, xrd: 90, afm: 5.5, passivation: 0.48, targetT: 310, bgT: 293, tint: 7.0 });
    } else if (kind === "qe") {
      Object.assign(P, { scene: "human", inas: 4.24, gasb: 2.13, periods: 100, interfaceNm: 0.30, interfaceQ: 0.46, mBarrier: true, detT: 77, bias: 0.05, xrd: 90, afm: 5.5, passivation: 0.42, targetT: 310, bgT: 293, tint: 5.0 });
    } else {
      Object.assign(P, { scene: "human", inas: 3.63, gasb: 1.22, periods: 100, interfaceNm: 0.18, interfaceQ: 0.84, mBarrier: true, detT: 77, bias: 0.05, xrd: 64, afm: 2.4, passivation: 0.78, targetT: 310, bgT: 293, tint: 5.0 });
      kind = "reset";
    }
    setActiveScenario(kind);
    syncControls();
    render();
  }

  function init() {
    ids.forEach(function (id) { els[id] = $(id); });
    setViewMode("experiment");
    setMechanismFocus("band");
    Object.keys(P).forEach(function (k) {
      if (!els[k]) return;
      els[k].addEventListener("input", function () { activeReferenceId = ""; setActiveScenario(""); render(); });
      els[k].addEventListener("change", function () { activeReferenceId = ""; setActiveScenario(""); render(); });
    });
    els.modePrinciple.addEventListener("click", function () {
      setViewMode("principle");
      requestAnimationFrame(render);
    });
    els.modeExperiment.addEventListener("click", function () {
      setViewMode("experiment");
      requestAnimationFrame(render);
    });
    els.focusBand.addEventListener("click", function () { setMechanismFocus("band"); });
    els.focusDark.addEventListener("click", function () { setMechanismFocus("dark"); });
    els.focusReadout.addEventListener("click", function () { setMechanismFocus("readout"); });
    els.presetLwir.addEventListener("click", function () { applyPreset("lwir"); });
    els.presetMwir.addEventListener("click", function () { applyPreset("mwir"); });
    els.presetBbir.addEventListener("click", function () { applyPreset("bbir"); });
    els.issueBand.addEventListener("click", function () { applyIssuePreset("band"); });
    els.issueHot.addEventListener("click", function () { applyIssuePreset("hot"); });
    els.issueSat.addEventListener("click", function () { applyIssuePreset("sat"); });
    els.issueDark.addEventListener("click", function () { applyIssuePreset("dark"); });
    els.issueQe.addEventListener("click", function () { applyIssuePreset("qe"); });
    els.issueReset.addEventListener("click", function () { applyIssuePreset("reset"); });
    window.addEventListener("resize", render);
    loadReferenceData().then(function (loaded) {
      setReferenceDataKey("lwir");
      if (loaded && applyScenarioData("preset", "")) {
        requestAnimationFrame(animationLoop);
        return;
      }
      render();
      requestAnimationFrame(animationLoop);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
