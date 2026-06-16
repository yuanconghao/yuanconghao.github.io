(function () {
  "use strict";

  var P = {
    inas: 5.45,
    gasb: 3.03,
    periods: 70,
    interfaceNm: 0.20,
    interfaceQ: 0.86,
    mBarrier: true,
    detT: 77,
    bias: 0.05,
    pixel: 25,
    tint: 5,
    targetT: 310,
    bgT: 293,
    xrd: 24,
    afm: 2.0,
    passivation: 0.80,
    scene: "human"
  };

  var scenes = {
    human: {
      label: "LWIR · 行人",
      target: "行人目标",
      note: "8-14 μm 常温热辐射峰值",
      band: "LWIR 8-14 μm",
      resolution: "128 × 128"
    },
    plume: {
      label: "MWIR · 尾焰",
      target: "发动机尾焰",
      note: "3-5 μm 高温目标强辐射",
      band: "MWIR 3-5 μm",
      resolution: "128 × 128"
    },
    gas: {
      label: "BBIR · 气体",
      target: "气体泄漏场景",
      note: "2-14 μm 宽带气体/宽域侦测",
      band: "BBIR 2-14 μm",
      resolution: "128 × 128"
    }
  };

  var anchors = [
    { name: "Delmas 5.15 μm", inas: 2.12, gasb: 1.21, al: 0, lambda: 5.15 },
    { name: "Chen 5.30 μm", inas: 2.43, gasb: 2.42, al: 0, lambda: 5.30 },
    { name: "Xie 5.00 μm", inas: 2.40, gasb: 2.40, al: 0, lambda: 5.00 },
    { name: "Jiang M-type 14 μm", inas: 5.45, gasb: 3.03, al: 1.52, lambda: 14.00 }
  ];

  var els = {};
  var latestModel = null;
  var ids = [
    "inas", "gasb", "periods", "interfaceNm", "interfaceQ", "mBarrier", "detT", "bias", "pixel", "tint", "targetT", "bgT", "xrd", "afm", "passivation",
    "inasV", "gasbV", "periodsV", "interfaceNmV", "interfaceQV", "detTV", "biasV", "pixelV", "tintV", "targetTV", "bgTV", "xrdV", "afmV", "passivationV",
    "lambdaHero", "bandHero", "snrHero", "egOut", "absOut", "qeOut", "iphOut", "jdarkOut", "r0aOut", "dstarOut", "netdOut",
    "absBadge", "darkBadge", "readBadge", "absDiag", "darkDiag", "readDiag",
    "chainRad", "chainAbs", "chainSep", "chainNoise", "chainRoi",
    "presetLwir", "presetMwir", "presetBbir"
  ];

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

  function model() {
    var opt = opticalEstimate();
    var periodNm = P.inas + P.gasb + P.interfaceNm * 2;
    var absNm = periodNm * P.periods;
    var absUm = absNm / 1000;
    var materialQ = clamp(1 - (P.xrd - 18) / 170 - (P.afm - 1.2) / 25, 0.25, 1);
    var overlap = clamp(0.30 + P.interfaceQ * 0.52 + (P.mBarrier ? 0.05 : 0) - Math.max(0, opt.lambda - 14) * 0.025, 0.15, 0.88);
    var absorption = 1 - Math.exp(-1.15 * absUm * clamp(opt.lambda / 10, 0.55, 1.35));
    var qe = clamp(absorption * overlap * materialQ * (0.78 + P.passivation * 0.22), 0.03, 0.78);
    var photonSignal = Math.max(0, (Math.pow(P.targetT / 300, 4) - Math.pow(P.bgT / 300, 4)));
    var pixelAreaCm2 = Math.pow(P.pixel * 1e-4, 2);
    var iph = 2.2e-7 * qe * photonSignal * pixelAreaCm2 * 1e6;
    var tempTerm = Math.exp(-opt.eg / (2 * 8.617e-5 * Math.max(P.detT, 1)));
    var defect = 0.20 + (1 - materialQ) * 1.4 + (1 - P.interfaceQ) * 1.2 + (1 - P.passivation) * 1.0;
    var biasTerm = 1 + Math.pow(P.bias / 0.08, 1.7);
    var barrierTerm = P.mBarrier ? 0.34 : 1.0;
    var jdark = clamp(8e1 * tempTerm * defect * biasTerm * barrierTerm, 1e-10, 4e-1);
    var idark = jdark * pixelAreaCm2;
    var noise = Math.sqrt(Math.max(idark, 1e-18) * Math.max(P.tint, 0.1) / 5) * 2.0e-2 + 1.5e-12;
    var snr = iph / noise;
    var r0a = clamp(0.025 / Math.max(jdark, 1e-12), 1e-2, 1e9);
    var dstar = clamp(1.0e10 * qe / Math.sqrt(Math.max(jdark, 1e-10) / 1e-6) * (P.pixel / 25), 1e7, 8e12);
    var netd = clamp(90 / Math.sqrt(Math.max(snr, 0.05)) + (1 - qe) * 18 + Math.max(0, P.detT - 120) * 0.45, 5, 500);
    var darkRatio = idark / Math.max(iph, 1e-18);
    return {
      opt: opt,
      periodNm: periodNm,
      absNm: absNm,
      materialQ: materialQ,
      overlap: overlap,
      absorption: absorption,
      qe: qe,
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
    setText("absDiag", "当前观察对象为" + s.label + "，截止波长约 " + m.opt.lambda.toFixed(1) + " μm。红外光子进入 T2SL 吸收区后产生电子/空穴对；QE 表示入射光子最终变成可收集电荷的效率。");

    var darkCls = m.darkRatio < 0.25 ? "good" : m.darkRatio < 2 ? "warn" : "bad";
    setBadge("darkBadge", darkCls === "good" ? "暗电流受控" : darkCls === "warn" ? "暗电流接近信号" : "暗电流主导", darkCls);
    setText("darkDiag", "探测器温度升高或反向偏压过大，会让没有光照也产生的暗电流变强；暗电流越接近光电流，图像越容易被噪声淹没。");

    var readCls = m.snr > 10 ? "good" : m.snr > 2 ? "warn" : "bad";
    setBadge("readBadge", readCls === "good" ? "ROIC 可稳定积分" : readCls === "warn" ? "读出临界" : "信号偏弱", readCls);
    setText("readDiag", "目标与背景热辐射差形成光电流；积分时间越长信号越多，但暗电流和饱和风险也会增加。当前 SNR 趋势约 " + m.snr.toFixed(1) + "。");

    [["chainRad", m.photonSignal > 0.01], ["chainAbs", m.qe > 0.18], ["chainSep", m.overlap > 0.45], ["chainNoise", m.darkRatio < 2], ["chainRoi", m.snr > 2]].forEach(function (it) {
      if (els[it[0]]) els[it[0]].classList.toggle("on", !!it[1]);
    });
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
  function drawThermalOutput(ctx, x, y, size, m, heat, tAnim) {
    var res = 128;
    if (!thermalCanvas) thermalCanvas = document.createElement("canvas");
    thermalCanvas.width = res;
    thermalCanvas.height = res;
    var tctx = thermalCanvas.getContext("2d");
    var img = tctx.createImageData(res, res);
    var sceneKey = P.scene || "human";
    var sceneSignal = sceneKey === "gas" ? clamp(m.snr / 18, 0.28, 0.74) : clamp(0.70 + heat * 0.30, 0.70, 1);
    var sceneNoise = sceneKey === "gas" ? 0.16 : 0.045;
    for (var py = 0; py < res; py++) {
      for (var px = 0; px < res; px++) {
        var nx = px / (res - 1), ny = py / (res - 1);
        var shape = Math.pow(clamp(objectHeat(sceneKey, nx, ny), 0, 1), 0.72);
        var vignette = 0.08 * (1 - Math.min(1, Math.hypot(nx - 0.5, ny - 0.5) / 0.72));
        var fixedNoise = Math.sin(px * 12.9898 + py * 78.233 + tAnim * 2.1) * 43758.5453;
        fixedNoise = fixedNoise - Math.floor(fixedNoise);
        var rowNoise = Math.sin(py * 0.55 + tAnim * 1.4) * 0.035;
        var val = 0.10 + vignette + shape * (0.62 + heat * 0.30) * sceneSignal + (fixedNoise - 0.5) * sceneNoise + rowNoise;
        if (sceneKey === "plume") val += blob({ x: nx, y: ny }, 0.46, 0.52, 0.14, 0.09) * 0.30;
        if (sceneKey === "gas") {
          var plumeBand = Math.max(blob({ x: nx, y: ny }, 0.64, 0.39, 0.24, 0.17), blob({ x: nx, y: ny }, 0.74, 0.52, 0.22, 0.15));
          val = 0.13 + shape * 0.48 * sceneSignal + plumeBand * 0.16 + (fixedNoise - 0.5) * sceneNoise + rowNoise * 0.55;
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

    var flowY = 38;
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
    ctx.fillStyle = "#b8322a";
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
    ctx.fillStyle = "#1f6f8b";
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
      ctx.fillStyle = idx === 1 ? "#1f6f8b" : "#687787";
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

    var flowY = 38;
    stepBadge(ctx, 1, W * 0.05, flowY, "热辐射", "目标发出 " + sceneBand + " 光子", "#a96716");
    stepBadge(ctx, 2, W * 0.25, flowY, "光学聚焦", "镜头把红外聚到像元", "#7a5c24");
    stepBadge(ctx, 3, W * 0.45, flowY, "T2SL 吸收", "hν ≥ Eg 产生 e-/h+", "#1f6f8b");
    stepBadge(ctx, 4, W * 0.65, flowY, "结区分离", "内建电场收集载流子", "#0e7665");
    stepBadge(ctx, 5, W * 0.82, flowY, "ROIC 输出", "积分电荷形成灰度", "#687787");
    drawArrow(ctx, W * 0.18, flowY, W * 0.23, flowY, "#c47b38", 1.5);
    drawArrow(ctx, W * 0.38, flowY, W * 0.43, flowY, "#1f6f8b", 1.5);
    drawArrow(ctx, W * 0.58, flowY, W * 0.63, flowY, "#0e7665", 1.5);
    drawArrow(ctx, W * 0.76, flowY, W * 0.80, flowY, "#687787", 1.5);

    ctx.strokeStyle = "#e8eef3";
    ctx.beginPath();
    ctx.moveTo(28, 88);
    ctx.lineTo(W - 28, 88);
    ctx.stroke();

    var mainTop = 108, footerTop = H - 76, mainH = footerTop - mainTop;
    var opticalX = W * 0.035, opticalW = W * 0.265;
    var pixelX = W * 0.325, pixelW = W * 0.285;
    var roicX = W * 0.635, roicW = W * 0.330;
    var panelY = mainTop + 18, panelH = mainH - 36;

    function lane(x, w, title, subtitle) {
      ctx.fillStyle = "#fbfcfd";
      rr(ctx, x, panelY, w, panelH, 12);
      ctx.fill();
      ctx.strokeStyle = "#edf2f6";
      ctx.stroke();
      ctx.fillStyle = "#17212b";
      ctx.font = "800 13px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(title, x + 18, panelY + 26);
      ctx.fillStyle = "#66727f";
      ctx.font = "10px sans-serif";
      ctx.fillText(subtitle, x + 18, panelY + 43);
    }
    lane(opticalX, opticalW, "光学输入", "目标热辐射被镜头聚焦到窗口");
    lane(pixelX, pixelW, "像元剖面", "II 类超晶格吸收并分离载流子");
    lane(roicX, roicW, "读出链路", "光电流积分、放大、数字化");

    var heat = clamp((P.targetT - 250) / 170, 0, 1);
    var targetX = opticalX + opticalW * 0.23, targetY = panelY + panelH * 0.44;
    var lensX = opticalX + opticalW * 0.58, lensY = targetY;
    var focusX = pixelX + pixelW * 0.50, focusY = panelY + 104;
    var tg = ctx.createLinearGradient(targetX - 48, targetY - 58, targetX + 48, targetY + 58);
    tg.addColorStop(0, "rgba(123,92,36,.16)");
    tg.addColorStop(1, "rgba(196,123,56," + (0.34 + heat * 0.44).toFixed(2) + ")");
    ctx.fillStyle = tg;
    rr(ctx, targetX - 58, targetY - 62, 116, 124, 14);
    ctx.fill();
    ctx.strokeStyle = "#c47b38";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(currentScene.target, targetX, targetY - 22);
    try {
      drawTargetGlyph(ctx, P.scene, targetX, targetY + 2, 58);
    } catch (err) {
      ctx.fillStyle = "rgba(23,33,43,.72)";
      ctx.font = "800 18px sans-serif";
      ctx.fillText("IR", targetX, targetY + 8);
    }
    ctx.fillStyle = "#7a5c24";
    ctx.font = "12px sans-serif";
    ctx.fillText(P.targetT.toFixed(0) + " K", targetX, targetY + 44);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText(currentScene.note, targetX, targetY + 61);

    ctx.strokeStyle = "#7a5c24";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(lensX, lensY, 15, 72, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "rgba(122,92,36,.10)";
    ctx.fill();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 12px sans-serif";
    ctx.fillText("红外镜头", lensX, lensY + 92);
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("Ge / ZnSe", lensX, lensY + 108);

    ctx.save();
    ctx.shadowColor = "rgba(212,138,56,.42)";
    ctx.shadowBlur = 8;
    ctx.lineWidth = 1.5;
    for (var pr = 0; pr < 7; pr++) {
      var off = (pr - 3) * 16;
      ctx.strokeStyle = "rgba(196,123,56,.44)";
      ctx.beginPath();
      ctx.moveTo(targetX + 58, targetY + off * 0.72);
      ctx.bezierCurveTo(lensX - 52, lensY + off, lensX + 54, lensY + off * 0.28, focusX, focusY + off * 0.15);
      ctx.stroke();
      var p = (tAnim * (0.18 + m.photonSignal * 0.18) + pr * 0.13) % 1;
      var px = lerp(targetX + 62, focusX, p);
      var py = lerp(targetY + off * 0.72, focusY + off * 0.15, p) - Math.sin(p * Math.PI) * 20;
      ctx.fillStyle = "#c47b38";
      ctx.beginPath();
      ctx.arc(px, py, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    var pixOuterX = pixelX + pixelW * 0.08, pixOuterY = panelY + 52, pixOuterW = pixelW * 0.84, pixOuterH = Math.min(560, panelH - 70);
    ctx.fillStyle = "#ffffff";
    rr(ctx, pixOuterX, pixOuterY, pixOuterW, pixOuterH, 10);
    ctx.fill();
    ctx.strokeStyle = "#d6e0e8";
    ctx.stroke();

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
    fieldGrad.addColorStop(0, "rgba(212,138,56,.10)");
    fieldGrad.addColorStop(0.5, "rgba(57,169,200,.05)");
    fieldGrad.addColorStop(1, "rgba(74,160,216,.10)");
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
    ctx.save();
    ctx.shadowColor = "rgba(212,138,56,.55)";
    ctx.shadowBlur = 8;
    for (var ph = 0; ph < 3; ph++) {
      var rx = absX + absW * (0.40 + ph * 0.10);
      drawArrow(ctx, rx, winY + 26, rx, absY + absH * 0.70, "#d48a38", 1.5);
      var rp = (tAnim * (0.26 + m.photonSignal * 0.18) + ph * 0.17) % 1;
      ctx.fillStyle = "#d48a38";
      ctx.beginPath();
      ctx.arc(rx, lerp(winY + 28, absY + absH * 0.70, rp), 3, 0, Math.PI * 2);
      ctx.fill();
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
    ctx.setLineDash([5, 5]);
    drawArrow(ctx, fieldX, contactBotY - 2, fieldX, contactTopY + 25, "#0e7665", 1.6);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,.92)";
    rr(ctx, fieldX - 106, contactBotY + 27, 98, 21, 4);
    ctx.fill();
    ctx.fillStyle = "#0e7665";
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("内建电场 / 反偏", fieldX - 99, contactBotY + 42);

    ctx.save();
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = "rgba(74,160,216,.34)";
    for (var ep = 0; ep < 4; ep++) {
      var pathX = absX + absW * (0.34 + ep * 0.10);
      ctx.beginPath();
      ctx.moveTo(pathX, absY + absH * 0.52);
      ctx.bezierCurveTo(pathX - 12, absY + absH * 0.68, pathX - 4, contactBotY - 8, pathX, contactBotY + 8);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(212,138,56,.34)";
    for (var hp = 0; hp < 4; hp++) {
      var hPathX = absX + absW * (0.39 + hp * 0.10);
      ctx.beginPath();
      ctx.moveTo(hPathX, absY + absH * 0.48);
      ctx.bezierCurveTo(hPathX + 10, absY + absH * 0.30, hPathX + 2, contactTopY + 32, hPathX, contactTopY + 10);
      ctx.stroke();
    }
    ctx.restore();

    var carrierCount = Math.max(3, Math.round(3 + m.qe * 5));
    for (var c = 0; c < carrierCount; c++) {
      var phase = (tAnim * 0.46 + c / carrierCount) % 1;
      var cx = absX + absW * (0.34 + (c % 4) * 0.10);
      var cy = absY + absH * (0.36 + ((c * 17) % 24) / 100);
      ctx.fillStyle = "#4aa0d8";
      ctx.beginPath();
      ctx.arc(cx + Math.sin(tAnim + c) * 3, lerp(cy, contactBotY + 11, phase), 3.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#d48a38";
      ctx.beginPath();
      ctx.arc(cx + 10 + Math.cos(tAnim + c) * 3, lerp(cy, contactTopY + 11, phase), 3.3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#d48a38";
    ctx.font = "800 10px sans-serif";
    ctx.fillText("h+ → p", pixOuterX + 30, contactTopY - 10);
    ctx.fillStyle = "#4aa0d8";
    ctx.fillText("e- → n", pixOuterX + 30, contactBotY + 38);

    var leakAlpha = clamp(Math.log10(Math.max(m.darkRatio, 0.001)) / 2 + 0.35, 0.16, 0.80);
    var leakX = pixOuterX + pixOuterW - 24;
    ctx.strokeStyle = "rgba(184,50,42," + leakAlpha.toFixed(2) + ")";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(leakX, contactBotY + 10);
    ctx.bezierCurveTo(leakX + 18, absY + absH + 18, leakX + 18, absY - 8, leakX, contactTopY + 10);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,.94)";
    rr(ctx, leakX - 116, subY + 38, 112, 22, 4);
    ctx.fill();
    ctx.fillStyle = "#b8322a";
    ctx.font = "800 10px sans-serif";
    ctx.fillText("暗电流 / 表面漏电", leakX - 109, subY + 53);

    var roicOuterX = roicX + roicW * 0.05, roicOuterY = pixOuterY + 8, roicOuterW = roicW * 0.90, roicOuterH = pixOuterH + 42;
    ctx.fillStyle = "#eef4f7";
    rr(ctx, roicOuterX, roicOuterY, roicOuterW, roicOuterH, 12);
    ctx.fill();
    ctx.strokeStyle = "#b6c2cc";
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("ROIC 读出电路", roicOuterX + roicOuterW / 2, roicOuterY + 26);

    var fill = clamp(m.snr / 25, 0.05, 1);
    var blockX = roicOuterX + 24, blockW = roicOuterW - 48;
    var capY = roicOuterY + 46, capH = 46;
    ctx.fillStyle = "#ffffff";
    rr(ctx, blockX, capY, blockW, capH, 7);
    ctx.fill();
    ctx.strokeStyle = "#687787";
    ctx.stroke();
    ctx.fillStyle = "#17212b";
    ctx.font = "800 11px sans-serif";
    ctx.fillText("CTIA 积分电容", roicOuterX + roicOuterW / 2, capY + 14);
    ctx.fillStyle = "#edf4f8";
    rr(ctx, blockX + 14, capY + 22, blockW - 28, 12, 4);
    ctx.fill();
    ctx.fillStyle = "#1f6f8b";
    rr(ctx, blockX + 14, capY + 22, (blockW - 28) * fill, 12, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(104,119,135,.40)";
    ctx.lineWidth = 1;
    for (var tick = 0; tick <= 4; tick++) {
      var tx = blockX + 14 + (blockW - 28) * tick / 4;
      ctx.beginPath();
      ctx.moveTo(tx, capY + 20);
      ctx.lineTo(tx, capY + 38);
      ctx.stroke();
    }
    ctx.fillStyle = "#66727f";
    ctx.font = "10px sans-serif";
    ctx.fillText("Iphoto - Idark → Vint", roicOuterX + roicOuterW / 2, capY + 40);

    var chainY = capY + capH + 14, unitGap = 10, unitW = (blockW - unitGap * 2) / 3, unitH = 38;
    [["S/H", "采样保持"], ["AMP", "低噪声放大"], ["ADC", "数字化"]].forEach(function (u, idx) {
      var ux = blockX + idx * (unitW + unitGap);
      ctx.fillStyle = "#ffffff";
      rr(ctx, ux, chainY, unitW, unitH, 7);
      ctx.fill();
      ctx.strokeStyle = "#c7d1da";
      ctx.stroke();
      ctx.fillStyle = idx === 1 ? "#1f6f8b" : "#687787";
      ctx.font = "800 11px sans-serif";
      ctx.fillText(u[0], ux + unitW / 2, chainY + 15);
      ctx.fillStyle = "#66727f";
      ctx.font = "9px sans-serif";
      ctx.fillText(u[1], ux + unitW / 2, chainY + 28);
      ctx.fillStyle = idx === 1 ? "#28c5a7" : "#d99a3d";
      ctx.beginPath();
      ctx.arc(ux + 12, chainY + 12, 3.2, 0, Math.PI * 2);
      ctx.fill();
      if (idx < 2) drawArrow(ctx, ux + unitW + 2, chainY + unitH / 2, ux + unitW + unitGap - 3, chainY + unitH / 2, "#687787", 1.2);
    });

    var gridY = chainY + unitH + 14;
    var mapSize = Math.min(blockW, roicOuterY + roicOuterH - gridY - 36);
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
    ctx.shadowColor = "rgba(57,169,200,.65)";
    ctx.shadowBlur = 9;
    drawArrow(ctx, pixOuterX + pixOuterW + 8, currentY, roicOuterX - 10, currentY, "#39a9c8", 2.2);
    ctx.restore();
    ctx.fillStyle = "#39a9c8";
    ctx.font = "800 10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("光电流", pixOuterX + pixOuterW + 16, currentY - 8);

    var legendY = footerTop + 28;
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

  function drawSignal(m) {
    var o = canvasCtx("signalCanvas"), ctx = o.ctx, W = o.w, H = o.h;
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    var padL = 44, padR = 18, padT = 24, padB = 34, pw = W - padL - padR, ph = H - padT - padB;
    ctx.strokeStyle = "#d9e0e7"; ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(padL + pw, padT + ph); ctx.stroke();
    var sig = clamp(Math.log10(Math.max(m.iph, 1e-14)) + 13, 0, 7) / 7;
    var dark = clamp(Math.log10(Math.max(m.idark, 1e-14)) + 13, 0, 7) / 7;
    function bar(x, val, color, label) {
      var h = val * ph;
      ctx.fillStyle = color; rr(ctx, x, padT + ph - h, pw * .18, h, 5); ctx.fill();
      ctx.fillStyle = "#66727f"; ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.fillText(label, x + pw * .09, padT + ph + 18);
    }
    bar(padL + pw * .18, sig, "#1f6f8b", "photo");
    bar(padL + pw * .48, dark, "#b8322a", "dark");
    ctx.strokeStyle = "#0e7665"; ctx.lineWidth = 2;
    var y = padT + ph * (1 - clamp(m.snr / 25, 0, 1));
    ctx.beginPath(); ctx.moveTo(padL + pw * .05, y); ctx.lineTo(padL + pw * .88, y); ctx.stroke();
    ctx.fillStyle = "#0e7665"; ctx.font = "700 12px sans-serif"; ctx.textAlign = "left"; ctx.fillText("SNR trend " + m.snr.toFixed(1), padL + 4, y - 6);
    ctx.fillStyle = "#17212b"; ctx.fillText("ROIC integrates charge for " + P.tint.toFixed(1) + " ms", padL, 18);
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
    drawBand(m);
    drawSignal(m);
    drawTrend(m);
  }

  function animationLoop(ts) {
    if (latestModel) drawDevice(latestModel, ts);
    requestAnimationFrame(animationLoop);
  }

  function applyPreset(kind) {
    if (kind === "mwir") {
      Object.assign(P, { scene: "plume", inas: 2.43, gasb: 2.42, periods: 100, interfaceNm: 0.09, interfaceQ: 0.90, mBarrier: false, detT: 150, bias: 0.08, xrd: 24, afm: 2.0, passivation: 0.82, targetT: 760, bgT: 300, tint: 2.5 });
    } else if (kind === "bbir") {
      Object.assign(P, { scene: "gas", inas: 5.45, gasb: 3.03, periods: 70, interfaceNm: 0.16, interfaceQ: 0.78, mBarrier: true, detT: 120, bias: 0.06, xrd: 36, afm: 2.6, passivation: 0.76, targetT: 325, bgT: 300, tint: 8.0 });
    } else {
      Object.assign(P, { scene: "human", inas: 5.45, gasb: 3.03, periods: 70, interfaceNm: 0.20, interfaceQ: 0.86, mBarrier: true, detT: 77, bias: 0.05, xrd: 24, afm: 2.0, passivation: 0.80, targetT: 310, bgT: 293, tint: 5 });
    }
    Object.keys(P).forEach(function (k) {
      if (!els[k]) return;
      if (els[k].type === "checkbox") els[k].checked = !!P[k];
      else els[k].value = P[k];
    });
    render();
  }

  function init() {
    ids.forEach(function (id) { els[id] = $(id); });
    Object.keys(P).forEach(function (k) {
      if (!els[k]) return;
      els[k].addEventListener("input", render);
      els[k].addEventListener("change", render);
    });
    els.presetLwir.addEventListener("click", function () { applyPreset("lwir"); });
    els.presetMwir.addEventListener("click", function () { applyPreset("mwir"); });
    els.presetBbir.addEventListener("click", function () { applyPreset("bbir"); });
    window.addEventListener("resize", render);
    render();
    requestAnimationFrame(animationLoop);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
