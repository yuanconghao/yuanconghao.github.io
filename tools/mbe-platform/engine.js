/*
 * MBE 工艺训练台 · 通用引擎 (mbe-platform/engine.js)
 * 与具体炉子无关的机制：命名空间、工具函数、画布注册/自适应、通用图表
 * （应变曲线 / RHEED 振荡曲线 / RHEED 相屏）、表单工具。
 *
 * 一个炉子页面的加载顺序：
 *   platform/engine.js  →  <furnace>/core.js → render.js → app.js
 * 引擎在此创建 window.MBE；炉子各文件以 (function(M){...})(window.MBE) 往上挂载
 * 领域模型（CELL/COL/recipe/物理）、领域示意图与读数、控件绑定。
 */
window.MBE = {};
window.MBE.needsRedraw = true;
(function (M) {
  "use strict";

  // ---------- 通用 UI 调色板（炉子的材料色 COL 各自定义）----------
  M.C = {
    canvasBg: "#ffffff", subFill: "#e3ead9", subText: "#5f6b78",
    label: "#5f6b78", beam: "#c2cad3", front: "rgba(23,33,43,.28)",
    axis: "#d4dae1", axisText: "#5f6b78", line: "#245f9d",
    good: "#0f7a68", warn: "#b97818", bad: "#c0392b",
    zone: "rgba(192,57,43,.08)", screenBg: "#06180f"
  };

  // ---------- 通用状态/引用占位 ----------
  M.st = null; M.atoms = []; M.dims = {};
  M._canvases = [];

  // ---------- 工具函数 ----------
  M.$ = function (id) { return document.getElementById(id); };
  M.lerp = function (a, b, t) { return a + (b - a) * t; };
  M.clamp = function (v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; };
  M.fmtHMS = function (sec) {
    sec = Math.max(0, Math.round(sec));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  };
  M.setText = function (id, txt) { var el = M.$(id); if (el) el.textContent = txt; };
  M.setBadge = function (id, o) { var el = M.$(id); if (el) { el.textContent = o.txt; el.className = "v badge " + o.cls; } };
  M.clsBadge = function (id, txt, cls) { M.setBadge(id, { txt: txt, cls: cls }); };
  M.pct = function (v) { return Math.round(v * 100) + "%"; };
  M.rr = function (c, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r); c.closePath();
  };

  // ---------- 画布注册 / 自适应 ----------
  function setupCanvas(c, ctx) {
    var dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
    var w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    c.width = w * dpr; c.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: w, h: h };
  }
  // 注册一块画布：id=元素id，ctxKey=挂到 M 的 context 名(如 "cx")，wKey/hKey=dims 宽高键
  M.registerCanvas = function (id, ctxKey, wKey, hKey) {
    var el = M.$(id); if (!el) return null;
    var ctx = el.getContext("2d");
    M[ctxKey] = ctx;
    M._canvases.push({ el: el, ctx: ctx, wKey: wKey, hKey: hKey });
    return ctx;
  };
  M.resizeAll = function () {
    for (var i = 0; i < M._canvases.length; i++) {
      var c = M._canvases[i], d = setupCanvas(c.el, c.ctx);
      M.dims[c.wKey] = d.w; M.dims[c.hKey] = d.h;
    }
    M.needsRedraw = true;
  };
  M.setupCanvas = setupCanvas;

  // ---------- 通用图表：应变累积 ----------
  // 需炉子提供：M.STRAIN_CRIT、M.strainStatus(accStrain)、M.sx、M.dims.stW/stH
  M.drawStrain = function () {
    var st = M.st, sx = M.sx, C = M.C, CRIT = M.STRAIN_CRIT, W = M.dims.stW, H = M.dims.stH;
    if (!sx) return;
    sx.clearRect(0, 0, W, H); sx.fillStyle = C.canvasBg; sx.fillRect(0, 0, W, H);
    var padL = 46, padR = 10, padT = 14, padB = 18, pw = W - padL - padR, ph = H - padT - padB;
    var ymax = CRIT * 1.35; function Y(s) { return padT + (1 - (s + ymax) / (2 * ymax)) * ph; }
    sx.fillStyle = C.zone; sx.fillRect(padL, padT, pw, Y(CRIT) - padT); sx.fillRect(padL, Y(-CRIT), pw, padT + ph - Y(-CRIT));
    sx.fillStyle = "rgba(15,122,104,.055)"; sx.fillRect(padL, Y(CRIT), pw, Y(-CRIT) - Y(CRIT));
    sx.strokeStyle = C.axis; sx.beginPath(); sx.moveTo(padL, Y(0)); sx.lineTo(padL + pw, Y(0)); sx.stroke();
    sx.strokeStyle = "rgba(192,57,43,.45)"; sx.setLineDash([4, 3]);
    sx.beginPath(); sx.moveTo(padL, Y(CRIT)); sx.lineTo(padL + pw, Y(CRIT)); sx.stroke();
    sx.beginPath(); sx.moveTo(padL, Y(-CRIT)); sx.lineTo(padL + pw, Y(-CRIT)); sx.stroke(); sx.setLineDash([]);
    sx.fillStyle = C.axisText; sx.font = "700 9px sans-serif"; sx.textAlign = "right";
    sx.fillText("拉(+)", padL - 6, Y(CRIT * .7));
    sx.fillText("0", padL - 6, Y(0) + 3);
    sx.fillText("压(-)", padL - 6, Y(-CRIT * .7));
    sx.textAlign = "left"; sx.font = "9px sans-serif";
    sx.fillText("+临界", padL + 4, Y(CRIT) - 4);
    sx.fillText("-临界", padL + 4, Y(-CRIT) + 10);
    var data = st ? st.strainHist : []; if (data.length < 2) return;
    var tMax = Math.max(data[data.length - 1].t, .5), stat = M.strainStatus(st.accStrain);
    sx.strokeStyle = stat.cls === "good" ? C.good : stat.cls === "warn" ? C.warn : C.bad; sx.lineWidth = 1.7; sx.beginPath();
    for (var i = 0; i < data.length; i++) { var px = padL + (data[i].t / tMax) * pw, py = Y(M.clamp(data[i].s, -ymax, ymax)); if (i === 0) sx.moveTo(px, py); else sx.lineTo(px, py); }
    sx.stroke();
  };

  // ---------- 通用图表：RHEED 强度振荡曲线 ----------
  // 需炉子提供：M.rheedC/M.rx、M.dims.rhW/rhH、st.rheedHist = [{t, v, sensors:[...]}]
  M.drawRheed = function () {
    if (!M.rx) return;
    var st = M.st, rx = M.rx, C = M.C, W = M.dims.rhW, H = M.dims.rhH;
    rx.clearRect(0, 0, W, H); rx.fillStyle = C.canvasBg; rx.fillRect(0, 0, W, H);
    var padL = 42, padR = 18, padT = 18, padB = 30, pw = W - padL - padR, ph = H - padT - padB;
    rx.strokeStyle = "#f1f5f9"; rx.lineWidth = 1;
    for (var val = 0.1; val <= 1.0; val += 0.1) { var gy = padT + (1 - val) * ph; rx.beginPath(); rx.moveTo(padL, gy); rx.lineTo(padL + pw, gy); rx.stroke(); }
    rx.strokeStyle = C.axis; rx.lineWidth = 1.2;
    rx.beginPath(); rx.moveTo(padL, padT); rx.lineTo(padL, padT + ph); rx.lineTo(padL + pw, padT + ph); rx.stroke();
    rx.fillStyle = C.axisText; rx.font = "9px sans-serif"; rx.textAlign = "right";
    rx.fillText("1.0", padL - 6, padT + 4); rx.fillText("0.5", padL - 6, padT + ph / 2 + 3); rx.fillText("0", padL - 6, padT + ph + 3);
    var data = st ? st.rheedHist : []; if (data.length < 2) return;
    var tLast = data[data.length - 1].t, zoomSec = M.rheedZoomSec || 0;
    var tMin = 0, tMax = Math.max(st.totalReal || tLast, tLast, 1), windowSec = tMax - tMin;
    if (zoomSec > 0) {
      windowSec = Math.max(5, zoomSec);
      tMax = Math.max(tLast, windowSec);
      tMin = Math.max(0, tMax - windowSec);
    }
    var pts = []; for (var i = 0; i < data.length; i++) if (data[i].t >= tMin && data[i].t <= tMax) pts.push(data[i]);
    if (pts.length < 2) return;
    var colors = ["#f97316", "#facc15", "#b45309", "#ec4899"];
    function sensorValue(pt, idx) {
      if (pt.sensors && pt.sensors[idx] != null) return pt.sensors[idx];
      var offsets = [0.18, 0.52, 0.66, 0.78], gains = [0.16, 0.18, 0.17, 0.15];
      return M.clamp(offsets[idx] + gains[idx] * (pt.v || 0), 0, 1);
    }
    for (var s = 0; s < 4; s++) {
      rx.strokeStyle = colors[s];
      rx.lineWidth = s === 0 ? 1.8 : 1.5;
      rx.beginPath();
      for (var j = 0; j < pts.length; j++) {
        var px = padL + ((pts[j].t - tMin) / Math.max(1, windowSec)) * pw;
        var py = padT + (1 - sensorValue(pts[j], s)) * ph;
        if (j === 0) rx.moveTo(px, py); else rx.lineTo(px, py);
      }
      rx.stroke();
    }
    rx.fillStyle = C.axisText; rx.textAlign = "center"; rx.font = "9px sans-serif";
    var tickStep = windowSec <= 40 ? 5 : windowSec <= 120 ? 15 : windowSec <= 600 ? 60 : Math.ceil(windowSec / 8 / 60) * 60;
    for (var sec = Math.ceil(tMin / tickStep) * tickStep; sec <= tMax; sec += tickStep) {
      var sxp = padL + ((sec - tMin) / Math.max(1, windowSec)) * pw;
      if (sxp >= padL && sxp <= padL + pw) rx.fillText(Math.round(sec) + "s", sxp, padT + ph + 16);
    }
    rx.textAlign = "left"; rx.font = "700 10px sans-serif";
    rx.fillText(zoomSec > 0 ? "Zoom: last " + Math.round(windowSec) + " s" : "Full growth timeline", padL, 12);
    for (var k = 0; k < 4; k++) {
      var lx = W - 184 + k * 44;
      rx.fillStyle = colors[k]; rx.fillRect(lx, 6, 10, 10);
      rx.fillStyle = C.axisText; rx.fillText("S" + (k + 1), lx + 14, 15);
    }
  };

  // ---------- 通用图表：RHEED 绿色相屏（条纹↔斑点 + 脱氧漫散射）----------
  // 需炉子提供：M.scx、M.dims.scW/scH、M.rheedState(Q)、M.rheedClean(ph)、M.stageOf()、M.curStep()
  // 约定：脱氧阶段 stage 包含 "load"/"deoxide"，st.oxideOpacity 控制漫散射
  M.drawScreen = function (Q) {
    var scx = M.scx; if (!scx) return;
    var C = M.C, W = M.dims.scW, H = M.dims.scH, rs = M.rheedState(Q), st = M.st, cur = M.curStep(), clamp = M.clamp, lerp = M.lerp, rr = M.rr;
    var now = (window.performance && window.performance.now ? window.performance.now() : Date.now()) / 1000;
    var dep = M.currentDeposition ? M.currentDeposition(cur) : { mat: cur.mat };
    var growing = !!dep.mat;
    var phase = st ? st.phase + now * (growing ? 0.20 : 0.04) : now * 0.04;
    var osc = growing ? (0.72 + 0.28 * M.rheedClean(phase)) : 0.38;
    var brightness = clamp((growing ? osc : 0.55), 0, 1);
    var oxideOp = (st && st.oxideOpacity != null) ? st.oxideOpacity : 0;
    var stg = M.stageOf();
    var isDeoxideStage = st && (stg === "load" || stg === "deoxide");
    var shapeQ = isDeoxideStage ? clamp(Q * (1 - oxideOp), 0, 1) : Q;
    scx.clearRect(0, 0, W, H); scx.fillStyle = C.screenBg; scx.fillRect(0, 0, W, H);
    var g = scx.createRadialGradient(W / 2, H * .5, 10, W / 2, H * .5, W * .65);
    var diffuseIntensity = isDeoxideStage ? (0.22 + 0.38 * oxideOp + 0.32 * shapeQ * brightness) : (0.22 + 0.32 * Q * brightness);
    g.addColorStop(0, "rgba(40,120,70," + clamp(diffuseIntensity, 0, 1).toFixed(2) + ")");
    g.addColorStop(1, "rgba(8,30,18,.2)");
    scx.fillStyle = g; scx.fillRect(0, 0, W, H);
    var y0 = H * .52, n = 5, sp = W / (n + 1), streakH = H * .58, spotR = Math.max(3, W * .02);
    var streakAlphaFactor = isDeoxideStage ? (1 - oxideOp) : 1;
    if (streakAlphaFactor > 0.05) {
      for (var i = 0; i < n; i++) {
        var drift = Math.sin(now * 2.3 + i * 1.7) * Math.max(0, 0.72 - shapeQ) * W * .018;
        var x = sp * (i + 1) + drift, spec = (i === (n - 1) / 2);
        var pulse = 0.82 + 0.18 * Math.sin(2 * Math.PI * phase + i * .55);
        var hgt = lerp(spotR * 1.4, streakH, Math.pow(shapeQ, 1.1));
        var al = (spec ? .95 : .7) * (.35 + .65 * Q) * pulse * brightness * streakAlphaFactor;
        var gr = scx.createLinearGradient(0, y0 - hgt / 2, 0, y0 + hgt / 2);
        gr.addColorStop(0, "rgba(170,240,190,0)"); gr.addColorStop(.5, "rgba(200,255,210," + clamp(al, 0, 1) + ")"); gr.addColorStop(1, "rgba(170,240,190,0)");
        scx.fillStyle = gr; rr(scx, x - spotR / 2, y0 - hgt / 2, spotR, hgt, spotR / 2); scx.fill();
        if (shapeQ < 0.70) {
          scx.fillStyle = "rgba(210,255,220," + clamp(.85 * (1 - shapeQ) * brightness * (0.75 + 0.25 * Math.sin(now * 5 + i)) * streakAlphaFactor, 0, 1) + ")";
          for (var d = -1; d <= 1; d++) { scx.beginPath(); scx.arc(x + Math.sin(now * 3 + d + i) * 3, y0 + d * H * .17, spotR * (shapeQ < .35 ? 1.15 : .7), 0, 7); scx.fill(); }
        }
      }
      if (shapeQ < 0.30) {
        scx.strokeStyle = "rgba(190,255,205," + (0.42 * streakAlphaFactor).toFixed(2) + ")"; scx.lineWidth = 2;
        for (var r = 0; r < 3; r++) { scx.beginPath(); scx.ellipse(W / 2, y0, W * (.18 + r * .10), H * (.10 + r * .06), 0, 0, 7); scx.stroke(); }
      }
    }
    if (isDeoxideStage && oxideOp > 0.05) {
      var haloG = scx.createRadialGradient(W / 2, y0, 10, W / 2, y0, W * 0.42);
      haloG.addColorStop(0, "rgba(180, 240, 205, " + (0.42 * oxideOp * (0.85 + 0.15 * Math.sin(now * 8))).toFixed(3) + ")");
      haloG.addColorStop(0.5, "rgba(100, 180, 120, " + (0.15 * oxideOp).toFixed(3) + ")");
      haloG.addColorStop(1, "rgba(8, 30, 18, 0)");
      scx.fillStyle = haloG; scx.beginPath(); scx.arc(W / 2, y0, W * 0.42, 0, 7); scx.fill();
    }
    if (growing) {
      var scanX = (now * 38) % W;
      scx.fillStyle = "rgba(220,255,230,.10)"; scx.fillRect(scanX, 0, 2, H);
      scx.fillStyle = "rgba(220,255,230,.72)"; scx.font = "9px sans-serif"; scx.textAlign = "left";
      scx.fillText("oscillation phase " + (st.phase % 1).toFixed(2), 8, 16);
    } else {
      scx.fillStyle = "rgba(220,255,230,.60)"; scx.font = "9px sans-serif"; scx.textAlign = "left";
      scx.fillText(isDeoxideStage ? "substrate thermal cleaning / deoxidizing" : "thermal / shutter calibration", 8, 16);
    }
    scx.fillStyle = "rgba(220,255,230,.72)"; scx.font = "700 10px sans-serif"; scx.textAlign = "center";
    var rheedText = rs.txt, rheedDesc = rs.desc;
    if (isDeoxideStage && oxideOp > 0.05) {
      rheedText = oxideOp > 0.85 ? "Amorphous Halo" : oxideOp > 0.3 ? "Weak Halo & Spotty" : "Reconstructing Streaks";
      rheedDesc = oxideOp > 0.85 ? "非晶态氧化层漫散射环，衬底表面无结构重建" : oxideOp > 0.3 ? "氧化层开始分解，微弱重建斑点出现" : "脱氧接近完成，晶态重构长条纹显露";
    }
    var sensorColors = ["#f97316", "#facc15", "#b45309", "#ec4899"];
    var boxes = [
      { x: 0.12, y: 0.50, label: "1" },
      { x: 0.80, y: 0.50, label: "2" },
      { x: 0.48, y: 0.49, label: "3" },
      { x: 0.52, y: 0.18, label: "4" }
    ];
    scx.font = "700 10px sans-serif";
    scx.textAlign = "left";
    for (var bi = 0; bi < boxes.length; bi++) {
      var bw = Math.max(34, W * 0.075), bh = Math.max(30, H * 0.12), bx = boxes[bi].x * W, by = boxes[bi].y * H;
      scx.strokeStyle = sensorColors[bi];
      scx.lineWidth = 1.6;
      scx.strokeRect(bx, by, bw, bh);
      scx.fillStyle = sensorColors[bi];
      scx.fillText(boxes[bi].label, bx + 3, by + 11);
    }
    scx.fillStyle = "rgba(220,255,230,.72)";
    scx.font = "700 10px sans-serif";
    scx.textAlign = "center";
    scx.fillText("RHEED: " + rheedText, W / 2, H - 20);
    scx.font = "9px sans-serif"; scx.fillText(rheedDesc, W / 2, H - 7);
  };

  // ---------- 通用滑块绑定 ----------
  M.bindSlider = function (id, key, fmt, onChange) {
    var el = M.$(id), out = M.$(id + "V");
    if (!el) return;
    el.addEventListener("input", function () {
      M.P[key] = parseFloat(el.value);
      if (out) out.textContent = fmt ? fmt(M.P[key]) : M.P[key];
      if (onChange) onChange();
      M.needsRedraw = true;
      if (M.updateReadout) M.updateReadout();
    });
  };

})(window.MBE);
