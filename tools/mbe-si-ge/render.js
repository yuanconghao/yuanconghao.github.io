/*
 * Si/Ge 炉子 · 领域示意图 (render.js)
 * 生长腔(Si/Ge 电子束 + B 热源) · Si/SiGe 周期堆叠 · Ge 组分/带隙。
 * 通用图表（应变 / RHEED 曲线 / 相屏）与画布管理由 engine.js 提供。
 */
(function (M) {
  "use strict";
  var clamp = M.clamp, lerp = M.lerp, C = M.C, COL = M.COL, rr = M.rr;

  function bufIndexOf(st) { for (var i = 0; i < st.steps.length; i++) if (st.steps[i].stage === "buffer") return i; return -1; }

  function drawSource(cx, x, y, w, h, key, open) {
    var cell = M.CELL[key], col = cell.col;
    cx.fillStyle = open ? "rgba(15,122,104,.08)" : "#f8fafc";
    cx.strokeStyle = open ? col : "#e2e8f0"; cx.lineWidth = open ? 1.8 : 1;
    rr(cx, x, y, w, h, 6); cx.fill(); cx.stroke();
    cx.fillStyle = col; cx.font = "700 12px sans-serif"; cx.textAlign = "center";
    cx.fillText(key, x + w / 2, y + 14);
    cx.fillStyle = "#475569"; cx.font = "8px sans-serif"; cx.textAlign = "left";
    cx.fillText(cell.kind === "ebeam" ? "e-beam" : "thermal", x + 6, y + 27);
    cx.fillText(cell.temp.toFixed(0) + cell.unit, x + 6, y + 39);
    cx.fillStyle = open ? C.good : "#94a3b8"; cx.font = "700 10px sans-serif"; cx.textAlign = "right";
    cx.fillText(open ? "ON" : "OFF", x + w - 6, y + 34);
  }
  function drawBeam(cx, x0, y0, x1, y1, col, open) {
    cx.save(); cx.strokeStyle = open ? col : "rgba(194,202,211,.35)"; cx.globalAlpha = open ? 0.55 : 0.16; cx.lineWidth = open ? 6 : 2;
    cx.beginPath(); cx.moveTo(x0, y0); cx.lineTo(x1, y1); cx.stroke(); cx.restore();
  }
  function drawFallingAtoms(cx, mx, pw, frontY, mat) {
    var atoms = M.atoms, st = M.st;
    while (atoms.length < 9) atoms.push({ x: mx + Math.random() * pw, y: 75 + Math.random() * 25, v: 1 + Math.random() * 2 });
    cx.fillStyle = COL[mat] || "#888";
    for (var i = atoms.length - 1; i >= 0; i--) { var at = atoms[i]; at.y += at.v * (2 + st.speedMul); cx.beginPath(); cx.arc(at.x, at.y, 2.4, 0, 7); cx.fill(); if (at.y >= frontY - 2) atoms.splice(i, 1); }
  }

  function drawChamber(Q) {
    var st = M.st, cx = M.cx, W = M.dims.chW, H = M.dims.chH, sh = M.shutters, keys = M.ELEMENTS;
    cx.clearRect(0, 0, W, H); cx.fillStyle = C.canvasBg; cx.fillRect(0, 0, W, H);
    var mx = clamp(W * 0.04, 18, 40), pw = W - mx * 2, sourceY = 8, sourceH = 50, gap = clamp(W * 0.02, 10, 20);
    var sourceW = (pw - gap * (keys.length - 1)) / keys.length, centers = [];
    for (var k = 0; k < keys.length; k++) { var sx0 = mx + k * (sourceW + gap); centers.push(sx0 + sourceW / 2); drawSource(cx, sx0, sourceY, sourceW, sourceH, keys[k], sh[keys[k]]); }

    var groundY = H - 34, topPad = sourceY + sourceH + 16, maxPx = groundY - topPad, cur = M.curStep();
    var bi = bufIndexOf(st), bufNm = bi >= 0 ? st.steps[bi].nm : 0, bufGrown = 0;
    if (bi >= 0) { if (st.si > bi) bufGrown = bufNm; else if (st.si === bi) bufGrown = bufNm * st.sp; }
    var bufMaxH = Math.min(maxPx * 0.24, 80), bufH = bufGrown > 0 ? Math.max(5, bufMaxH * clamp(bufGrown / Math.max(bufNm, 1), 0, 1)) : 0;
    var filmBaseY = groundY - bufH, upperNm = 0;
    for (var ui = bi + 1; ui <= st.si && ui < st.steps.length; ui++) { var us = st.steps[ui]; upperNm += (ui < st.si) ? us.nm : us.nm * st.sp; }
    var viewNm = 60, avail = Math.max(44, filmBaseY - topPad), pxNm = avail / viewNm, viewBot = Math.max(0, upperNm - viewNm);
    var frontY = cur.stage === "buffer" ? filmBaseY : filmBaseY - Math.min(upperNm - viewBot, viewNm) * pxNm;
    for (var b = 0; b < keys.length; b++) drawBeam(cx, centers[b], sourceY + sourceH, W / 2 + (b - (keys.length - 1) / 2) * 24, frontY, M.CELL[keys[b]].col, sh[keys[b]]);

    cx.fillStyle = "rgba(23,33,43,.04)"; cx.fillRect(0, topPad - 8, W, groundY - topPad + 28);
    cx.strokeStyle = "#d4dae1"; cx.beginPath(); cx.moveTo(0, topPad - 8); cx.lineTo(W, topPad - 8); cx.stroke();

    cx.fillStyle = C.subFill; cx.fillRect(0, groundY, W, 20);
    cx.fillStyle = C.subText; cx.font = "10px sans-serif"; cx.textAlign = "center"; cx.fillText("Si(001) substrate", W / 2, groundY + 14);

    if (st && st.oxideOpacity > 0) {
      cx.save(); cx.globalAlpha = st.oxideOpacity; cx.fillStyle = "#a89078"; cx.fillRect(0, groundY - 4, W, 4);
      cx.fillStyle = "#8a705a"; cx.font = "italic 9px sans-serif"; cx.textAlign = "center";
      cx.fillText("Native SiO2 desorbing (" + (st.oxideOpacity * 100).toFixed(0) + "% remaining)", W / 2, groundY - 9); cx.restore();
    }
    if (bufH > 0) {
      cx.fillStyle = COL.BUF; cx.fillRect(0, filmBaseY, W, bufH);
      cx.fillStyle = "#ffffff"; cx.font = "700 10px sans-serif"; cx.textAlign = "center";
      cx.fillText("Si buffer 100 nm", W / 2, filmBaseY + Math.min(bufH - 5, 14));
    }
    var cumul = 0;
    for (var i = bi + 1; i <= st.si && i < st.steps.length; i++) {
      var s = st.steps[i], bandNm = (i < st.si) ? s.nm : s.nm * st.sp, c0 = cumul, c1 = cumul + bandNm; cumul = c1;
      if (bandNm <= 0 || c1 <= viewBot) continue;
      var lo = Math.max(c0, viewBot), yB = filmBaseY - (lo - viewBot) * pxNm, yT = filmBaseY - (c1 - viewBot) * pxNm;
      cx.fillStyle = COL[s.mat] || C.BUF; cx.fillRect(0, yT, W, Math.max(0.7, yB - yT));
      if (Q < 0.72 && s.mat && i === st.si) { cx.fillStyle = "rgba(192,57,43,.18)"; for (var rr2 = 0; rr2 < Math.round((1 - Q) * 14); rr2++) { var rx = (rr2 * 37) % W; cx.fillRect(rx, yT - (1 - Q) * 5, 10, (1 - Q) * 5); } }
    }
    if (!st.done && cur.mat && st.sp < 1) drawFallingAtoms(cx, 0, W, frontY, cur.mat);
    if (Math.abs(st.accStrain) > M.STRAIN_CRIT * 0.7) {
      cx.strokeStyle = "rgba(192,57,43,.75)"; cx.lineWidth = 1.6;
      for (var d = 0; d < 4; d++) { var dx = W * (0.18 + d * 0.2); cx.beginPath(); cx.moveTo(dx, groundY - 8); cx.lineTo(dx + 18, groundY - 80 - d * 8); cx.stroke(); }
    }
    cx.fillStyle = C.axisText; cx.font = "9px sans-serif"; cx.textAlign = "right"; cx.fillText("SL scale 10nm · buffer compressed", W - 8, groundY - 10);
  }

  function drawStackMap() {
    if (!M.mcx) return;
    var st = M.st, mcx = M.mcx, W = M.dims.mpW, H = M.dims.mpH, n = Math.max(1, M.P.nPer);
    mcx.clearRect(0, 0, W, H); mcx.fillStyle = C.canvasBg; mcx.fillRect(0, 0, W, H);
    var padT = 8, padB = 18, ph = H - padT - padB, cur = M.curPeriod();
    var preW = Math.min(Math.max(70, W * .18), 120), subW = preW * .38, bufW = preW - subW;
    var bi = bufIndexOf(st), bufDone = bi >= 0 && st.si >= bi;
    var cycleX0 = preW, cycleW = Math.max(1, W - preW), stepW = cycleW / n;
    mcx.fillStyle = COL.SUB; rr(mcx, 0, padT, subW, ph, 4); mcx.fill();
    mcx.fillStyle = bufDone ? COL.BUF : "#d9e1ea"; rr(mcx, subW, padT, bufW, ph, 4); mcx.fill();
    mcx.fillStyle = "#fff"; mcx.font = "8px sans-serif"; mcx.textAlign = "center";
    mcx.fillText("Si sub", subW / 2, padT + ph / 2 + 3); mcx.fillText("Si buf", subW + bufW / 2, padT + ph / 2 + 3);
    var gap = stepW > 4 ? Math.min(1.5, stepW * .12) : 0;
    for (var p = 0; p < n; p++) {
      var x = cycleX0 + p * stepW, done = M.stageOf() === "sl" ? p + 1 < cur : M.stageIdx(M.stageOf()) > M.stageIdx("sl"), active = M.stageOf() === "sl" && p + 1 === cur;
      var ix = x + gap, iw = Math.max(1, stepW - gap * 2), sigeW = iw * .55, siW = Math.max(1, iw - sigeW);
      mcx.globalAlpha = done ? 0.92 : active ? 1 : 0.26;
      mcx.fillStyle = COL.SiGe; mcx.fillRect(ix, padT, sigeW, ph);
      mcx.fillStyle = COL.Si; mcx.fillRect(ix + sigeW, padT, siW, ph);
      if (active) { mcx.globalAlpha = 1; mcx.strokeStyle = C.bad; mcx.lineWidth = 2; mcx.strokeRect(x - 1, padT - 1, Math.max(3, stepW + 2), ph + 2); }
    }
    mcx.globalAlpha = 1; mcx.fillStyle = C.axisText; mcx.font = "9px sans-serif"; mcx.textAlign = "center";
    mcx.fillText("左→右：Si 衬底 / Si buffer / 重复 (SiGe 阱 + Si 隔垒) 周期", W / 2, H - 4);
  }

  function drawBand() {
    if (!M.bcx) return;
    var bx = M.bcx, W = M.dims.bdW, H = M.dims.bdH, x = M.geFraction(), eg = M.sigeBandgap(x);
    bx.clearRect(0, 0, W, H); bx.fillStyle = C.canvasBg; bx.fillRect(0, 0, W, H);
    var mid = H * .54, left = 24, right = W - 24;
    // 导带/价带：SiGe 阱（压应变 → 价带抬升、阱在价带侧）
    var wellX0 = W * .40, wellX1 = W * .60, dGe = x * 60;
    bx.strokeStyle = COL.Si; bx.lineWidth = 2;
    bx.beginPath(); bx.moveTo(left, mid - 40); bx.lineTo(wellX0, mid - 40); bx.lineTo(wellX0, mid - 40 + dGe * .3); bx.lineTo(wellX1, mid - 40 + dGe * .3); bx.lineTo(wellX1, mid - 40); bx.lineTo(right, mid - 40); bx.stroke();
    bx.strokeStyle = COL.Ge;
    bx.beginPath(); bx.moveTo(left, mid + 40); bx.lineTo(wellX0, mid + 40); bx.lineTo(wellX0, mid + 40 + dGe); bx.lineTo(wellX1, mid + 40 + dGe); bx.lineTo(wellX1, mid + 40); bx.lineTo(right, mid + 40); bx.stroke();
    bx.fillStyle = "rgba(140,179,105,.18)"; bx.fillRect(wellX0, 26, wellX1 - wellX0, H - 56);
    bx.fillStyle = C.axisText; bx.font = "10px sans-serif"; bx.textAlign = "center";
    bx.fillText("Si", W * .2, H - 16); bx.fillText("SiGe 阱 (x=" + x.toFixed(2) + ")", (wellX0 + wellX1) / 2, H - 16); bx.fillText("Si", W * .8, H - 16);
    bx.textAlign = "left"; bx.font = "700 12px sans-serif"; bx.fillStyle = C.label;
    bx.fillText("Eg(SiGe) " + eg.toFixed(3) + " eV", 24, 18);
    bx.font = "10px sans-serif"; bx.fillText("失配 " + (4.18 * x).toFixed(2) + "% · 压应变 → 价带偏移、空穴限域", 24, H - 4);
  }

  M.drawChamber = drawChamber; M.drawStackMap = drawStackMap; M.drawBand = drawBand;
})(window.MBE);
