/*
 * InAs/GaSb 炉子 · 领域示意图 (render.js)
 * 生长腔炉源/截面层状生长 · 周期堆叠总览 · 快门时序 · II 类超晶格能带。
 * 通用图表（应变/RHEED 曲线/相屏）与画布管理由 engine.js 提供。
 */
(function (M) {
  "use strict";
  var clamp = M.clamp, lerp = M.lerp, C = M.C, COL = M.COL, rr = M.rr;

  function getBufferIndex(st) {
    for (var i = 0; i < st.steps.length; i++) if (st.steps[i].stage === "buffer") return i;
    return -1;
  }

  function drawSource(cx, x, y, w, h, key, open) {
    var cell = M.CELL[key], col = cell.col;
    cx.fillStyle = open ? "rgba(15,122,104,.08)" : "#f8fafc";
    cx.strokeStyle = open ? col : "#e2e8f0"; cx.lineWidth = open ? 1.8 : 1;
    rr(cx, x, y, w, h, 6); cx.fill(); cx.stroke();

    // Row 1: Element Symbol Centered
    cx.fillStyle = col; cx.font = "700 12px sans-serif"; cx.textAlign = "center";
    cx.fillText(key, x + w / 2, y + 14);

    // Column 1 (Rows 2 & 3): Temp & BEP Left-aligned
    cx.fillStyle = "#475569"; cx.font = "9px sans-serif"; cx.textAlign = "left";
    var tempStr = cell.temp.toFixed(0) + " ℃";
    cx.fillText(tempStr, x + 8, y + 28);
    cx.fillText(cell.bep + " Torr", x + 8, y + 41);

    // Column 2 (Rows 2 & 3 Merged): Shutter ON/OFF Right-aligned
    cx.fillStyle = open ? C.good : "#94a3b8"; cx.font = "700 11px sans-serif"; cx.textAlign = "right";
    cx.fillText(open ? "ON" : "OFF", x + w - 8, y + 35);
  }

  function drawBeam(cx, x0, y0, x1, y1, col, open) {
    cx.save();
    cx.strokeStyle = open ? col : "rgba(194,202,211,.35)";
    cx.globalAlpha = open ? 0.58 : 0.18;
    cx.lineWidth = open ? 6 : 2;
    cx.beginPath(); cx.moveTo(x0, y0); cx.lineTo(x1, y1); cx.stroke();
    cx.restore();
  }

  function drawFallingAtoms(cx, mx, pw, frontY, mat) {
    var atoms = M.atoms, st = M.st;
    while (atoms.length < 9) atoms.push({ x: mx + Math.random() * pw, y: 75 + Math.random() * 25, v: 1 + Math.random() * 2 });
    cx.fillStyle = COL[mat] || "#888";
    for (var i = atoms.length - 1; i >= 0; i--) {
      var at = atoms[i]; at.y += at.v * (2 + st.speedMul);
      cx.beginPath(); cx.arc(at.x, at.y, 2.4, 0, 7); cx.fill();
      if (at.y >= frontY - 2) atoms.splice(i, 1);
    }
  }

  function layerColor(s) {
    if (!s || !s.mat) return C.BUF;
    if (M.interfaceMetrics().gaAsRisk > 0.36 && s.mat === "InSb") return COL.GaAs;
    return COL[s.mat] || C.BUF;
  }

  function drawChamber(Q) {
    var st = M.st, cx = M.cx, W = M.dims.chW, H = M.dims.chH, sh = M.shutters, chem = M.shutterChemistry();
    cx.clearRect(0, 0, W, H); cx.fillStyle = C.canvasBg; cx.fillRect(0, 0, W, H);

    var mx = clamp(W * 0.025, 16, 28), pw = W - mx * 2, sourceY = 8, sourceH = 54, gap = clamp(W * 0.012, 8, 14);
    var keys = ["In", "Ga", "Al", "As", "Sb"];
    var sourceW = (pw - gap * (keys.length - 1)) / keys.length, centers = [];
    for (var k = 0; k < keys.length; k++) {
      var sx0 = mx + k * (sourceW + gap);
      centers.push(sx0 + sourceW / 2);
      drawSource(cx, sx0, sourceY, sourceW, sourceH, keys[k], sh[keys[k]]);
    }

    var groundY = H - 34, topPad = sourceY + sourceH + 16, maxPx = groundY - topPad, cur = M.curStep();
    var bufIndex = getBufferIndex(st), bufferNm = bufIndex >= 0 ? st.steps[bufIndex].nm : 0;
    var bufferGrown = 0;
    if (bufIndex >= 0) {
      if (st.si > bufIndex) bufferGrown = bufferNm;
      else if (st.si === bufIndex) bufferGrown = bufferNm * st.sp;
    }
    var bufferMaxH = Math.min(maxPx * 0.26, 88);
    var bufferH = bufferGrown > 0 ? Math.max(5, bufferMaxH * clamp(bufferGrown / Math.max(bufferNm, 1), 0, 1)) : 0;
    var filmBaseY = groundY - bufferH;
    var upperNm = 0;
    for (var ui = bufIndex + 1; ui <= st.si && ui < st.steps.length; ui++) {
      var us = st.steps[ui];
      upperNm += (ui < st.si) ? us.nm : us.nm * st.sp;
    }
    var upperViewNm = 62, upperAvail = Math.max(44, filmBaseY - topPad), pxNm = upperAvail / upperViewNm;
    var upperViewBot = Math.max(0, upperNm - upperViewNm);
    var frontY = cur.stage === "buffer" ? filmBaseY : filmBaseY - Math.min(upperNm - upperViewBot, upperViewNm) * pxNm;
    var targetX = W / 2, targetY = frontY;
    for (var b = 0; b < keys.length; b++) drawBeam(cx, centers[b], sourceY + sourceH, targetX + (b - (keys.length - 1) / 2) * 22, targetY, M.CELL[keys[b]].col, sh[keys[b]]);

    // Chamber casing background and border separation
    cx.fillStyle = "rgba(23,33,43,.04)";
    cx.fillRect(0, topPad - 8, W, groundY - topPad + 28);
    cx.strokeStyle = "#d4dae1";
    cx.beginPath();
    cx.moveTo(0, topPad - 8);
    cx.lineTo(W, topPad - 8);
    cx.stroke();

    // Substrate aligned with left/right of canvas
    cx.fillStyle = C.subFill; cx.fillRect(0, groundY, W, 20);
    cx.fillStyle = C.subText; cx.font = "10px sans-serif"; cx.textAlign = "center";
    cx.fillText("GaSb substrate", W / 2, groundY + 14);

    // Native Oxide layer drawing (full width)
    if (st && st.oxideOpacity > 0) {
      cx.save();
      cx.globalAlpha = st.oxideOpacity;
      cx.fillStyle = "#a89078"; 
      cx.fillRect(0, groundY - 4, W, 4);
      cx.strokeStyle = "#7c6450";
      cx.lineWidth = 1.2;
      cx.beginPath();
      for (var x = 0; x <= W; x += 8) {
        var yOffset = Math.sin(x * 0.2) * 1.5;
        if (x === 0) cx.moveTo(x, groundY - 4 + yOffset);
        else cx.lineTo(x, groundY - 4 + yOffset);
      }
      cx.stroke();
      cx.fillStyle = "#8a705a";
      cx.font = "italic 9px sans-serif";
      cx.textAlign = "center";
      cx.fillText("Native Oxide desorbing (" + (st.oxideOpacity * 100).toFixed(0) + "% remaining)", W / 2, groundY - 9);
      cx.restore();
    }

    // GaSb Buffer (full width)
    if (bufferH > 0) {
      cx.fillStyle = COL.BUF;
      cx.fillRect(0, filmBaseY, W, bufferH);
      cx.fillStyle = "#ffffff";
      cx.font = "700 10px sans-serif";
      cx.textAlign = "center";
      cx.fillText("GaSb buffer 500 nm (compressed single layer)", W / 2, filmBaseY + Math.min(bufferH - 6, 16));
      if (cur.stage === "buffer") {
        cx.fillStyle = C.axisText;
        cx.textAlign = "left";
        cx.fillText("buffer progress " + bufferGrown.toFixed(0) + " / " + bufferNm.toFixed(0) + " nm", 8, Math.max(topPad + 12, filmBaseY - 8));
      }
    }

    // Superlattice film growth (full width)
    var cumul = 0;
    for (var i = bufIndex + 1; i <= st.si && i < st.steps.length; i++) {
      var s = st.steps[i], bandNm = (i < st.si) ? s.nm : s.nm * st.sp;
      var c0 = cumul, c1 = cumul + bandNm; cumul = c1;
      if (bandNm <= 0 || c1 <= upperViewBot) continue;
      var lo = Math.max(c0, upperViewBot), yB = filmBaseY - (lo - upperViewBot) * pxNm, yT = filmBaseY - (c1 - upperViewBot) * pxNm;
      cx.fillStyle = layerColor(s);
      cx.fillRect(0, yT, W, Math.max(0.7, yB - yT));
      if (Q < 0.72 && s.mat && i === st.si) {
        cx.fillStyle = "rgba(192,57,43,.18)";
        for (var r = 0; r < Math.round((1 - Q) * 14); r++) {
          var rx = ((r * 37) % W), rh = (1 - Q) * (4 + (r % 4));
          cx.fillRect(rx, yT - rh, 10, rh);
        }
      }
    }

    if (!st.done && cur.mat && st.sp < 1) drawFallingAtoms(cx, 0, W, frontY, cur.mat);
    if (Math.abs(st.accStrain) > M.STRAIN_CRIT * 0.7) {
      cx.strokeStyle = "rgba(192,57,43,.75)"; cx.lineWidth = 1.6;
      for (var d = 0; d < 4; d++) {
        var dx = W * (0.18 + d * 0.2);
        cx.beginPath(); cx.moveTo(dx, groundY - 8); cx.lineTo(dx + 18, groundY - 85 - d * 8); cx.stroke();
      }
    }

    cx.fillStyle = C.axisText; cx.font = "9px sans-serif"; cx.textAlign = "right";
    cx.fillText("SL scale: 10 nm · buffer compressed", W - 8, groundY - 10);
    cx.strokeStyle = C.axisText; cx.beginPath(); cx.moveTo(W - 63, groundY - 16); cx.lineTo(W - 8, groundY - 16); cx.stroke();
  }

  function drawStackMap() {
    if (!M.mcx) return;
    var st = M.st, mcx = M.mcx, W = M.dims.mpW, H = M.dims.mpH, n = Math.max(1, M.P.nPer);
    mcx.clearRect(0, 0, W, H); mcx.fillStyle = C.canvasBg; mcx.fillRect(0, 0, W, H);
    var padL = 0, padR = 0, padT = 8, padB = 18, pw = W - padL - padR, ph = H - padT - padB;
    var bufferIndex = getBufferIndex(st), bufferDone = bufferIndex >= 0 && st.si >= bufferIndex;
    var currentPeriod = M.curPeriod();
    var preW = Math.min(Math.max(76, pw * .18), 132), subW = preW * .38, bufW = preW - subW;
    var cycleX0 = padL + preW, cycleW = Math.max(1, pw - preW);
    var cycleTop = padT, cycleH = ph, stepW = cycleW / n;

    mcx.globalAlpha = 1;
    mcx.fillStyle = COL.SUB;
    rr(mcx, padL, cycleTop, subW, cycleH, 4); mcx.fill();
    mcx.fillStyle = bufferDone ? COL.BUF : "#d9e1ea";
    rr(mcx, padL + subW, cycleTop, bufW, cycleH, 4); mcx.fill();
    mcx.fillStyle = "#ffffff"; mcx.font = "8px sans-serif"; mcx.textAlign = "center";
    mcx.fillText("substrate", padL + subW / 2, cycleTop + cycleH / 2 + 3);
    mcx.fillText("buffer", padL + subW + bufW / 2, cycleTop + cycleH / 2 + 3);

    var hasGap = stepW > 4, gap = hasGap ? Math.min(1.5, stepW * .12) : 0;
    for (var p = 0; p < n; p++) {
      var x = cycleX0 + p * stepW, done = M.stageOf() === "sl" ? p + 1 < currentPeriod : M.stageIdx(M.stageOf()) > M.stageIdx("sl");
      var active = M.stageOf() === "sl" && p + 1 === currentPeriod;
      var innerX = x + gap, innerW = Math.max(1, stepW - gap * 2);
      var insbW = Math.max(1, innerW * .08), inasW = innerW * .62, insb2W = Math.max(1, innerW * .08);
      var gasbW = Math.max(1, innerW - insbW - inasW - insb2W);
      mcx.globalAlpha = done ? 0.92 : active ? 1 : 0.26;
      mcx.fillStyle = COL.InSb; mcx.fillRect(innerX, cycleTop, insbW, cycleH);
      mcx.fillStyle = COL.InAs; mcx.fillRect(innerX + insbW, cycleTop, inasW, cycleH);
      mcx.fillStyle = COL.InSb; mcx.fillRect(innerX + insbW + inasW, cycleTop, insb2W, cycleH);
      mcx.fillStyle = COL.GaSb; mcx.fillRect(innerX + insbW + inasW + insb2W, cycleTop, gasbW, cycleH);
      if (active) {
        mcx.globalAlpha = 1;
        mcx.strokeStyle = C.bad; mcx.lineWidth = 2;
        mcx.strokeRect(x - 1, cycleTop - 1, Math.max(3, stepW + 2), cycleH + 2);
      }
    }
    mcx.globalAlpha = 1;
    mcx.strokeStyle = C.axis; mcx.lineWidth = 1; mcx.strokeRect(padL, cycleTop, preW, cycleH);
    mcx.strokeRect(cycleX0, cycleTop, cycleW, cycleH);
    mcx.fillStyle = C.axisText; mcx.font = "9px sans-serif"; mcx.textAlign = "left";
    mcx.fillText("substrate", padL + 4, H - 4);
    mcx.fillText("P1", cycleX0, H - 4);
    mcx.textAlign = "center"; mcx.fillText("left to right: substrate / buffer / repeated InSb-like-InAs-InSb-like-GaSb periods", padL + pw / 2, H - 4);
    mcx.textAlign = "right"; mcx.fillText("P" + n, cycleX0 + cycleW - 4, H - 4);
  }

  function drawTiming() {
    if (!M.tcx) return;
    var tcx = M.tcx, W = M.dims.tmW, H = M.dims.tmH, keys = ["In", "Ga", "As", "Sb"], colors = [M.CELL.In.col, M.CELL.Ga.col, M.CELL.As.col, M.CELL.Sb.col];
    tcx.clearRect(0, 0, W, H); tcx.fillStyle = C.canvasBg; tcx.fillRect(0, 0, W, H);
    var padL = 34, padR = 8, padT = 12, rowH = (H - padT - 12) / 4, pw = W - padL - padR;
    tcx.fillStyle = C.axisText; tcx.font = "9px sans-serif"; tcx.textAlign = "right";
    for (var k = 0; k < keys.length; k++) {
      var y = padT + k * rowH + rowH * .5;
      tcx.fillText(keys[k], padL - 8, y + 3);
      tcx.strokeStyle = "#edf1f5"; tcx.beginPath(); tcx.moveTo(padL, y); tcx.lineTo(padL + pw, y); tcx.stroke();
    }
    var blocks = [
      { x: .02, w: .22, on: ["In", "As"] },
      { x: .28, w: .09, on: ["Sb"] },
      { x: .40, w: .12, on: ["In", "Sb"] },
      { x: .58, w: .26, on: ["Ga", "Sb"] },
      { x: .88, w: .08, on: ["Sb"] }
    ];
    for (var b = 0; b < blocks.length; b++) {
      for (var j = 0; j < keys.length; j++) if (blocks[b].on.indexOf(keys[j]) >= 0) {
        tcx.fillStyle = colors[j]; rr(tcx, padL + blocks[b].x * pw, padT + j * rowH + 4, blocks[b].w * pw, rowH - 8, 3); tcx.fill();
      }
    }
    tcx.fillStyle = C.axisText; tcx.font = "9px sans-serif"; tcx.textAlign = "center"; tcx.fillText("standard shutter recipe: InAs -> Sb soak -> InSb-like -> GaSb", padL + pw / 2, H - 3);
  }

  function drawBand() {
    if (!M.bcx) return;
    var bx = M.bcx, W = M.dims.bdW, H = M.dims.bdH, eg = M.effGap(), lam = M.cutoff(eg), overlap = M.interfaceMetrics().abruptness * M.surfaceQuality();
    bx.clearRect(0, 0, W, H); bx.fillStyle = C.canvasBg; bx.fillRect(0, 0, W, H);
    var pad = 28, mid = H * .52, left = pad, right = W - pad;
    bx.strokeStyle = COL.InAs; bx.lineWidth = 2;
    bx.beginPath(); bx.moveTo(left, mid - 38); bx.bezierCurveTo(W * .34, mid - 70, W * .42, mid - 12, W * .52, mid - 34); bx.bezierCurveTo(W * .62, mid - 58, W * .73, mid - 20, right, mid - 42); bx.stroke();
    bx.strokeStyle = COL.GaSb;
    bx.beginPath(); bx.moveTo(left, mid + 38); bx.bezierCurveTo(W * .35, mid + 18, W * .48, mid + 70, W * .58, mid + 36); bx.bezierCurveTo(W * .70, mid + 8, W * .78, mid + 60, right, mid + 34); bx.stroke();
    bx.fillStyle = "rgba(31,158,140,.16)"; bx.fillRect(W * .20, 28, W * .20, H - 58);
    bx.fillStyle = "rgba(207,138,28,.16)"; bx.fillRect(W * .52, 28, W * .20, H - 58);
    bx.fillStyle = C.axisText; bx.font = "10px sans-serif"; bx.textAlign = "center";
    bx.fillText("InAs electron well", W * .30, H - 18); bx.fillText("GaSb hole well", W * .62, H - 18);
    bx.textAlign = "left"; bx.font = "700 12px sans-serif"; bx.fillStyle = C.label;
    bx.fillText("Eg_eff " + eg.toFixed(3) + " eV", pad, 18);
    bx.fillText("lambda_c " + lam.toFixed(1) + " μm", W * .52, 18);
    bx.font = "10px sans-serif"; bx.fillText("wavefunction overlap " + Math.round(overlap * 100) + "%", pad, H - 4);
  }

  // 通用图表 drawScreen/drawStrain/drawRheed 由 engine.js 提供；此处仅导出领域示意图
  M.drawChamber = drawChamber;
  M.drawStackMap = drawStackMap; M.drawTiming = drawTiming; M.drawBand = drawBand;

})(window.MBE);
