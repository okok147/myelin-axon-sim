(function () {
  "use strict";

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function rand() {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function alphaN(v) {
    const x = (10 - v) / 10;
    return 0.01 * (10 - v) / (Math.exp(x) - 1);
  }

  function betaN(v) {
    return 0.125 * Math.exp(-v / 80);
  }

  function alphaM(v) {
    const x = (25 - v) / 10;
    return 0.1 * (25 - v) / (Math.exp(x) - 1);
  }

  function betaM(v) {
    return 4 * Math.exp(-v / 18);
  }

  function alphaH(v) {
    return 0.07 * Math.exp(-v / 20);
  }

  function betaH(v) {
    return 1 / (Math.exp((30 - v) / 10) + 1);
  }

  function safeRate(fn, v) {
    const x = fn(v);
    if (!Number.isFinite(x)) {
      return 1e-6;
    }
    return Math.max(1e-6, x);
  }

  function colorForVoltage(vmMv) {
    const v = clamp((vmMv + 90) / 130, 0, 1);
    let r;
    let g;
    let b;
    if (v < 0.35) {
      const t = v / 0.35;
      r = lerp(16, 57, t);
      g = lerp(34, 99, t);
      b = lerp(95, 173, t);
    } else if (v < 0.7) {
      const t = (v - 0.35) / 0.35;
      r = lerp(57, 236, t);
      g = lerp(99, 166, t);
      b = lerp(173, 71, t);
    } else {
      const t = (v - 0.7) / 0.3;
      r = lerp(236, 255, t);
      g = lerp(166, 90, t);
      b = lerp(71, 54, t);
    }
    return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
  }

  function thresholdCrossing(timeMs, traceMv, thresholdMv) {
    for (let i = 1; i < traceMv.length; i += 1) {
      if (traceMv[i - 1] < thresholdMv && traceMv[i] >= thresholdMv) {
        return timeMs[i];
      }
    }
    return null;
  }

  function buildDamageProfile(preset, severity, xNorm, seed) {
    const rand = mulberry32(seed);
    const profile = new Float64Array(xNorm.length);

    if (preset === "CMT") {
      for (let i = 0; i < profile.length; i += 1) {
        profile[i] = 0.65 + 0.25 * severity;
      }
      return profile;
    }

    if (preset === "HEALTHY") {
      for (let i = 0; i < profile.length; i += 1) {
        profile[i] = 1;
      }
      return profile;
    }

    if (preset === "ADEM") {
      for (let i = 0; i < profile.length; i += 1) {
        const x = xNorm[i];
        const diffuse = 0.55 + 0.25 * Math.sin(8 * x + 1.3) + 0.2 * (rand() - 0.5);
        profile[i] = clamp(diffuse, 0.2, 1);
      }
      return profile;
    }

    const nLesionsBase = {
      MS: 2 + Math.round(3 * severity),
      GBS: 3 + Math.round(4 * severity),
      CIDP: 4 + Math.round(5 * severity),
    };

    const nLesions = nLesionsBase[preset] || (2 + Math.round(3 * severity));
    const widthBase = preset === "MS" ? 0.06 : preset === "GBS" ? 0.045 : 0.038;

    for (let l = 0; l < nLesions; l += 1) {
      const center = 0.08 + 0.84 * rand();
      const width = widthBase + 0.11 * severity * (0.5 + rand());
      const amp = 0.6 + 0.4 * rand();
      for (let i = 0; i < profile.length; i += 1) {
        const dx = (xNorm[i] - center) / width;
        const lesion = amp * Math.exp(-dx * dx);
        profile[i] += lesion;
      }
    }

    for (let i = 0; i < profile.length; i += 1) {
      profile[i] = clamp(profile[i], 0.15, 1);
    }
    return profile;
  }

  function simulateAxon(config) {
    const nNodes = 16;
    const internodeSegs = 6;
    const nodeLenUm = 1.0;
    const internodeLenUm = config.preset === "CMT" ? 620 : 820;
    const diameterUm = config.preset === "GBS" || config.preset === "CIDP" || config.preset === "CMT" ? 10 : 8;

    const umToM = 1e-6;
    const dtMs = 0.002;
    const dtS = dtMs * 1e-3;
    const tStopMs = 8.8;
    const steps = Math.floor(tStopMs / dtMs);

    const nodeLen = nodeLenUm * umToM;
    const internodeLen = internodeLenUm * umToM;
    const diameter = diameterUm * umToM;

    const lengths = [];
    const isNode = [];
    const nodeIndices = [];

    for (let n = 0; n < nNodes; n += 1) {
      nodeIndices.push(lengths.length);
      lengths.push(nodeLen);
      isNode.push(true);
      if (n < nNodes - 1) {
        const segLen = internodeLen / internodeSegs;
        for (let s = 0; s < internodeSegs; s += 1) {
          lengths.push(segLen);
          isNode.push(false);
        }
      }
    }

    const nComp = lengths.length;
    const area = new Float64Array(nComp);
    const xCenter = new Float64Array(nComp);
    const xNorm = new Float64Array(nComp);

    let x = 0;
    for (let i = 0; i < nComp; i += 1) {
      const len = lengths[i];
      xCenter[i] = x + 0.5 * len;
      x += len;
      area[i] = Math.PI * diameter * len;
    }
    for (let i = 0; i < nComp; i += 1) {
      xNorm[i] = xCenter[i] / x;
    }

    const aCross = Math.PI * Math.pow(diameter * 0.5, 2);
    const baseRa = config.preset === "CMT" ? 0.88 : config.preset === "GBS" || config.preset === "CIDP" ? 0.76 : 0.7;

    const gAxL = new Float64Array(nComp);
    const gAxR = new Float64Array(nComp);
    for (let i = 0; i < nComp - 1; i += 1) {
      const dx = 0.5 * (lengths[i] + lengths[i + 1]);
      const g = aCross / (baseRa * dx);
      gAxR[i] = g;
      gAxL[i + 1] = g;
    }

    const damageProfile = buildDamageProfile(config.preset, config.severity, xNorm, config.seed + 7);

    const cmNode = 0.028;
    const cmMyelin = 0.0010;
    const rmMyelinBase = config.preset === "CMT" ? 2.7 : 4.6;

    const cMem = new Float64Array(nComp);
    const gLeakPassive = new Float64Array(nComp);
    const nodeDamage = new Float64Array(nComp);

    let avgDamage = 0;

    for (let i = 0; i < nComp; i += 1) {
      const lesion = clamp(damageProfile[i], 0, 1);
      const localSeverity = clamp(config.severity * lesion, 0, 1);
      nodeDamage[i] = localSeverity;
      avgDamage += localSeverity;

      if (isNode[i]) {
        const cm = cmNode * (1 + 0.15 * localSeverity);
        cMem[i] = area[i] * cm;
        gLeakPassive[i] = 0;
      } else {
        const myelinIntegrity = clamp(1 - localSeverity, 0.02, 1);
        const rm = rmMyelinBase * (0.1 + 0.9 * myelinIntegrity * myelinIntegrity);
        const cm = cmMyelin * (1 + 8.5 * (1 - myelinIntegrity));
        const leakBoost = 1 + 2.2 * localSeverity;
        cMem[i] = area[i] * cm;
        gLeakPassive[i] = (area[i] / rm) * leakBoost;
      }
    }

    avgDamage /= nComp;

    const q10 = Math.pow(3, (config.tempC - 6.3) / 10);
    const tempStress = Math.max(0, config.tempC - 37);

    const vRest = -0.065;
    const eNa = 0.050;
    const eK = -0.077;
    const eL = -0.0544;
    const gNaBase = 1200;
    const gKBase = 360;
    const gLNode = 3;

    const v = new Float64Array(nComp);
    const m = new Float64Array(nComp);
    const h = new Float64Array(nComp);
    const n = new Float64Array(nComp);

    const vmStore = new Array(steps);
    const tStore = new Float64Array(steps);

    for (let i = 0; i < nComp; i += 1) {
      v[i] = vRest;
      m[i] = 0.05;
      h[i] = 0.6;
      n[i] = 0.32;
    }

    let node0Peak = -100;
    let distalPeak = -100;

    for (let step = 0; step < steps; step += 1) {
      const tMs = step * dtMs;
      tStore[step] = tMs;

      for (let k = 0; k < nodeIndices.length; k += 1) {
        const i = nodeIndices[k];
        const vRelMv = (v[i] - vRest) * 1e3;
        const aM = safeRate(alphaM, vRelMv);
        const bM = safeRate(betaM, vRelMv);
        const aH = safeRate(alphaH, vRelMv);
        const bH = safeRate(betaH, vRelMv);
        const aN = safeRate(alphaN, vRelMv);
        const bN = safeRate(betaN, vRelMv);

        m[i] += dtMs * q10 * (aM * (1 - m[i]) - bM * m[i]);
        h[i] += dtMs * q10 * (aH * (1 - h[i]) - bH * h[i]);
        n[i] += dtMs * q10 * (aN * (1 - n[i]) - bN * n[i]);

        m[i] = clamp(m[i], 0, 1);
        h[i] = clamp(h[i], 0, 1);
        n[i] = clamp(n[i], 0, 1);
      }

      const dv = new Float64Array(nComp);

      for (let i = 0; i < nComp; i += 1) {
        let iAxial = 0;
        if (i > 0) {
          iAxial += gAxL[i] * (v[i - 1] - v[i]);
        }
        if (i < nComp - 1) {
          iAxial += gAxR[i] * (v[i + 1] - v[i]);
        }

        let iMem;
        if (isNode[i]) {
          const damage = nodeDamage[i];
          const naComp = 1 + 0.32 * damage;
          const tempPenalty = tempStress * 0.058 * damage;
          const gNa = gNaBase * naComp * (1 - tempPenalty);
          const gK = gKBase * (1 + 0.03 * tempStress);

          const iNa = area[i] * gNa * Math.pow(m[i], 3) * h[i] * (v[i] - eNa);
          const iK = area[i] * gK * Math.pow(n[i], 4) * (v[i] - eK);
          const iL = area[i] * gLNode * (v[i] - eL);
          iMem = iNa + iK + iL;
        } else {
          iMem = gLeakPassive[i] * (v[i] - vRest);
        }

        let iStim = 0;
        if (i === nodeIndices[0] && tMs >= 0.2 && tMs <= 0.65) {
          iStim = (config.preset === "CMT" ? 2.15e-9 : 1.95e-9) * (1 + 0.22 * config.severity);
        }

        dv[i] = (iAxial + iStim - iMem) / cMem[i];
      }

      for (let i = 0; i < nComp; i += 1) {
        v[i] += dtS * dv[i];
      }

      const row = new Float32Array(nComp);
      for (let i = 0; i < nComp; i += 1) {
        const mv = v[i] * 1e3;
        row[i] = mv;
      }
      vmStore[step] = row;

      const n0 = row[nodeIndices[0]];
      const nd = row[nodeIndices[nodeIndices.length - 1]];
      if (n0 > node0Peak) {
        node0Peak = n0;
      }
      if (nd > distalPeak) {
        distalPeak = nd;
      }
    }

    const nodeTraces = nodeIndices.map((idx) => {
      const trace = new Float64Array(steps);
      for (let i = 0; i < steps; i += 1) {
        trace[i] = vmStore[i][idx];
      }
      return trace;
    });

    const thresholdMv = -20;
    const nodeTimes = nodeTraces.map((trace) => thresholdCrossing(tStore, trace, thresholdMv));
    const tFirst = nodeTimes[0];
    const tLast = nodeTimes[nodeTimes.length - 1];

    const distanceM = xCenter[nodeIndices[nodeIndices.length - 1]] - xCenter[nodeIndices[0]];
    let velocity = null;
    if (tFirst !== null && tLast !== null && tLast > tFirst) {
      velocity = distanceM / ((tLast - tFirst) * 1e-3);
    }

    const success = tLast !== null;

    const riskBase =
      0.72 * config.severity +
      0.24 * avgDamage +
      0.09 * tempStress +
      (config.preset === "GBS" || config.preset === "CIDP" ? 0.08 : 0);
    const speedTerm = velocity === null ? 1 : clamp((45 - velocity) / 45, 0, 1);

    let blockProbability = clamp(0.4 * riskBase + 0.58 * speedTerm, 0.01, 0.99);
    if (!success) {
      blockProbability = Math.max(blockProbability, 0.74);
    }
    if (config.preset === "CMT") {
      blockProbability *= 0.5;
    }
    blockProbability = clamp(blockProbability, 0.01, 0.99);

    return {
      tMs: tStore,
      vmStore,
      xUm: Array.from(xCenter, (d) => d / 1e-6),
      nodeIndices,
      nodeTimes,
      velocity,
      blockProbability,
      success,
      node0Peak,
      distalPeak,
      averageDamage: avgDamage,
      totalLengthUm: x / umToM,
    };
  }

  function drawAxon(canvas, sim, frameIndex, label) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padX = 34;
    const top = 34;
    const laneY = 104;
    const barH = 42;

    ctx.fillStyle = "#f8fcff";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#d6e4ed";
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, laneY - barH / 2, w - padX * 2, barH);

    const frame = sim.vmStore[Math.floor(frameIndex)];
    const xMin = sim.xUm[0];
    const xMax = sim.xUm[sim.xUm.length - 1];

    for (let i = 0; i < sim.xUm.length; i += 1) {
      const xNorm = (sim.xUm[i] - xMin) / (xMax - xMin);
      const px = lerp(padX, w - padX, xNorm);
      const nextX = i < sim.xUm.length - 1 ? lerp(padX, w - padX, (sim.xUm[i + 1] - xMin) / (xMax - xMin)) : px + 6;
      const width = Math.max(2, nextX - px + 1);

      ctx.fillStyle = colorForVoltage(frame[i]);
      ctx.fillRect(px, laneY - barH / 2 + 1, width, barH - 2);
    }

    for (let k = 0; k < sim.nodeIndices.length; k += 1) {
      const idx = sim.nodeIndices[k];
      const xNorm = (sim.xUm[idx] - xMin) / (xMax - xMin);
      const px = lerp(padX, w - padX, xNorm);
      const vm = frame[idx];
      ctx.beginPath();
      ctx.fillStyle = vm > -25 ? "#ff7842" : "#296188";
      ctx.arc(px, laneY, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(16, 35, 50, 0.35)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    const t = sim.tMs[Math.floor(frameIndex)];
    ctx.fillStyle = "#1d3343";
    ctx.font = "600 14px Space Grotesk";
    ctx.fillText(`${label} | t = ${t.toFixed(2)} ms`, padX, 22);

    const n0 = sim.vmStore[Math.floor(frameIndex)][sim.nodeIndices[0]];
    const nd = sim.vmStore[Math.floor(frameIndex)][sim.nodeIndices[sim.nodeIndices.length - 1]];
    ctx.font = "500 12px Space Grotesk";
    ctx.fillStyle = "#486171";
    ctx.fillText(`proximal node: ${n0.toFixed(1)} mV`, padX, h - 20);
    ctx.fillText(`distal node: ${nd.toFixed(1)} mV`, padX + 210, h - 20);
  }

  function drawHeat(canvas, sim, frameIndex) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#f8fcff";
    ctx.fillRect(0, 0, w, h);

    const padX = 40;
    const padY = 20;
    const plotW = w - padX - 12;
    const plotH = h - padY - 26;

    const rows = 120;
    const frame = Math.floor(frameIndex);
    const start = Math.max(0, frame - rows + 1);
    const end = frame;
    const sampleRows = end - start + 1;

    for (let r = 0; r < sampleRows; r += 1) {
      const idx = start + r;
      const y0 = padY + (plotH * r) / rows;
      const y1 = padY + (plotH * (r + 1)) / rows;
      const row = sim.vmStore[idx];
      for (let c = 0; c < row.length; c += 1) {
        const x0 = padX + (plotW * c) / row.length;
        const x1 = padX + (plotW * (c + 1)) / row.length;
        ctx.fillStyle = colorForVoltage(row[c]);
        ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
      }
    }

    ctx.strokeStyle = "#d6e4ed";
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, padY, plotW, plotH);

    ctx.fillStyle = "#1d3343";
    ctx.font = "600 13px Space Grotesk";
    ctx.fillText("Space-time map (recent window)", 14, 14);

    ctx.font = "500 11px Space Grotesk";
    ctx.fillStyle = "#486171";
    const t0 = sim.tMs[start];
    const t1 = sim.tMs[end];
    ctx.fillText(`time window: ${t0.toFixed(2)} to ${t1.toFixed(2)} ms`, padX, h - 8);
  }

  function nodeProgressAt(sim, tMs) {
    let passed = 0;
    for (let i = 0; i < sim.nodeTimes.length; i += 1) {
      if (sim.nodeTimes[i] !== null && sim.nodeTimes[i] <= tMs) {
        passed = i;
      }
    }
    return passed / (sim.nodeTimes.length - 1);
  }

  function drawSynchrony(canvas, simA, simB, frameIndex, freqHz) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#f8fcff";
    ctx.fillRect(0, 0, w, h);

    const pad = 36;
    const yA = 44;
    const yB = 94;
    const x0 = pad;
    const x1 = w - pad;

    ctx.strokeStyle = "#d6e4ed";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x0, yA);
    ctx.lineTo(x1, yA);
    ctx.moveTo(x0, yB);
    ctx.lineTo(x1, yB);
    ctx.stroke();

    const t = simA.tMs[Math.floor(frameIndex)];
    const pA = nodeProgressAt(simA, t);
    const pB = nodeProgressAt(simB, t);

    const xA = lerp(x0, x1, pA);
    const xB = lerp(x0, x1, pB);

    ctx.fillStyle = "#0f7a75";
    ctx.beginPath();
    ctx.arc(xA, yA, 6.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#d66b2c";
    ctx.beginPath();
    ctx.arc(xB, yB, 6.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#224154";
    ctx.font = "600 12px Space Grotesk";
    ctx.fillText("Pathway A", 8, yA + 4);
    ctx.fillText("Pathway B", 8, yB + 4);

    ctx.strokeStyle = "#203646";
    ctx.beginPath();
    ctx.moveTo(x1, 18);
    ctx.lineTo(x1, h - 14);
    ctx.stroke();
    ctx.fillText("Target", x1 - 18, 14);

    const tA = simA.nodeTimes[simA.nodeTimes.length - 1];
    const tB = simB.nodeTimes[simB.nodeTimes.length - 1];

    let delayMs = null;
    let phaseRad = null;
    if (tA !== null && tB !== null) {
      delayMs = tA - tB;
      phaseRad = 2 * Math.PI * freqHz * (delayMs / 1000);
    }

    ctx.fillStyle = "#486171";
    ctx.font = "500 11px Space Grotesk";
    const delayTxt = delayMs === null ? "n/a" : `${delayMs.toFixed(3)} ms`;
    const phaseTxt = phaseRad === null ? "n/a" : `${phaseRad.toFixed(3)} rad`;
    ctx.fillText(`delay delta: ${delayTxt}  |  phase delta @ ${freqHz} Hz: ${phaseTxt}`, pad, h - 8);

    return { delayMs, phaseRad };
  }

  const controls = {
    preset: document.getElementById("preset"),
    practice: document.getElementById("practice"),
    severity: document.getElementById("severity"),
    temperature: document.getElementById("temperature"),
    phaseFreq: document.getElementById("phaseFreq"),
    runBtn: document.getElementById("runBtn"),
    pauseBtn: document.getElementById("pauseBtn"),
    resetBtn: document.getElementById("resetBtn"),
    severityValue: document.getElementById("severityValue"),
    temperatureValue: document.getElementById("temperatureValue"),
    phaseFreqValue: document.getElementById("phaseFreqValue"),
    practiceValue: document.getElementById("practiceValue"),
    velocityMetric: document.getElementById("velocityMetric"),
    blockMetric: document.getElementById("blockMetric"),
    delayMetric: document.getElementById("delayMetric"),
    phaseMetric: document.getElementById("phaseMetric"),
  };

  const canvases = {
    axon: document.getElementById("axonCanvas"),
    heat: document.getElementById("heatCanvas"),
    sync: document.getElementById("syncCanvas"),
  };

  const presetOffsets = {
    HEALTHY: 0,
    MS: 0.12,
    GBS: 0.18,
    CIDP: 0.21,
    CMT: 0.1,
    ADEM: 0.15,
  };

  const practiceProfiles = {
    progressive: {
      msg: "Adaptive training rate: +3% myelin integrity per run",
      delta: -0.03,
    },
    correct: {
      msg: "Consistent good practice: +1.5% integrity per run",
      delta: -0.015,
    },
    plateau: {
      msg: "Repetition without challenge: 0% integrity change",
      delta: 0,
    },
    wrong: {
      msg: "Wrong practice: -2.2% integrity per run",
      delta: 0.022,
    },
  };

  const state = {
    trainingShift: 0,
    simA: null,
    simB: null,
    frame: 0,
    raf: null,
    playing: false,
    lastTs: null,
    phaseInfo: { delayMs: null, phaseRad: null },
  };

  function effectiveSeverity() {
    const base = Number(controls.severity.value) / 100;
    const preset = controls.preset.value;
    const offset = presetOffsets[preset] || 0;
    const eff = clamp(base + offset + state.trainingShift, 0, 0.98);
    return eff;
  }

  function updateValueLabels() {
    const eff = effectiveSeverity();
    const basePct = Number(controls.severity.value);
    controls.severityValue.textContent = `${basePct.toFixed(0)}% (effective ${Math.round(eff * 100)}%)`;
    controls.temperatureValue.textContent = `${Number(controls.temperature.value).toFixed(1)} deg C`;
    controls.phaseFreqValue.textContent = `${Number(controls.phaseFreq.value).toFixed(0)} Hz`;
    controls.practiceValue.textContent = practiceProfiles[controls.practice.value].msg;
  }

  function buildConfig(seedAdjust) {
    const severity = effectiveSeverity();
    return {
      preset: controls.preset.value,
      severity,
      tempC: Number(controls.temperature.value),
      seed: 19 + seedAdjust,
    };
  }

  function refreshMetrics() {
    if (!state.simA || !state.simB) {
      return;
    }

    const velocityTxt =
      state.simA.velocity === null ? "blocked" : `${state.simA.velocity.toFixed(1)} m/s`;
    controls.velocityMetric.textContent = velocityTxt;
    controls.blockMetric.textContent = `${(state.simA.blockProbability * 100).toFixed(1)}%`;

    if (state.phaseInfo.delayMs === null || state.phaseInfo.phaseRad === null) {
      controls.delayMetric.textContent = "n/a";
      controls.phaseMetric.textContent = "n/a";
    } else {
      controls.delayMetric.textContent = `${state.phaseInfo.delayMs.toFixed(3)} ms`;
      const deg = (state.phaseInfo.phaseRad * 180) / Math.PI;
      controls.phaseMetric.textContent = `${state.phaseInfo.phaseRad.toFixed(3)} rad (${deg.toFixed(1)} deg)`;
    }
  }

  function renderFrame() {
    if (!state.simA || !state.simB) {
      return;
    }
    const frame = Math.floor(state.frame);
    drawAxon(canvases.axon, state.simA, frame, "Pathway A");
    drawHeat(canvases.heat, state.simA, frame);
    state.phaseInfo = drawSynchrony(
      canvases.sync,
      state.simA,
      state.simB,
      frame,
      Number(controls.phaseFreq.value)
    );
    refreshMetrics();
  }

  function runSimulation() {
    const practice = practiceProfiles[controls.practice.value];
    state.trainingShift = clamp(state.trainingShift + practice.delta, -0.28, 0.32);
    updateValueLabels();

    state.simA = simulateAxon(buildConfig(0));

    const refConfig = buildConfig(41);
    refConfig.severity = clamp(refConfig.severity * 0.48, 0, 0.7);
    if (refConfig.preset === "CMT") {
      refConfig.severity = clamp(refConfig.severity * 0.75, 0, 0.8);
    }
    state.simB = simulateAxon(refConfig);

    state.frame = 0;
    state.lastTs = null;
    state.playing = true;
    renderFrame();
  }

  function pauseSimulation() {
    state.playing = false;
  }

  function resetSimulation() {
    state.trainingShift = 0;
    state.frame = 0;
    state.playing = false;
    updateValueLabels();

    const baseline = buildConfig(0);
    state.simA = simulateAxon(baseline);

    const ref = buildConfig(41);
    ref.severity = clamp(ref.severity * 0.48, 0, 0.7);
    state.simB = simulateAxon(ref);

    renderFrame();
  }

  function tick(ts) {
    if (state.lastTs === null) {
      state.lastTs = ts;
    }
    const dt = ts - state.lastTs;
    state.lastTs = ts;

    if (state.playing && state.simA) {
      const speed = 0.46;
      state.frame += dt * speed;
      if (state.frame >= state.simA.tMs.length - 1) {
        state.frame = state.simA.tMs.length - 1;
        state.playing = false;
      }
      renderFrame();
    }

    state.raf = requestAnimationFrame(tick);
  }

  function handleControlChange() {
    updateValueLabels();
    const a = buildConfig(0);
    const b = buildConfig(41);
    b.severity = clamp(b.severity * 0.48, 0, 0.7);
    state.simA = simulateAxon(a);
    state.simB = simulateAxon(b);
    state.frame = 0;
    renderFrame();
  }

  controls.runBtn.addEventListener("click", runSimulation);
  controls.pauseBtn.addEventListener("click", pauseSimulation);
  controls.resetBtn.addEventListener("click", resetSimulation);

  controls.preset.addEventListener("change", handleControlChange);
  controls.practice.addEventListener("change", updateValueLabels);
  controls.severity.addEventListener("input", handleControlChange);
  controls.temperature.addEventListener("input", handleControlChange);
  controls.phaseFreq.addEventListener("input", () => {
    updateValueLabels();
    renderFrame();
  });

  updateValueLabels();
  resetSimulation();
  state.raf = requestAnimationFrame(tick);

  window.addEventListener("beforeunload", () => {
    if (state.raf) {
      cancelAnimationFrame(state.raf);
      state.raf = null;
    }
  });
})();
