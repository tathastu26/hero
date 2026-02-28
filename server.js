const express = require('express');
const axios = require('axios');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/', limits: { fileSize: 10 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const SERP_API_KEY    = 'e7ea59f9330c9708a3ccb9f8366d47ea8cf564da87d6ff5dd9505ff09b7cb836';
const SAPLING_API_KEY = process.env.SAPLING_API_KEY || '4B97BV3T70R6NZA277TPB2ZEL4N8NLPX';

// ─── SAPLING AI DETECTION ─────────────────────────────────────────────────────
async function callSaplingAI(text) {
  if (!SAPLING_API_KEY) return null;
  try {
    const res = await axios.post(
      'https://api.sapling.ai/api/v1/aidetect',
      { key: SAPLING_API_KEY, text },
      { timeout: 10000 }
    );
    const score = res.data?.score;
    return typeof score === 'number' ? score : null;
  } catch (e) {
    console.error('Sapling AI:', e.message);
    return null;
  }
}

// ─── HEURISTIC TEXT DETECTION ─────────────────────────────────────────────────
function analyzeTextHeuristics(text) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || text.split('\n').filter(Boolean);
  const words = text.toLowerCase().match(/\b[a-z']+\b/g) || [];
  const totalWords = words.length;
  if (totalWords < 10) return { score: 0.5, signals: [] };

  const signals = [];
  let aiScore = 0, totalWeight = 0;

  // 1. Sentence Length Variance
  const sentLengths = sentences.map(s => s.trim().split(/\s+/).length).filter(l => l > 2);
  if (sentLengths.length >= 3) {
    const mean = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
    const variance = sentLengths.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / sentLengths.length;
    const cv = Math.sqrt(variance) / mean;
    const w = 0.22, s = cv < 0.15 ? 0.90 : cv < 0.25 ? 0.70 : cv < 0.40 ? 0.45 : 0.20;
    aiScore += s * w; totalWeight += w;
    signals.push({ name: 'Sentence Uniformity', value: cv < 0.25 ? 'Very uniform (AI-like)' : 'Varied (human-like)', flag: cv < 0.25 ? 'ai' : 'human' });
  }

  // 2. AI Transition Phrase Density
  const aiPhrases = ['furthermore','moreover','in addition','it is worth noting','it is important to note',
    'in conclusion','to summarize','in summary','therefore','thus','additionally','notably',
    'significantly','interestingly','importantly','ultimately','in essence','overall',
    'needless to say','that being said','having said that','with that said','it can be argued',
    'it is clear that','by and large','to elaborate','in other words'];
  const lowerText = text.toLowerCase();
  const hits = aiPhrases.filter(p => lowerText.includes(p)).length;
  const phraseRate = hits / Math.max(sentLengths.length, 1);
  const w2 = 0.20, s2 = phraseRate > 0.8 ? 0.92 : phraseRate > 0.5 ? 0.78 : phraseRate > 0.25 ? 0.55 : 0.20;
  aiScore += s2 * w2; totalWeight += w2;
  signals.push({ name: 'AI Transition Phrases', value: `${hits} found (${hits > 2 ? 'high' : hits > 0 ? 'moderate' : 'none'})`, flag: hits > 2 ? 'ai' : hits > 0 ? 'uncertain' : 'human' });

  // 3. Vocabulary Richness (TTR)
  const ttr = new Set(words).size / totalWords;
  const adjTtr = totalWords > 200 ? ttr * (1 + Math.log10(totalWords / 200) * 0.3) : ttr;
  const w3 = 0.18, s3 = adjTtr < 0.40 ? 0.85 : adjTtr < 0.55 ? 0.55 : adjTtr < 0.70 ? 0.35 : 0.15;
  aiScore += s3 * w3; totalWeight += w3;
  signals.push({ name: 'Vocabulary Richness', value: `${Math.round(ttr * 100)}% unique words`, flag: adjTtr < 0.45 ? 'ai' : 'human' });

  // 4. Informal Punctuation (em-dash, ellipsis, exclamation)
  const informal = (text.match(/[—–]/g)||[]).length + (text.match(/\.\.\./, )||[]).length + (text.match(/!/g)||[]).length;
  const infRate = informal / Math.max(sentLengths.length, 1);
  const w4 = 0.14, s4 = infRate < 0.05 ? 0.80 : infRate < 0.15 ? 0.50 : infRate < 0.30 ? 0.30 : 0.10;
  aiScore += s4 * w4; totalWeight += w4;
  signals.push({ name: 'Informal Punctuation', value: informal === 0 ? 'None (AI-like)' : `${informal} instance(s)`, flag: informal === 0 ? 'ai' : 'human' });

  // 5. Lexical Burstiness
  const freq = {};
  words.forEach(w => { freq[w] = (freq[w] || 0) + 1; });
  const freqVals = Object.values(freq).filter(c => c > 1);
  const burst = freqVals.length > 0 ? Math.max(...freqVals) / (freqVals.reduce((a,b)=>a+b,0)/freqVals.length) : 1;
  const w5 = 0.13, s5 = burst < 2.5 ? 0.75 : burst < 4 ? 0.45 : 0.20;
  aiScore += s5 * w5; totalWeight += w5;
  signals.push({ name: 'Lexical Burstiness', value: burst < 2.5 ? 'Low — evenly spread (AI-like)' : 'High — clustered (human-like)', flag: burst < 2.5 ? 'ai' : 'human' });

  // 6. Paragraph Balance
  const paras = text.split(/\n\s*\n/).filter(p => p.trim().length > 20);
  if (paras.length >= 2) {
    const pLens = paras.map(p => p.trim().split(/\s+/).length);
    const pMean = pLens.reduce((a,b)=>a+b,0)/pLens.length;
    const pCV = Math.sqrt(pLens.reduce((a,b)=>a+Math.pow(b-pMean,2),0)/pLens.length) / pMean;
    const w6 = 0.13, s6 = pCV < 0.20 ? 0.82 : pCV < 0.35 ? 0.55 : 0.20;
    aiScore += s6 * w6; totalWeight += w6;
    signals.push({ name: 'Paragraph Balance', value: pCV < 0.25 ? 'Highly balanced (AI-like)' : 'Varied lengths (human-like)', flag: pCV < 0.25 ? 'ai' : 'human' });
  }

  return { score: totalWeight > 0 ? aiScore / totalWeight : 0.5, signals };
}

// ─── HEURISTIC IMAGE DETECTION ────────────────────────────────────────────────
function analyzeImageHeuristics(buffer, mimetype) {
  const signals = [];
  const votes = []; // { score, weight } — only added when we have real evidence

  const isJpeg = mimetype === 'image/jpeg' || mimetype === 'image/jpg' || (buffer[0] === 0xFF && buffer[1] === 0xD8);
  const isPng  = mimetype === 'image/png'  || (buffer[0] === 0x89 && buffer[1] === 0x50);
  const isWebP = mimetype === 'image/webp' || (buffer.length > 12 && buffer.slice(8, 12).toString('ascii') === 'WEBP');

  const AI_TOOLS = [
    'stable diffusion', 'dall-e', 'dall·e', 'midjourney', 'adobe firefly',
    'leonardo.ai', 'novel ai', 'comfyui', 'automatic1111', 'invokeai',
    'diffusers', 'dreamstudio', 'getimg.ai', 'nightcafe', 'artbreeder',
    'bluewillow', 'bing image creator', 'generative fill',
    'ai generated', 'generated by ai', 'wombo', 'runway ml'
  ];
  const AI_PARAM_KEYS = [
    'parameters\x00', 'prompt\x00', 'negative_prompt', 'negative prompt',
    'sd model', 'cfg scale', 'sampler name', 'steps\x00'
  ];

  // ── Full-buffer scan (up to 512 KB) for AI tool signatures ────────────────
  const scanStr = buffer.slice(0, Math.min(buffer.length, 524288)).toString('latin1').toLowerCase();
  const foundTool  = AI_TOOLS.find(t => scanStr.includes(t));
  const foundParam = AI_PARAM_KEYS.find(k => scanStr.includes(k));

  if (foundTool || foundParam) {
    const val = foundTool ? `"${foundTool}"` : `parameter key "${foundParam.replace('\x00','')}"`;
    signals.push({ name: 'AI Generator Signature', value: `Detected: ${val}`, flag: 'ai' });
    votes.push({ score: 0.97, weight: 10 }); // Definitive — very high weight
  }

  // ── PNG-specific ──────────────────────────────────────────────────────────
  if (isPng) {
    let hasSourceMeta = false, aiParamInChunk = false, creatorSW = null;
    let pos = 8;
    while (pos + 8 <= Math.min(buffer.length, 524288)) {
      let len;
      try { len = buffer.readUInt32BE(pos); } catch { break; }
      if (len < 0 || len > 524288) break;
      const type = buffer.slice(pos + 4, pos + 8).toString('ascii');
      if (['tEXt', 'iTXt', 'zTXt'].includes(type)) {
        const chunk = buffer.slice(pos + 8, Math.min(pos + 8 + len, pos + 8 + 4096)).toString('latin1');
        const lower = chunk.toLowerCase();
        // Check for actual AI generation parameters
        if (lower.includes('parameters') || lower.includes('negative prompt') ||
            lower.includes('negative_prompt') || lower.includes('cfg scale') ||
            lower.includes('sampler') || lower.includes('seed:')) {
          aiParamInChunk = true;
        }
        AI_TOOLS.forEach(t => { if (lower.includes(t)) creatorSW = t; });
        // Check for editing/camera software (positive human signal)
        if (/photoshop|lightroom|gimp|affinity|darktable|capture one|camera raw/i.test(chunk)) {
          hasSourceMeta = true;
        }
      } else if (type === 'eXIf') {
        hasSourceMeta = true; // EXIF embedded in PNG — real photo/scan
      }
      if (type === 'IDAT' || type === 'IEND') break;
      pos += 12 + len;
    }

    if (aiParamInChunk && !foundTool && !foundParam) {
      signals.push({ name: 'AI Prompt in PNG Chunks', value: 'Generation parameters found (Stable Diffusion style)', flag: 'ai' });
      votes.push({ score: 0.94, weight: 10 });
    }
    if (creatorSW && !foundTool) {
      signals.push({ name: 'PNG Creator Software', value: `AI tool identified: "${creatorSW}"`, flag: 'ai' });
      votes.push({ score: 0.95, weight: 10 });
    }
    if (hasSourceMeta) {
      signals.push({ name: 'PNG Source Metadata', value: 'Editing/camera software metadata found — processed real image', flag: 'human' });
      votes.push({ score: 0.12, weight: 3 });
    }
    // No metadata at all is NOT an AI signal — screenshots/web graphics never have it
    if (!hasSourceMeta && !aiParamInChunk && !foundTool && !creatorSW) {
      signals.push({ name: 'PNG Metadata', value: 'No metadata present (inconclusive — normal for screenshots & web graphics)', flag: 'uncertain' });
      // No vote added — cannot determine from absence alone
    }

  // ── JPEG-specific ─────────────────────────────────────────────────────────
  } else if (isJpeg) {
    let hasExif = false, hasCameraModel = false, hasGps = false, hasDatetime = false;
    let exifSoftware = null;

    for (let i = 2; i < Math.min(buffer.length - 3, 131072); i++) {
      if (buffer[i] !== 0xFF) continue;
      const marker = buffer[i + 1];
      if (marker === 0xDA) break; // Start of scan — image data begins, stop parsing

      if (marker === 0xE1 && i + 4 < buffer.length) {
        const segLen = buffer.readUInt16BE(i + 2);
        const seg = buffer.slice(i + 4, Math.min(i + 2 + segLen, buffer.length));
        const header = seg.slice(0, 6).toString('ascii');
        if (header.startsWith('Exif')) {
          hasExif = true;
          const exif = seg.toString('latin1');
          if (/Canon|Nikon|Sony|Apple|APPLE|iPhone|Samsung|Fujifilm|Panasonic|Olympus|Leica|Pentax|Ricoh|Xiaomi|Huawei|Google|OnePlus|Motorola/i.test(exif)) hasCameraModel = true;
          if (/GPS|GPSLatitude/i.test(exif)) hasGps = true;
          if (/20\d\d[:\-]\d\d[:\-]\d\d/.test(exif)) hasDatetime = true;
          const sw = AI_TOOLS.find(t => exif.toLowerCase().includes(t));
          if (sw) exifSoftware = sw;
        } else if (header.startsWith('http://') || seg.slice(0, 28).toString().includes('xpacket')) {
          // XMP data — check for AI tools
          const xmp = seg.toString('utf8');
          const sw = AI_TOOLS.find(t => xmp.toLowerCase().includes(t));
          if (sw && !foundTool) exifSoftware = sw;
        }
      }
    }

    if (exifSoftware && !foundTool) {
      signals.push({ name: 'EXIF/XMP Software Tag', value: `AI tool found: "${exifSoftware}"`, flag: 'ai' });
      votes.push({ score: 0.95, weight: 10 });
    }

    // Camera model is the strongest "real photo" signal
    if (hasCameraModel) {
      signals.push({ name: 'Camera EXIF', value: 'Camera make/model confirmed — real photograph', flag: 'human' });
      votes.push({ score: 0.07, weight: 4 });
    } else if (hasGps) {
      signals.push({ name: 'GPS Data', value: 'Location coordinates embedded — real photograph', flag: 'human' });
      votes.push({ score: 0.06, weight: 4 });
    } else if (hasDatetime) {
      signals.push({ name: 'Capture Timestamp', value: 'Original capture time present', flag: 'human' });
      votes.push({ score: 0.15, weight: 2 });
    } else if (hasExif) {
      signals.push({ name: 'Camera EXIF', value: 'EXIF present but no camera model (possibly stripped)', flag: 'uncertain' });
      votes.push({ score: 0.50, weight: 1 });
    } else {
      // No EXIF — very common in web images, social media, compressed photos — NOT a reliable AI signal
      signals.push({ name: 'Camera EXIF', value: 'No EXIF data (common in web-optimized & social media images — inconclusive)', flag: 'uncertain' });
      // Weak lean toward uncertain/AI but very low weight
      votes.push({ score: 0.55, weight: 0.5 });
    }

    // JPEG entropy — keep as informational only, very low weight
    const sosIdx = (() => {
      for (let i = 2; i < Math.min(buffer.length - 1, 65536); i++) {
        if (buffer[i] === 0xFF && buffer[i + 1] === 0xDA) return i + 4;
      }
      return -1;
    })();
    if (sosIdx > 0 && buffer.length > sosIdx + 2000) {
      const sample = buffer.slice(sosIdx, sosIdx + 2000);
      const freq = new Array(256).fill(0);
      sample.forEach(b => freq[b]++);
      const ent = freq.reduce((s, f) => { if (!f) return s; const p = f / 2000; return s - p * Math.log2(p); }, 0);
      const entLabel = ent > 7.6 ? 'high (photo-like)' : ent > 7.0 ? 'normal' : 'low';
      signals.push({ name: 'JPEG Compression Entropy', value: `${ent.toFixed(2)} bits/symbol — ${entLabel}`, flag: ent < 6.8 ? 'uncertain' : 'human' });
      // Entropy alone is very unreliable for AI detection — informational only, minimal weight
      if (ent < 6.5) votes.push({ score: 0.65, weight: 0.3 });
    }

  } else if (isWebP) {
    signals.push({ name: 'Format', value: 'WebP — used by AI tools and web images alike (inconclusive)', flag: 'uncertain' });
  } else {
    signals.push({ name: 'Format', value: 'Unknown/unsupported format — limited forensic analysis possible', flag: 'uncertain' });
  }

  // ── Compute final score ───────────────────────────────────────────────────
  // If we have no concrete evidence, default to uncertain (0.50)
  if (votes.length === 0) {
    return { score: 0.50, signals };
  }

  const totalW = votes.reduce((s, v) => s + v.weight, 0);
  const wavg   = votes.reduce((s, v) => s + v.score * v.weight, 0) / totalW;

  // If any single vote is a definitive AI signature (score ≥ 0.94), let it dominate
  const maxScore = Math.max(...votes.map(v => v.score));
  const finalScore = maxScore >= 0.94 ? maxScore : wavg;

  return { score: Math.max(0, Math.min(1, finalScore)), signals };
}

// ─── TEXT ENDPOINT ───────────────────────────────────────────────────────────
app.post('/api/analyze-text', async (req, res) => {
  const { text } = req.body;
  if (!text || text.trim().length < 20)
    return res.status(400).json({ error: 'Text must be at least 20 characters.' });

  try {
    const { score: heuristicScore, signals } = analyzeTextHeuristics(text);

    // Blend with Sapling AI if key is configured (Sapling is weighted 60%, heuristics 40%)
    const saplingScore = await callSaplingAI(text);
    const score = saplingScore !== null
      ? saplingScore * 0.60 + heuristicScore * 0.40
      : heuristicScore;

    if (saplingScore !== null) {
      signals.unshift({ name: 'Sapling AI Engine', value: `${Math.round(saplingScore * 100)}% AI probability`, flag: saplingScore > 0.6 ? 'ai' : saplingScore > 0.4 ? 'uncertain' : 'human' });
    }

    const detectionResult = {
      aiProbability: Math.round(score * 100),
      humanProbability: Math.round((1 - score) * 100),
      verdict: score > 0.68 ? 'AI-Generated' : score > 0.42 ? 'Uncertain' : 'Human-Written',
      confidence: score > 0.82 || score < 0.18 ? 'High' : score > 0.62 || score < 0.35 ? 'Medium' : 'Low',
      signals,
      engine: saplingScore !== null ? 'Sapling AI + Heuristics' : 'Heuristic Engine'
    };

    let sources = [];
    try {
      const snippet = text.slice(0, 120).replace(/[^a-zA-Z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
      const serpRes = await axios.get('https://serpapi.com/search', {
        params: { q: `"${snippet}"`, api_key: SERP_API_KEY, num: 5, engine: 'google' },
        timeout: 15000
      });
      const organic = serpRes.data.organic_results || [];
      sources = organic.slice(0, 5).map(r => ({
        title: r.title, url: r.link, snippet: r.snippet, displayUrl: r.displayed_link || r.link
      }));
    } catch (e) { console.error('SerpAPI:', e.message); }

    res.json({ detection: detectionResult, sources, type: 'text' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Analysis failed. Please try again.' });
  }
});

// ─── IMAGE ENDPOINT ──────────────────────────────────────────────────────────
app.post('/api/analyze-image', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image uploaded.' });

  const filePath = req.file.path;
  try {
    const imageBuffer = fs.readFileSync(filePath);
    const { score, signals } = analyzeImageHeuristics(imageBuffer, req.file.mimetype);

    const detectionResult = {
      aiProbability: Math.round(score * 100),
      humanProbability: Math.round((1 - score) * 100),
      verdict: score > 0.68 ? 'AI-Generated' : score > 0.42 ? 'Uncertain' : 'Real Image',
      confidence: score > 0.80 || score < 0.20 ? 'High' : score > 0.60 || score < 0.38 ? 'Medium' : 'Low',
      signals
    };

    // Reverse image search via SerpAPI (Google Lens)
    let sources = [];
    try {
      const serpRes = await axios.get('https://serpapi.com/search', {
        params: { engine: 'google_lens', url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png', api_key: SERP_API_KEY },
        timeout: 15000
      });
      const results = serpRes.data.visual_matches || serpRes.data.image_results || serpRes.data.organic_results || [];
      sources = results.slice(0, 5).map(r => ({
        title: r.title, url: r.link || r.source,
        snippet: r.snippet || r.source || '',
        displayUrl: r.displayed_link || r.link || r.source,
        thumbnail: r.thumbnail
      }));
    } catch (e) { console.error('SerpAPI reverse image:', e.message); }

    fs.unlinkSync(filePath);
    res.json({ detection: detectionResult, sources, type: 'image' });
  } catch (err) {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    console.error(err);
    res.status(500).json({ error: 'Image analysis failed. Please try again.' });
  }
});

// ─── Serve frontend ───────────────────────────────────────────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n🚀 AuthVerifier.ai running at http://localhost:${PORT}\n`);
});
