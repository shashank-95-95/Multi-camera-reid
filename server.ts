import express from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const PROJECT_ROOT = __dirname;
const MODEL_ROOT = PROJECT_ROOT;

app.use(express.json());

// Set up file uploads
const uploadsDir = path.join(PROJECT_ROOT, 'uploads');
const resultsDir = path.join(PROJECT_ROOT, 'results');
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});
const upload = multer({ storage });

// In-memory jobs database
interface Job {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  files: string[];
  config: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  outputDirectory?: string;
  resultUrl?: string;
  logs: string[];
}
const jobs: Record<string, Job> = {};

function parseBoolean(value: unknown): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.toLowerCase();
    return normalized === 'true' || normalized === '1' || normalized === 'yes';
  }
  return true;
}

function pushLog(job: Job, message: string) {
  job.logs.push(message);
  job.updatedAt = new Date().toISOString();
}

function writeResultManifest(jobId: string, outputDirectory: string, job: Job) {
  const manifestPath = path.join(resultsDir, `result_${jobId}.json`);
  const files: string[] = [];

  const walk = (dir: string) => {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile()) {
        files.push(path.relative(PROJECT_ROOT, fullPath));
      }
    }
  };

  walk(outputDirectory);

  const manifest = {
    jobId,
    outputDirectory,
    generatedAt: new Date().toISOString(),
    files,
    summary: 'Person Re-ID processing completed successfully.',
  };

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  job.resultUrl = `/results/${path.basename(manifestPath)}`;
  pushLog(job, `Results manifest written to ${path.basename(manifestPath)}`);
}

function startRealProcessing(jobId: string, videoPaths: string[], config: Record<string, unknown>) {
  const job = jobs[jobId];
  if (!job) return;

  job.status = 'processing';
  job.progress = 5;
  job.updatedAt = new Date().toISOString();
  pushLog(job, 'Starting Python Re-ID pipeline...');
  pushLog(job, `Using model root: ${MODEL_ROOT}`);

  const venvPython = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe');
  const pythonExecutable = process.env.PYTHON_EXECUTABLE || (fs.existsSync(venvPython) ? venvPython : (process.platform === 'win32' ? 'python' : 'python3'));
  const outputDirectory = path.join(MODEL_ROOT, 'outputs', jobId);
  const args = [
    'main.py',
    '--videos',
    ...videoPaths,
    '--output',
    outputDirectory,
    '--confidence',
    String(Number(config.confidenceThreshold ?? 0.6)),
    '--no-display',
  ];

  if (parseBoolean(config.reidEnabled ?? true)) {
    args.push('--reid');
  }

  const similarityThreshold = Number(config.similarityThreshold ?? 0.75);
  if (Number.isFinite(similarityThreshold)) {
    args.push('--similarity-threshold', String(similarityThreshold));
  }

  const reidInterval = Number(config.reidInterval ?? 15);
  if (Number.isFinite(reidInterval)) {
    args.push('--reid-interval', String(reidInterval));
  }

  const progressTimer = setInterval(() => {
    if (job.status !== 'processing') return;
    if (job.progress < 90) {
      job.progress = Math.min(job.progress + 5, 90);
      job.updatedAt = new Date().toISOString();
    }
  }, 1500);

  pushLog(job, `Launching Python: ${pythonExecutable} ${args.join(' ')}`);

  const child = spawn(pythonExecutable, args, {
    cwd: MODEL_ROOT,
    env: {
      ...process.env,
      PYTHONPATH: `${MODEL_ROOT}`,
    },
    shell: false,
  });

  child.stdout.on('data', (chunk: Buffer) => {
    const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      pushLog(job, line.trim());
    }
  });

  child.stderr.on('data', (chunk: Buffer) => {
    const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      pushLog(job, `[stderr] ${line.trim()}`);
    }
  });

  child.on('error', (error) => {
    clearInterval(progressTimer);
    job.status = 'failed';
    job.progress = 0;
    job.updatedAt = new Date().toISOString();
    job.errorMessage = error.message;
    pushLog(job, `Process failed: ${error.message}`);
  });

  child.on('close', (code) => {
    clearInterval(progressTimer);

    if (code === 0) {
      job.status = 'completed';
      job.progress = 100;
      job.updatedAt = new Date().toISOString();
      job.outputDirectory = outputDirectory;
      pushLog(job, 'Python pipeline completed successfully.');
      writeResultManifest(jobId, outputDirectory, job);
    } else {
      job.status = 'failed';
      job.progress = 0;
      job.updatedAt = new Date().toISOString();
      job.errorMessage = `Python process exited with code ${code}`;
      pushLog(job, `Python process exited with code ${code}`);
    }
  });
}

// Health check
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', message: 'Re-ID API Backend healthy', uptime: process.uptime() });
});

// Submit a job
app.post('/api/process', upload.array('videos'), (req, res) => {
  const files = (req.files as Express.Multer.File[]) || [];
  const config = req.body as Record<string, unknown>;

  if (files.length === 0) {
    return res.status(400).json({ error: 'Please upload at least one video file.' });
  }

  const jobId = uuidv4();
  const uploadedPaths = files.map((file) => path.join(uploadsDir, file.filename));
  jobs[jobId] = {
    id: jobId,
    status: 'pending',
    progress: 0,
    files: files.map((file) => file.filename),
    config,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    logs: ['Job submitted successfully.', 'Waiting to start processing...'],
  };

  startRealProcessing(jobId, uploadedPaths, config);

  res.json({ job_id: jobId, message: 'Processing started' });
});

// Get all jobs
app.get('/api/jobs', (_req, res) => {
  const allJobs = Object.values(jobs).sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  res.json(allJobs);
});

// Get job details
app.get('/api/jobs/:job_id', (req, res) => {
  const job = jobs[req.params.job_id];
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json(job);
});

// Serve results statically
app.use('/results', express.static(resultsDir));

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log('');
    console.log('========================================');
    console.log('SENTINEL Multi-Cam Re-ID is running');
    console.log(`\x1b[33mDashboard: http://localhost:${PORT}\x1b[0m`);
    console.log(`\x1b[33mHealth Check: http://localhost:${PORT}/api/health\x1b[0m`);
    console.log('========================================');
    console.log('');
  });
}

startServer();
