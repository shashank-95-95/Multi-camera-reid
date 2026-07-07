import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Job } from '../types';
import { ArrowLeft, CheckCircle2, FileJson, Loader2, XCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

export default function JobDetail({ jobId, onBack }: { jobId: string, onBack: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchJob = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error('Job not found');
        const data = await res.json();
        setJob(data);
      } catch (err: any) {
        setError(err.message);
      }
    };

    fetchJob();
    const interval = setInterval(() => {
      if (job?.status !== 'completed' && job?.status !== 'failed') {
        fetchJob();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, job?.status]);

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-red-400 mb-4">{error}</div>
        <Button onClick={onBack} variant="outline">Go Back</Button>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-zinc-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-4">
        <Button variant="ghost" size="icon" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <div className="text-xs font-bold text-blue-500 uppercase tracking-widest mb-1">Active Process Details</div>
          <h2 className="text-xl font-mono text-zinc-200">JOB-{job.id.split('-')[0].toUpperCase()}</h2>
        </div>
        <div className="ml-auto text-right">
           <div className="text-[10px] text-zinc-500 uppercase mb-1">Started</div>
           <div className="text-sm font-mono text-zinc-300">{formatDistanceToNow(new Date(job.createdAt), { addSuffix: true })}</div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <div className="md:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Processing Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-zinc-400 font-medium uppercase tracking-wider">{job.status}</span>
                </div>
                <span className="text-xs text-zinc-400">{job.progress}% Complete</span>
              </div>
              
              <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden mb-6">
                <div 
                  className={`h-full transition-all duration-500 ${job.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'}`}
                  style={{ width: `${job.progress}%` }}
                />
              </div>

              <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar font-mono text-[10px] p-4 bg-black/50 rounded border border-white/5">
                {job.logs.map((log, i) => (
                  <div key={i} className="text-zinc-400">
                    <span className="text-zinc-600 mr-2">[{new Date(job.updatedAt).toLocaleTimeString()}]</span>
                    {log}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {job.status === 'completed' && job.resultUrl && (
            <Card>
              <CardHeader>
                <CardTitle>Results & Output</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between p-4 bg-black/40 rounded border border-white/10">
                  <div className="flex items-center space-x-3">
                    <FileJson className="h-5 w-5 text-green-500" />
                    <div>
                      <div className="text-xs font-mono text-zinc-200">tracking_results.json</div>
                      <div className="text-[10px] text-zinc-500 uppercase mt-1">Cross-camera embeddings and bounding boxes</div>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={job.resultUrl} target="_blank" rel="noreferrer">
                      Download
                    </a>
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-xs">
              <div>
                <div className="text-[10px] font-bold text-zinc-500 uppercase mb-1.5">Inputs</div>
                <div className="space-y-1">
                  {job.files.map((file, i) => (
                    <div key={i} className="truncate text-zinc-400 bg-white/5 px-2 py-1.5 rounded text-[10px] font-mono border border-white/5">
                      {file}
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <div className="text-[10px] font-bold text-zinc-500 uppercase mb-1.5">Min Confidence</div>
                <div className="text-zinc-300 font-mono bg-white/5 px-2 py-1.5 rounded inline-block border border-white/5">
                  {job.config.confidenceThreshold || '0.5'}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
