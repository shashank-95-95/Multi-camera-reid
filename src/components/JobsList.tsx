import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Job } from '../types';
import { formatDistanceToNow } from 'date-fns';
import { PlayCircle, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { Button } from './ui/button';

export default function JobsList({ onViewJob }: { onViewJob: (jobId: string) => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch('/api/jobs');
        if (res.ok) {
          const data = await res.json();
          setJobs(data);
        }
      } finally {
        setLoading(false);
      }
    };
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent History</CardTitle>
      </CardHeader>
      <CardContent>
        {loading && jobs.length === 0 ? (
          <div className="py-8 text-center text-zinc-500 text-sm">Loading jobs...</div>
        ) : jobs.length === 0 ? (
          <div className="py-12 text-center text-zinc-500 text-sm">
            No jobs found. Submit a new video to get started.
          </div>
        ) : (
          <div className="flex-1 overflow-x-auto custom-scrollbar">
            <table className="w-full text-left text-xs border-separate border-spacing-y-2">
              <thead>
                <tr className="text-zinc-500">
                  <th className="font-medium pb-2 pl-2">ID</th>
                  <th className="font-medium pb-2">STATUS</th>
                  <th className="font-medium pb-2">FILES</th>
                  <th className="font-medium pb-2 text-right pr-2">ACTIONS</th>
                </tr>
              </thead>
              <tbody className="text-zinc-300">
                {jobs.map((job) => (
                  <tr key={job.id} className="hover:bg-white/5 transition-colors group">
                    <td className="py-3 pl-2 font-mono">#{job.id.split('-')[0]}</td>
                    <td className="py-3">
                      {job.status === 'completed' && <span className="px-2 py-1 bg-green-500/10 text-green-500 rounded text-[10px] font-bold uppercase">SUCCESS</span>}
                      {job.status === 'failed' && <span className="px-2 py-1 bg-red-500/10 text-red-500 rounded text-[10px] font-bold uppercase">FAILED</span>}
                      {job.status === 'processing' && <span className="px-2 py-1 bg-blue-500/10 text-blue-400 rounded text-[10px] font-bold uppercase">PROCESSING ({job.progress}%)</span>}
                      {job.status === 'pending' && <span className="px-2 py-1 bg-zinc-500/10 text-zinc-400 rounded text-[10px] font-bold uppercase">PENDING</span>}
                    </td>
                    <td className="py-3 text-zinc-400">{job.files.length} video(s)</td>
                    <td className="py-3 text-right pr-2">
                       <Button variant="outline" size="sm" onClick={() => onViewJob(job.id)} className="h-7 text-[10px]">
                         View
                       </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
