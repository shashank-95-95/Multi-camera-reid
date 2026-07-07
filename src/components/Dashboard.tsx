import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Activity, Server, Clock, Cpu } from 'lucide-react';
import { HealthCheck } from '../types';

export default function Dashboard() {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-sm font-semibold flex items-center justify-between text-zinc-200">
        Dashboard Overview
      </h2>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle>Backend API</CardTitle>
            <Activity className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-mono text-zinc-200">
              {loading ? 'Checking...' : health?.status === 'ok' ? 'ONLINE' : 'OFFLINE'}
            </div>
            <p className="text-[10px] text-zinc-500 mt-2 uppercase tracking-wide">
              {health?.message || 'Unable to connect to Re-ID pipeline'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle>Uptime</CardTitle>
            <Clock className="h-4 w-4 text-zinc-500" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-mono text-zinc-200">
              {health?.uptime ? `${Math.floor(health.uptime / 60)}m ${Math.floor(health.uptime % 60)}s` : '-'}
            </div>
            <p className="text-[10px] text-zinc-500 mt-2 uppercase tracking-wide">Server running time</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle>Pipeline Status</CardTitle>
            <Cpu className="h-4 w-4 text-zinc-500" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-mono text-zinc-200">READY</div>
            <p className="text-[10px] text-zinc-500 mt-2 uppercase tracking-wide">Accepting new jobs</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle>Storage</CardTitle>
            <Server className="h-4 w-4 text-zinc-500" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-mono text-zinc-200">MOUNTED</div>
            <p className="text-[10px] text-zinc-500 mt-2 uppercase tracking-wide">/uploads and /results available</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
