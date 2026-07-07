/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import Dashboard from './components/Dashboard';
import JobSubmit from './components/JobSubmit';
import JobsList from './components/JobsList';
import JobDetail from './components/JobDetail';
import { Camera, LayoutDashboard, Plus, ListVideo } from 'lucide-react';
import { Button } from './components/ui/button';

type View = 'dashboard' | 'submit' | 'jobs' | 'detail';

export default function App() {
  const [currentView, setCurrentView] = useState<View>('dashboard');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const navigateTo = (view: View, jobId?: string) => {
    setCurrentView(view);
    if (jobId) {
      setSelectedJobId(jobId);
    }
  };

  const NavItem = ({ view, label }: { view: View, label: string }) => {
    const isActive = currentView === view || (view === 'jobs' && currentView === 'detail');
    return (
      <button
        onClick={() => navigateTo(view)}
        className={`flex items-center w-full gap-3 px-6 py-3 text-sm transition-colors text-left ${
          isActive 
            ? 'text-zinc-200 active-nav' 
            : 'text-zinc-400 hover:bg-white/5 border-l-[3px] border-transparent'
        }`}
      >
        <span>{label}</span>
      </button>
    );
  };

  return (
    <div className="flex flex-col h-screen">
      {/* App Header */}
      <header className="h-16 border-b border-white/10 flex items-center justify-between px-8 bg-zinc-950 flex-shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center font-bold text-white">S</div>
          <h1 className="text-lg font-semibold tracking-tight text-white">SENTINEL <span className="text-blue-500 uppercase text-xs ml-1 tracking-widest font-bold">Multi-Cam RE-ID</span></h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="status-dot status-online"></span>
            <span className="text-xs text-zinc-400 font-medium">SYSTEM: ONLINE</span>
          </div>
          <div className="h-8 w-[1px] bg-white/10"></div>
          <div className="text-xs text-zinc-500 font-mono">v1.2.0-stable</div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Navigation Sidebar */}
        <aside className="w-64 border-r border-white/5 bg-zinc-950 flex flex-col flex-shrink-0">
          <nav className="flex-1 pt-6">
            <div className="px-4 mb-2 text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Main Menu</div>
            <NavItem view="dashboard" label="Dashboard" />
            <NavItem view="submit" label="Job Submissions" />
            <NavItem view="jobs" label="Job History" />
          </nav>
          <div className="p-6 border-t border-white/5 bg-zinc-900/30">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] font-bold text-zinc-500">GPU LOAD</span>
              <span className="text-[10px] text-blue-400">42%</span>
            </div>
            <div className="w-full bg-zinc-800 h-1 rounded-full">
              <div className="bg-blue-500 h-1 rounded-full" style={{ width: '42%' }}></div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 p-8 overflow-y-auto custom-scrollbar">
          <div className="max-w-6xl mx-auto space-y-8">
            {currentView === 'dashboard' && <Dashboard />}
            {currentView === 'submit' && (
              <JobSubmit onJobSubmitted={(jobId) => navigateTo('detail', jobId)} />
            )}
            {currentView === 'jobs' && (
              <JobsList onViewJob={(jobId) => navigateTo('detail', jobId)} />
            )}
            {currentView === 'detail' && selectedJobId && (
              <JobDetail 
                jobId={selectedJobId} 
                onBack={() => navigateTo('jobs')} 
              />
            )}
          </div>
        </main>
      </div>

      {/* Sticky Footer */}
      <footer className="h-8 bg-zinc-950 border-t border-white/10 flex items-center justify-between px-6 flex-shrink-0">
        <div className="flex items-center gap-4 text-[10px] text-zinc-500">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span> API READY
          </span>
          <span>DB: IN-MEMORY</span>
          <span>LOGS: STREAMING</span>
        </div>
        <div className="text-[10px] text-zinc-600">
          SESSION EXPIRES IN 2h 44m
        </div>
      </footer>
    </div>
  );
}

