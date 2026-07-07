import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { UploadCloud, FileVideo, X } from 'lucide-react';

export default function JobSubmit({ onJobSubmitted }: { onJobSubmitted: (jobId: string) => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [confidence, setConfidence] = useState('0.5');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) {
      setError('Please upload at least one video file.');
      return;
    }
    
    setLoading(true);
    setError(null);

    const formData = new FormData();
    files.forEach((file) => formData.append('videos', file));
    formData.append('confidenceThreshold', confidence);

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Failed to submit job');
      
      const data = await res.json();
      onJobSubmitted(data.job_id);
    } catch (err: any) {
      setError(err.message || 'An error occurred during submission.');
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card>
        <CardHeader>
          <div className="text-sm font-semibold flex items-center justify-between text-zinc-200">
            New Re-ID Task
            <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 text-[10px] rounded border border-blue-500/20 font-mono">POST /process</span>
          </div>
          <CardDescription>
            Upload multi-camera video feeds for person re-identification.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="block text-[10px] font-bold text-zinc-500 uppercase mb-1.5">Source Videos</label>
              
              <div className="border border-dashed border-white/20 bg-black/20 rounded-lg p-8 text-center hover:bg-white/5 transition-colors">
                <input
                  type="file"
                  multiple
                  accept="video/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="video-upload"
                />
                <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center">
                  <UploadCloud className="h-10 w-10 text-zinc-500 mb-2" />
                  <span className="text-sm text-zinc-300 font-medium">Click to upload videos</span>
                  <span className="text-xs text-zinc-500 mt-1">MP4, AVI, MKV up to 500MB</span>
                </label>
              </div>

              {files.length > 0 && (
                <div className="mt-4 space-y-2">
                  {files.map((file, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-black/40 rounded-md border border-white/10">
                      <div className="flex items-center space-x-3">
                        <FileVideo className="h-4 w-4 text-blue-400" />
                        <span className="text-xs truncate max-w-[200px] sm:max-w-sm text-zinc-300">{file.name}</span>
                      </div>
                      <button type="button" onClick={() => removeFile(i)} className="text-zinc-500 hover:text-red-400 transition-colors">
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <label className="block text-[10px] font-bold text-zinc-500 uppercase mb-1.5">Min Confidence</label>
              <input
                type="number"
                min="0.1"
                max="0.9"
                step="0.1"
                value={confidence}
                onChange={(e) => setConfidence(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-blue-500/50"
              />
            </div>

            {error && <div className="text-[10px] text-red-400 bg-red-500/10 p-3 rounded border border-red-500/20 font-bold uppercase">{error}</div>}

            <Button type="submit" className="w-full h-10 mt-4" disabled={loading || files.length === 0}>
              {loading ? 'Submitting...' : 'Execute AI Pipeline'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
