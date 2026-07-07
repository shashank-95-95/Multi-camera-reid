export interface Job {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  files: string[];
  config: Record<string, any>;
  createdAt: string;
  updatedAt: string;
  resultUrl?: string;
  logs: string[];
}

export interface HealthCheck {
  status: string;
  message: string;
  uptime?: number;
}
