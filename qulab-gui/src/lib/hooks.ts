import { useEffect, useState } from 'react';
import { apiClient, LABS } from './api-client';

// Re-export for pages that import apiClient/LABS from this module (required by Figma Make bundler)
export { apiClient, LABS } from './api-client';

// Hook to check all labs health status
export function useLabsHealth() {
  const [labsStatus, setLabsStatus] = useState<Record<string, { status: string; healthy: boolean }>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await apiClient.getAllLabsStatus();
        setLabsStatus(status);
      } catch (error) {
        console.error('Failed to fetch labs status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();

    // Poll every 30 seconds
    const interval = setInterval(fetchStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  return { labsStatus, loading };
}

// Hook for specific lab health
export function useLabHealth(labKey: string) {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await apiClient.checkHealth(labKey);
        setHealth(data);
        setError(null);
      } catch (err: any) {
        setError(err.message);
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();

    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, [labKey]);

  return { health, loading, error };
}

// Hook for lab thresholds/constants
export function useLabThresholds(labKey: string) {
  const [thresholds, setThresholds] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchThresholds = async () => {
      try {
        const data = await apiClient.getThresholds(labKey);
        setThresholds(data);
        setError(null);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchThresholds();
  }, [labKey]);

  return { thresholds, loading, error };
}

// Hook for lab assessment
export function useLabAssessment(labKey: string) {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const assess = async (data: any) => {
    setLoading(true);
    setError(null);

    try {
      const assessmentResult = await apiClient.assess(labKey, data);
      setResult(assessmentResult);
      return assessmentResult;
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { result, loading, error, assess };
}

// Get all labs configuration
export function useLabsConfig() {
  return { labs: LABS };
}

// Demo data hooks for production visualizer pages (wire to APIs when available)
export function useGenomicsData() {
  const genomics = [
    {
      id: '1',
      status: 'processing',
      sample_id: 'GS-001',
      chromosome: 'chr1',
      gene_expression: 0.92,
      quality_score: 98.2,
    },
    {
      id: '2',
      status: 'completed',
      sample_id: 'GS-002',
      chromosome: 'chr17',
      gene_expression: 0.88,
      quality_score: 97.1,
    },
  ];
  return { genomics, loading: false };
}

export function useNeuroData() {
  const neuro = [
    {
      id: '1',
      status: 'processing',
      patient_id: 'P-101',
      brain_region: 'frontal',
      scan_type: 'fMRI',
      activity_level: 0.76,
    },
    {
      id: '2',
      status: 'completed',
      patient_id: 'P-102',
      brain_region: 'temporal',
      scan_type: 'EEG',
      activity_level: 0.91,
    },
  ];
  return { neuro, loading: false };
}

export function useSystemStatus() {
  const systems = [
    {
      id: 'core',
      status: 'operational',
      load: 42,
      region: 'NA',
      uptime: 99.9,
      latency: 12,
      node_count: 120,
    },
    {
      id: 'mesh',
      status: 'syncing',
      load: 78,
      region: 'EU',
      uptime: 98.4,
      latency: 24,
      node_count: 88,
    },
    {
      id: 'edge',
      status: 'operational',
      load: 31,
      region: 'APAC',
      uptime: 99.1,
      latency: 18,
      node_count: 64,
    },
  ];
  return { systems, loading: false };
}

export function useLiveMetrics() {
  const metrics = [
    { id: 'latency', metric_name: 'Latency', metric_value: 14, metric_type: 'ms' },
    { id: 'throughput', metric_name: 'Throughput', metric_value: 4029, metric_type: '/sec' },
  ];
  return { metrics, loading: false };
}
