import { useEffect, useMemo, useState } from 'react'

type Policy = 'run_now' | 'threshold' | 'forecast_greedy' | 'oracle'
type View = 'overview' | 'jobs' | 'compare'
type UploadKey = 'jobs' | 'grid_truth' | 'forecasts'
type UploadFiles = Record<UploadKey, File | null>
interface Metrics {
  total_kwh: number | null
  emissions_avg_kg: number | null
  emissions_marg_kg: number | null
  total_jobs: number | null
  completed_jobs: number | null
  unfinished_jobs: number | null
  deadline_misses: number | null
  deadline_observed_jobs: number | null
  mean_added_delay_hours: number | null
  p95_added_delay_hours: number | null
  gpu_utilisation: number | null
  run_now_avg_kg: number | null
  percent_saved_vs_run_now_avg_pct: number | null
  greedy_perfect_ci_avg_kg: number | null
}

interface Config {
  policies: Policy[]
  k_values: number[]
  capacities: Array<number | null>
  forecasts: string[]
  truths: string[]
  truth_zones: Record<string, string[]>
  local_ready: boolean
}

interface JobRow {
  job_id: string
  arrival: string | null
  start_time: string | null
  finish_time: string | null
  deadline: string | null
  deferrable: boolean
  deadline_observed: boolean
  deadline_missed: boolean
  infeasible_on_arrival: boolean
  start_wait_hours: number | null
  added_delay_hours: number | null
  busy_gpu_hours: number | null
}

interface TracePoint {
  time: string
  actual_ci: number
  predicted_ci: number
}

interface SimulationResult {
  metrics: Metrics
  jobs: JobRow[]
  trace: TracePoint[]
}

interface CompareRow extends Metrics {
  policy: Policy
  k: number
  gpus: string
}

const POLICIES: Array<{ id: Policy; name: string; description: string }> = [
  { id: 'forecast_greedy', name: 'Forecast greedy', description: 'Schedule work in lower-carbon forecast hours.' },
  { id: 'threshold', name: 'Threshold', description: 'Run below the trailing carbon median.' },
  { id: 'run_now', name: 'Run now', description: 'Start eligible jobs as capacity allows.' },
  { id: 'oracle', name: 'Perfect CI', description: 'Greedy policy with future actual intensity.' },
]

const POLICY_NAME: Record<Policy, string> = Object.fromEntries(
  POLICIES.map(({ id, name }) => [id, name]),
) as Record<Policy, string>

const initialUploads: UploadFiles = { jobs: null, grid_truth: null, forecasts: null }

function number(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

function date(value: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

async function request<T>(path: string, payload?: object): Promise<T> {
  const response = await fetch(path, payload ? {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  } : undefined)
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`)
  return body as T
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 44 44" fill="none">
        <path d="M10 27.5C14 18.5 21 12 34 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <path d="M13 34C23 33 31 26 34 16" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <circle cx="11" cy="28" r="3" fill="currentColor" />
        <circle cx="34" cy="10" r="3" fill="currentColor" />
      </svg>
    </span>
  )
}

function MetricCard({ label, value, unit, note, tone = '' }: {
  label: string; value: string; unit?: string; note: string; tone?: string
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}<span>{unit}</span></div>
      <div className="metric-note">{note}</div>
    </article>
  )
}

function EmissionsChart({ metrics, policy }: { metrics: Metrics; policy: Policy }) {
  const rows = [
    { label: 'Run now', value: metrics.run_now_avg_kg, color: 'gray' },
    { label: POLICY_NAME[policy], value: metrics.emissions_avg_kg, color: 'green' },
    { label: 'Perfect CI greedy', value: metrics.greedy_perfect_ci_avg_kg, color: 'blue' },
  ]
  const max = Math.max(1, ...rows.map((row) => row.value ?? 0))
  return (
    <div className="bar-chart" role="img" aria-label="Average emissions comparison in kilograms">
      {rows.map((row) => (
        <div className="bar-row" key={row.label}>
          <div className="bar-heading"><span>{row.label}</span><strong>{number(row.value)} kg</strong></div>
          <div className="bar-track"><div className={`bar-fill ${row.color}`} style={{ width: `${Math.max(0, ((row.value ?? 0) / max) * 100)}%` }} /></div>
        </div>
      ))}
    </div>
  )
}

function downloadSvg(id: string, filename: string): void {
  const original = document.getElementById(id)
  if (!(original instanceof SVGSVGElement)) return
  const graphic = original.cloneNode(true) as SVGSVGElement
  graphic.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const blob = new Blob([new XMLSerializer().serializeToString(graphic)], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function TraceChart({ points }: { points: TracePoint[] }) {
  const valid = points.filter((point) => Number.isFinite(point.actual_ci) && Number.isFinite(point.predicted_ci) && Number.isFinite(Date.parse(point.time)))
  if (!valid.length) return <p className="graph-empty">No matching actual and forecast hours are available.</p>
  const width = 640; const height = 250
  const left = 52; const right = 18; const top = 34; const bottom = 42
  const plotWidth = width - left - right; const plotHeight = height - top - bottom
  const times = valid.map((point) => Date.parse(point.time))
  const first = times[0]; const span = Math.max(1, times[times.length - 1] - first)
  const values = valid.flatMap((point) => [point.actual_ci, point.predicted_ci])
  const low = Math.floor(Math.min(...values) / 50) * 50
  const high = Math.max(low + 50, Math.ceil(Math.max(...values) / 50) * 50)
  const x = (time: number) => left + ((time - first) / span) * plotWidth
  const y = (value: number) => top + ((high - value) / (high - low)) * plotHeight
  const path = (field: 'actual_ci' | 'predicted_ci') => valid.map((point, index) => `${index ? 'L' : 'M'} ${x(times[index]).toFixed(1)} ${y(point[field]).toFixed(1)}`).join(' ')
  const dateLabel = (time: number) => new Date(time).toLocaleString('en-GB', { timeZone: 'UTC', month: 'short', day: 'numeric', hour: '2-digit' })
  return <>
    <svg id="carbon-intensity-graph" className="graph-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Hourly actual and one-hour-ahead forecast carbon intensity">
      <rect width={width} height={height} fill="white" />
      <line x1={left} x2={left + 16} y1="14" y2="14" stroke="#279b5e" strokeWidth="3" /><text x={left + 22} y="18" fill="#48654f" fontSize="11">Actual CI</text>
      <line x1={left + 110} x2={left + 126} y1="14" y2="14" stroke="#7195cf" strokeWidth="3" strokeDasharray="5 4" /><text x={left + 132} y="18" fill="#48654f" fontSize="11">One-hour forecast</text>
      <text x={width - right} y="18" textAnchor="end" fill="#748b79" fontSize="11">gCO₂/kWh</text>
      {[0, 1, 2, 3, 4].map((tick) => { const value = low + (high - low) * tick / 4; return <g key={tick}><line x1={left} x2={width - right} y1={y(value)} y2={y(value)} stroke="#e4ebe4" /><text x={left - 9} y={y(value) + 4} textAnchor="end" fill="#748b79" fontSize="11">{number(value, 0)}</text></g> })}
      {[0, 0.5, 1].map((fraction) => <text key={fraction} x={left + fraction * plotWidth} y={height - 12} textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'} fill="#748b79" fontSize="11">{dateLabel(first + span * fraction)}</text>)}
      <path d={path('actual_ci')} fill="none" stroke="#279b5e" strokeWidth="2.5" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      <path d={path('predicted_ci')} fill="none" stroke="#7195cf" strokeWidth="2" strokeLinejoin="round" strokeDasharray="5 4" vectorEffect="non-scaling-stroke" />
    </svg>
    <div className="graph-legend"><span><i className="legend-actual" />Actual CI</span><span><i className="legend-forecast" />One-hour-ahead forecast</span><span>gCO₂/kWh · UTC</span></div>
  </>
}

function WaitChart({ jobs }: { jobs: JobRow[] }) {
  const waits = jobs.map((job) => job.start_wait_hours).filter((value): value is number => value != null && Number.isFinite(value) && value >= 0)
  if (!waits.length) return <p className="graph-empty">No job start waits are available.</p>
  const buckets = [
    { label: '<1h', count: waits.filter((value) => value < 1).length },
    { label: '1–6h', count: waits.filter((value) => value >= 1 && value < 6).length },
    { label: '6–12h', count: waits.filter((value) => value >= 6 && value < 12).length },
    { label: '12–24h', count: waits.filter((value) => value >= 12 && value < 24).length },
    { label: '24h+', count: waits.filter((value) => value >= 24).length },
  ]
  const width = 640; const height = 250
  const left = 42; const right = 18; const top = 34; const bottom = 42
  const plotWidth = width - left - right; const plotHeight = height - top - bottom
  const max = Math.max(1, ...buckets.map((bucket) => bucket.count))
  const step = plotWidth / buckets.length; const barWidth = Math.min(68, step * 0.58)
  return <>
    <svg id="job-wait-graph" className="graph-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Number of jobs by start wait time">
      <rect width={width} height={height} fill="white" />
      <text x={left} y="18" fill="#748b79" fontSize="11">Job count by start wait</text>
      {[0, 0.5, 1].map((fraction) => <g key={fraction}><line x1={left} x2={width - right} y1={top + plotHeight * (1 - fraction)} y2={top + plotHeight * (1 - fraction)} stroke="#e4ebe4" /><text x={left - 8} y={top + plotHeight * (1 - fraction) + 4} textAnchor="end" fill="#748b79" fontSize="11">{number(max * fraction, 0)}</text></g>)}
      {buckets.map((bucket, index) => { const barHeight = bucket.count / max * plotHeight; const center = left + step * (index + 0.5); return <g key={bucket.label}><rect x={center - barWidth / 2} y={top + plotHeight - barHeight} width={barWidth} height={barHeight} rx="5" fill="#36a66a"><title>{bucket.count} jobs waited {bucket.label}</title></rect><text x={center} y={top + plotHeight - barHeight - 7} textAnchor="middle" fill="#315942" fontSize="11" fontWeight="700">{bucket.count}</text><text x={center} y={height - 12} textAnchor="middle" fill="#748b79" fontSize="11">{bucket.label}</text></g> })}
    </svg>
    <div className="graph-legend"><span>{waits.length} started jobs · arrival to first start</span></div>
  </>
}

function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [view, setView] = useState<View>('overview')
  const [source, setSource] = useState<'local' | 'upload'>('local')
  const [forecastFile, setForecastFile] = useState('')
  const [truthFile, setTruthFile] = useState('')
  const [zone, setZone] = useState('')
  const [uploadZone, setUploadZone] = useState('')
  const [alignJobsYear, setAlignJobsYear] = useState(false)
  const [uploads, setUploads] = useState<UploadFiles>(initialUploads)
  const [policy, setPolicy] = useState<Policy>('forecast_greedy')
  const [k, setK] = useState(24)
  const [capacity, setCapacity] = useState('64')
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [resultPolicy, setResultPolicy] = useState<Policy>('forecast_greedy')
  const [comparison, setComparison] = useState<CompareRow[] | null>(null)
  const [loading, setLoading] = useState<'simulate' | 'compare' | null>(null)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [jobFilter, setJobFilter] = useState<'all' | 'missed' | 'unfinished'>('all')
  const [shownJobs, setShownJobs] = useState(12)

  useEffect(() => {
    request<Config>('/api/config')
      .then((data) => {
        setConfig(data)
        setForecastFile(data.forecasts[0] || '')
        setTruthFile(data.truths[0] || '')
        setZone(data.truth_zones[data.truths[0]]?.[0] || '')
      })
      .catch((err: Error) => setError(`Cannot reach the Python API: ${err.message}`))
  }, [])

  useEffect(() => {
    setResult(null)
    setComparison(null)
  }, [source, forecastFile, truthFile, zone, uploadZone, alignJobsYear, uploads, policy, k, capacity])

  const ready = source === 'local'
    ? Boolean(config?.local_ready && forecastFile && truthFile)
    : Boolean(uploads.jobs && uploads.grid_truth && uploads.forecasts)

  async function payload() {
    const common = {
      source,
      policy,
      k,
      gpus: capacity === 'unlimited' ? null : Number(capacity),
      zone: source === 'local' ? zone || null : uploadZone.trim() || null,
      align_jobs_year: alignJobsYear,
    }
    if (source === 'local') return { ...common, truth_file: truthFile, forecast_file: forecastFile }
    if (!uploads.jobs || !uploads.grid_truth || !uploads.forecasts) {
      throw new Error('Choose all three CSV files first')
    }
    const [jobs, grid_truth, forecasts] = await Promise.all([
      uploads.jobs.text(), uploads.grid_truth.text(), uploads.forecasts.text(),
    ])
    return { ...common, uploads: { jobs, grid_truth, forecasts } }
  }

  async function simulate() {
    if (!ready || loading) return
    setLoading('simulate')
    setError('')
    try {
      const data = await request<SimulationResult>('/api/simulate', await payload())
      setResult(data)
      setResultPolicy(policy)
      setComparison(null)
      setView('overview')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Simulation failed')
    } finally {
      setLoading(null)
    }
  }

  async function compare() {
    if (!ready || loading) return
    setLoading('compare')
    setError('')
    try {
      const data = await request<{ rows: CompareRow[] }>('/api/compare', await payload())
      setComparison(data.rows)
      setView('compare')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setLoading(null)
    }
  }

  const filteredJobs = useMemo(() => (result?.jobs || []).filter((job) => {
    if (search && !job.job_id.toLowerCase().includes(search.toLowerCase())) return false
    if (jobFilter === 'missed' && !job.deadline_missed) return false
    if (jobFilter === 'unfinished' && job.finish_time) return false
    return true
  }), [result, search, jobFilter])

  const metrics = result?.metrics
  const saving = metrics?.percent_saved_vs_run_now_avg_pct
  const comparisonMax = Math.max(1, ...(comparison || []).map((row) => row.emissions_avg_kg || 0))

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><BrandMark /><div><strong>carbon<span>lab</span></strong><small>SCHEDULER STUDIO</small></div></div>
        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <button className={view === 'overview' ? 'active' : ''} onClick={() => setView('overview')}><span className="nav-glyph">◫</span> Overview</button>
          <button className={view === 'jobs' ? 'active' : ''} onClick={() => setView('jobs')}><span className="nav-glyph">≡</span> Job explorer</button>
          <button className={view === 'compare' ? 'active' : ''} onClick={() => setView('compare')}><span className="nav-glyph">▥</span> Compare policies</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="side-info"><span className="pulse" /> SIMULATION ENGINE<br /><strong>{config ? 'API connected' : 'Waiting for API'}</strong></div>
          <p>Explore the carbon impact of when training jobs run.</p>
          <div className="side-version">TRACE-DRIVEN RESEARCH TOOL <span>v1.0</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Carbon scheduler <span>/</span> <strong>{view === 'overview' ? 'Overview' : view === 'jobs' ? 'Job explorer' : 'Compare policies'}</strong></div>
          <div className={`connection-pill ${config ? 'online' : ''}`}><span /> {config ? 'Python API connected' : 'API offline'}</div>
        </header>

        <div className="content-wrap">
          <section className="hero">
            <div className="hero-copy">
              <div className="eyebrow"><span className="eyebrow-dot" /> CARBON-AWARE COMPUTE</div>
              <h1>Better timing.<br /><em>Lower impact.</em></h1>
              <p>Explore how forecast-informed scheduling changes emissions, delay, and GPU use across your training jobs.</p>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" />
              <div className="hero-center"><span>CO₂</span><strong>↓</strong></div>
              <span className="orbit-dot dot-one" /><span className="orbit-dot dot-two" /><span className="orbit-dot dot-three" />
            </div>
          </section>

          {error && <div className="error-banner" role="alert"><strong>Could not complete request.</strong> {error}</div>}

          <section className="section-heading"><div><span className="section-index">01 / CONFIGURE</span><h2>Simulation setup</h2></div><p>Choose a trace, policy, and available GPU capacity.</p></section>
          <div className="setup-grid">
            <div className="panel data-panel">
              <div className="panel-top"><div className="panel-icon">↗</div><div><h3>Input data</h3><p>Use your local traces or upload CSVs from your computer.</p></div></div>
              <div className="segmented" role="group" aria-label="Data source">
                <button className={source === 'local' ? 'selected' : ''} onClick={() => setSource('local')}>Local data folder</button>
                <button className={source === 'upload' ? 'selected' : ''} onClick={() => setSource('upload')}>Upload CSV files</button>
              </div>
              {source === 'local' ? (
                <div className="local-data-fields">
                  <div className="field-group">
                    <label htmlFor="truth">Grid truth trace</label>
                    <select id="truth" value={truthFile} onChange={(event) => { const name = event.target.value; setTruthFile(name); setZone(config?.truth_zones[name]?.[0] || '') }}>
                      {config?.truths.length ? config.truths.map((name) => <option key={name} value={name}>{name}</option>) : <option value="">No grid truth CSV found</option>}
                    </select>
                  </div>
                  <div className="field-group">
                    <label htmlFor="forecast">Forecast trace</label>
                    <select id="forecast" value={forecastFile} onChange={(event) => setForecastFile(event.target.value)}>
                      {config?.forecasts.length ? config.forecasts.map((name) => <option key={name} value={name}>{name}</option>) : <option value="">No forecast CSV found</option>}
                    </select>
                  </div>
                  {Boolean(config?.truth_zones[truthFile]?.length) && <div className="field-group">
                    <label htmlFor="zone">Grid zone</label>
                    <select id="zone" value={zone} onChange={(event) => setZone(event.target.value)}>
                      {config?.truth_zones[truthFile]?.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </div>}
                  <div className={`data-status ${config?.local_ready ? 'good' : 'warn'}`}><span />{config?.local_ready ? 'Jobs, grid truth, and forecast traces found in data/' : 'Add jobs, grid truth, and forecast CSVs to data/'}</div>
                </div>
              ) : (
                <div className="upload-list">
                  {(['jobs', 'grid_truth', 'forecasts'] as UploadKey[]).map((key) => (
                    <label className="upload-row" key={key}>
                      <span><strong>{key === 'grid_truth' ? 'Grid truth' : key === 'jobs' ? 'Jobs' : 'Forecasts'}</strong><small>{uploads[key]?.name || `${key}.csv`}</small></span>
                      <span className="upload-action">{uploads[key] ? 'Change' : 'Choose file'}</span>
                      <input type="file" accept=".csv,text/csv" onChange={(event) => setUploads((current) => ({ ...current, [key]: event.target.files?.[0] || null }))} />
                    </label>
                  ))}
                  <div className="field-group"><label htmlFor="upload-zone">Grid zone (for multi-zone uploads)</label><input id="upload-zone" value={uploadZone} onChange={(event) => setUploadZone(event.target.value)} placeholder="For example, IN-EA" /></div>
                  <p className="field-hint">Uploads are processed in memory and are not saved by the API.</p>
                </div>
              )}
              <label className="align-option"><input type="checkbox" checked={alignJobsYear} onChange={(event) => setAlignJobsYear(event.target.checked)} /><span>Align job arrival year to forecast year in memory<small>Use this when job dates and trace dates differ. The jobs CSV is not changed.</small></span></label>
            </div>

            <div className="panel scenario-panel">
              <div className="panel-top"><div className="panel-icon accent">◷</div><div><h3>Scenario</h3><p>Pick the scheduling rule and its operating limits.</p></div></div>
              <div className="scenario-fields">
                <div className="field-group policy-field"><label htmlFor="policy">Scheduling policy</label><select id="policy" value={policy} onChange={(event) => setPolicy(event.target.value as Policy)}>{POLICIES.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><span className="field-hint">{POLICIES.find((item) => item.id === policy)?.description}</span></div>
                <div className="field-group"><label htmlFor="window">Deferral window</label><select id="window" value={k} onChange={(event) => setK(Number(event.target.value))}>{(config?.k_values || [6, 12, 24, 48]).map((value) => <option key={value} value={value}>{value} hours</option>)}</select></div>
                <div className="field-group"><label htmlFor="capacity">GPU capacity</label><select id="capacity" value={capacity} onChange={(event) => setCapacity(event.target.value)}>{(config?.capacities || [64, 32, 128, null]).map((value) => <option key={value ?? 'unlimited'} value={value ?? 'unlimited'}>{value == null ? 'Unlimited' : `${value} GPUs`}</option>)}</select></div>
              </div>
              <div className="scenario-actions"><button className="primary-button" disabled={!ready || Boolean(loading)} onClick={simulate}>{loading === 'simulate' ? 'Running simulation…' : 'Run simulation'} <span>→</span></button><button className="secondary-button" disabled={!ready || Boolean(loading)} onClick={compare}>{loading === 'compare' ? 'Comparing…' : 'Compare all policies'}</button></div>
            </div>
          </div>

          {view === 'overview' && <>
            <section className="section-heading results-heading"><div><span className="section-index">02 / ANALYZE</span><h2>Simulation results</h2></div><p>{result ? `${POLICY_NAME[resultPolicy]} · ${number(metrics?.total_jobs, 0)} jobs analyzed` : 'Results appear after you run a scenario.'}</p></section>
            {result && metrics ? <>
              <div className="metrics-grid">
                <MetricCard label="Average emissions" value={number(metrics.emissions_avg_kg)} unit="kg CO₂" note="Actual carbon intensity" tone="metric-featured" />
                <MetricCard label="Saved vs. run now" value={number(saving)} unit="%" note={saving == null ? 'Requires completed runs' : saving >= 0 ? 'Lower emissions than baseline' : 'Higher emissions than baseline'} tone={saving != null && saving < 0 ? 'metric-negative' : ''} />
                <MetricCard label="Jobs completed" value={number(metrics.completed_jobs, 0)} unit={`/ ${number(metrics.total_jobs, 0)}`} note={`${number(metrics.unfinished_jobs, 0)} unfinished jobs`} />
                <MetricCard label="Deadline misses" value={number(metrics.deadline_misses, 0)} unit="jobs" note={`${number(metrics.deadline_observed_jobs, 0)} observed deadlines`} />
              </div>
              <div className="analysis-grid">
                <div className="panel chart-panel"><div className="panel-header"><div><span className="section-index">EMISSIONS</span><h3>Carbon comparison</h3></div><span className="panel-badge">Actual CI · kg CO₂</span></div><EmissionsChart metrics={metrics} policy={resultPolicy} /><p className="chart-caption">Perfect CI greedy is a foresight benchmark, not a globally optimal schedule.</p></div>
                <div className="panel detail-panel"><div className="panel-header"><div><span className="section-index">OPERATIONS</span><h3>Workload outcomes</h3></div></div><div className="detail-list"><div><span>Mean added delay</span><strong>{number(metrics.mean_added_delay_hours)} h</strong></div><div><span>95th percentile delay</span><strong>{number(metrics.p95_added_delay_hours)} h</strong></div><div><span>GPU utilisation</span><strong>{metrics.gpu_utilisation == null ? '—' : `${number(metrics.gpu_utilisation * 100)}%`}</strong></div><div><span>Total energy</span><strong>{number(metrics.total_kwh)} kWh</strong></div><div><span>Marginal emissions</span><strong>{number(metrics.emissions_marg_kg)} kg</strong></div></div></div>
              </div>
              <div className="graphs-grid">
                <div className="panel graph-panel"><div className="panel-header"><div><span className="section-index">GRID SIGNAL</span><h3>Carbon intensity over time</h3></div><button className="graph-export" onClick={() => downloadSvg('carbon-intensity-graph', 'carbon-intensity.svg')} disabled={!result.trace.length}>Download SVG</button></div><TraceChart points={result.trace} /><p className="chart-caption">The forecast shown for each hour was issued one hour earlier. Values are sampled when the period exceeds 240 hours.</p></div>
                <div className="panel graph-panel"><div className="panel-header"><div><span className="section-index">JOB DELAY</span><h3>Start wait distribution</h3></div><button className="graph-export" onClick={() => downloadSvg('job-wait-graph', 'job-start-waits.svg')} disabled={!result.jobs.some((job) => job.start_wait_hours != null)}>Download SVG</button></div><WaitChart jobs={result.jobs} /><p className="chart-caption">Counts include jobs that started; jobs without a start time are excluded.</p></div>
              </div>
              <div className="next-action"><div><strong>See what happened to each job</strong><p>Inspect arrival, start, finish, wait time, and deadline outcomes.</p></div><button onClick={() => setView('jobs')}>Open job explorer <span>→</span></button></div>
            </> : <div className="empty-panel"><div className="empty-symbol">↗</div><h3>Your results will appear here</h3><p>{ready ? 'Run a simulation to see emissions, job outcomes, and GPU utilisation.' : 'Add the required CSV inputs above, then run a simulation.'}</p></div>}
          </>}

          {view === 'jobs' && <>
            <section className="section-heading results-heading"><div><span className="section-index">02 / EXPLORE</span><h2>Job explorer</h2></div><p>Follow each job from arrival to completion.</p></section>
            {result ? <div className="panel jobs-panel"><div className="table-toolbar"><div><h3>Per-job outcomes</h3><p>{filteredJobs.length} of {result.jobs.length} jobs</p></div><div className="table-controls"><input aria-label="Search job ID" placeholder="Search job ID…" value={search} onChange={(event) => { setSearch(event.target.value); setShownJobs(12) }} /><select aria-label="Filter jobs" value={jobFilter} onChange={(event) => { setJobFilter(event.target.value as typeof jobFilter); setShownJobs(12) }}><option value="all">All jobs</option><option value="missed">Deadline missed</option><option value="unfinished">Unfinished</option></select></div></div><div className="table-scroll"><table><thead><tr><th>Job ID</th><th>Arrival</th><th>Start</th><th>Finish</th><th>Wait</th><th>GPU hours</th><th>Status</th></tr></thead><tbody>{filteredJobs.slice(0, shownJobs).map((job) => <tr key={job.job_id}><td className="mono strong">{job.job_id}</td><td>{date(job.arrival)}</td><td>{date(job.start_time)}</td><td>{date(job.finish_time)}</td><td>{number(job.start_wait_hours)} h</td><td>{number(job.busy_gpu_hours)}</td><td><span className={`status-tag ${job.deadline_missed ? 'bad' : !job.finish_time ? 'waiting' : 'good'}`}>{job.deadline_missed ? 'Missed' : !job.finish_time ? 'Unfinished' : 'Completed'}</span></td></tr>)}</tbody></table>{filteredJobs.length === 0 && <div className="table-empty">No jobs match your filter.</div>}</div>{filteredJobs.length > shownJobs && <button className="load-more" onClick={() => setShownJobs((count) => count + 24)}>Show more jobs ↓</button>}</div> : <div className="empty-panel"><div className="empty-symbol">≡</div><h3>No job results yet</h3><p>Run a simulation to inspect per-job timing and outcomes.</p><button className="secondary-button" onClick={() => setView('overview')}>Back to overview</button></div>}
          </>}

          {view === 'compare' && <>
            <section className="section-heading results-heading"><div><span className="section-index">02 / COMPARE</span><h2>Policy comparison</h2></div><p>Same trace, deferral window, and GPU capacity for every policy.</p></section>
            {comparison ? <div className="panel compare-panel"><div className="panel-header"><div><span className="section-index">AVERAGE EMISSIONS</span><h3>Side-by-side policies</h3></div><span className="panel-badge">Lower is better</span></div><div className="compare-bars">{comparison.map((row) => <div className="compare-item" key={row.policy}><div className="compare-label"><strong>{POLICY_NAME[row.policy]}</strong><span>{number(row.emissions_avg_kg)} kg CO₂</span></div><div className="bar-track"><div className={`bar-fill ${row.policy === 'run_now' ? 'gray' : row.policy === 'oracle' ? 'blue' : 'green'}`} style={{ width: `${Math.max(0, ((row.emissions_avg_kg || 0) / comparisonMax) * 100)}%` }} /></div><div className="compare-detail"><span>{number(row.percent_saved_vs_run_now_avg_pct)}% saved</span><span>{number(row.deadline_misses, 0)} deadline misses</span><span>{number(row.mean_added_delay_hours)} h mean delay</span></div></div>)}</div><p className="chart-caption">Savings show as unavailable when a run leaves jobs unfinished. Oracle uses actual future CI as a forecast benchmark.</p></div> : <div className="empty-panel"><div className="empty-symbol">▥</div><h3>No comparison yet</h3><p>Select your inputs, then choose “Compare all policies” above.</p></div>}
          </>}

          <footer className="page-footer"><span>Carbon Scheduler Studio</span><span>Simulation results are calculated by the Python scheduler and simulator.</span></footer>
        </div>
      </main>
    </div>
  )
}

export default App
