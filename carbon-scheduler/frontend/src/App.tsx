import { useEffect, useMemo, useState } from 'react'
import { parseCsv, toSummaryRows, toJobRows, jobFileName, type CompareRow, type JobRow, type Metrics, type Policy } from './results'

type View = 'overview' | 'jobs' | 'compare'

const POLICIES: Array<{ id: Policy; name: string; description: string }> = [
  { id: 'forecast_greedy', name: 'Forecast greedy', description: 'Schedule work in lower-carbon forecast hours.' },
  { id: 'threshold', name: 'Threshold', description: 'Run below the trailing carbon median.' },
  { id: 'run_now', name: 'Run now', description: 'Start eligible jobs as capacity allows.' },
  { id: 'oracle', name: 'Perfect CI', description: 'Greedy policy with future actual intensity.' },
]
const COMPARE_ORDER: Policy[] = ['run_now', 'threshold', 'forecast_greedy', 'oracle']

const POLICY_NAME: Record<Policy, string> = Object.fromEntries(
  POLICIES.map(({ id, name }) => [id, name]),
) as Record<Policy, string>

function number(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

function date(value: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value.replace(' ', 'T'))
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

async function fetchCsv(name: string): Promise<string> {
  const response = await fetch(`/api/results/${encodeURIComponent(name)}`, { cache: 'no-store' })
  if (!response.ok) throw new Error(`${name} not found in results/ (HTTP ${response.status})`)
  return response.text()
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

function SavedFiles({ files }: { files: Record<string, string> }) {
  return <div className="panel output-panel"><div><span className="section-index">SAVED OUTPUT</span><h3>CSV tables and PNG graphs</h3><p>Saved in carbon-scheduler/results/</p></div><div className="output-links">{Object.entries(files).map(([kind, name]) => <a key={kind} href={`/api/results/${encodeURIComponent(name)}`} download>{kind.replace(/_/g, ' ')} <span>{name}</span></a>)}</div></div>
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

function SavingsChart({ rows, ks }: { rows: CompareRow[]; ks: number[] }) {
  const lines: Array<{ policy: Policy; color: string; dash?: string }> = [
    { policy: 'forecast_greedy', color: '#279b5e' },
    { policy: 'threshold', color: '#c8923a' },
    { policy: 'oracle', color: '#7195cf', dash: '5 4' },
  ]
  const points = lines.map((line) => ({
    ...line,
    values: ks.map((k) => rows.find((row) => row.policy === line.policy && row.k === k)?.percent_saved_vs_run_now_avg_pct ?? null),
  }))
  const all = points.flatMap((line) => line.values).filter((value): value is number => value != null && Number.isFinite(value))
  if (!all.length || ks.length < 2) return <p className="graph-empty">No comparable savings values for this zone and capacity.</p>
  const width = 640; const height = 250
  const left = 52; const right = 18; const top = 34; const bottom = 42
  const plotWidth = width - left - right; const plotHeight = height - top - bottom
  const low = Math.min(0, Math.floor(Math.min(...all) / 5) * 5)
  const high = Math.max(low + 5, Math.ceil(Math.max(...all) / 5) * 5)
  const x = (index: number) => left + (index / (ks.length - 1)) * plotWidth
  const y = (value: number) => top + ((high - value) / (high - low)) * plotHeight
  return <>
    <svg className="graph-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Percent carbon saved versus deferral window">
      <rect width={width} height={height} fill="white" />
      <text x={width - right} y="18" textAnchor="end" fill="#748b79" fontSize="11">% saved vs run now</text>
      {[0, 1, 2, 3, 4].map((tick) => { const value = low + (high - low) * tick / 4; return <g key={tick}><line x1={left} x2={width - right} y1={y(value)} y2={y(value)} stroke="#e4ebe4" /><text x={left - 9} y={y(value) + 4} textAnchor="end" fill="#748b79" fontSize="11">{number(value, 0)}</text></g> })}
      {ks.map((k, index) => <text key={k} x={x(index)} y={height - 12} textAnchor="middle" fill="#748b79" fontSize="11">{k}h</text>)}
      {points.map((line) => {
        const segments = line.values.map((value, index) => value == null || !Number.isFinite(value) ? null : { cx: x(index), cy: y(value) })
        const path = segments.map((point, index) => point ? `${segments.slice(0, index).some(Boolean) ? 'L' : 'M'} ${point.cx.toFixed(1)} ${point.cy.toFixed(1)}` : '').join(' ')
        return <g key={line.policy}><path d={path} fill="none" stroke={line.color} strokeWidth="2.5" strokeDasharray={line.dash} strokeLinejoin="round" />{segments.map((point, index) => point && <circle key={index} cx={point.cx} cy={point.cy} r="3.5" fill={line.color} />)}</g>
      })}
    </svg>
    <div className="graph-legend">{lines.map((line) => <span key={line.policy}><i style={{ background: line.color }} />{POLICY_NAME[line.policy]}</span>)}<span>x-axis: deferral window k</span></div>
  </>
}

function App() {
  const [summary, setSummary] = useState<CompareRow[] | null>(null)
  const [view, setView] = useState<View>('overview')
  const [zone, setZone] = useState('')
  const [policy, setPolicy] = useState<Policy>('forecast_greedy')
  const [k, setK] = useState(24)
  const [capacity, setCapacity] = useState('64')
  const [jobs, setJobs] = useState<JobRow[] | null>(null)
  const [jobsError, setJobsError] = useState('')
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [jobFilter, setJobFilter] = useState<'all' | 'missed' | 'unfinished'>('all')
  const [shownJobs, setShownJobs] = useState(12)

  useEffect(() => {
    fetchCsv('summary.csv')
      .then((text) => {
        const rows = toSummaryRows(parseCsv(text))
        if (!rows.length) throw new Error('summary.csv has no runs. Run: python run.py sweep ...')
        setSummary(rows)
        const first = rows[0]
        setZone(first.zone)
        setK(rows.some((row) => row.k === 24) ? 24 : first.k)
        setCapacity(rows.some((row) => row.gpus === '64') ? '64' : first.gpus)
      })
      .catch((err: Error) => setError(`Cannot load results/summary.csv: ${err.message}. Start python api.py from carbon-scheduler/ after running the sweep.`))
  }, [])

  const zones = useMemo(() => [...new Set((summary || []).map((row) => row.zone))].filter(Boolean).sort(), [summary])
  const ks = useMemo(() => [...new Set((summary || []).map((row) => row.k))].sort((a, b) => a - b), [summary])
  const capacities = useMemo(() => [...new Set((summary || []).map((row) => row.gpus))].sort((a, b) => (a === 'unlimited' ? 1 : b === 'unlimited' ? -1 : Number(a) - Number(b))), [summary])

  const selected = useMemo(() => (summary || []).find((row) => row.zone === zone && row.policy === policy && row.k === k && row.gpus === capacity) || null, [summary, zone, policy, k, capacity])
  const comparison = useMemo(() => COMPARE_ORDER.map((id) => (summary || []).find((row) => row.zone === zone && row.policy === id && row.k === k && row.gpus === capacity)).filter((row): row is CompareRow => Boolean(row)), [summary, zone, k, capacity])
  const zoneCapacityRows = useMemo(() => (summary || []).filter((row) => row.zone === zone && row.gpus === capacity), [summary, zone, capacity])

  const jobFile = selected ? jobFileName(selected) : ''
  useEffect(() => {
    setJobs(null); setJobsError(''); setSearch(''); setJobFilter('all'); setShownJobs(12)
    if (!jobFile) return
    let cancelled = false
    fetchCsv(jobFile)
      .then((text) => { if (!cancelled) setJobs(toJobRows(parseCsv(text))) })
      .catch((err: Error) => { if (!cancelled) setJobsError(err.message) })
    return () => { cancelled = true }
  }, [jobFile])

  const filteredJobs = useMemo(() => (jobs || []).filter((job) => {
    if (search && !job.job_id.toLowerCase().includes(search.toLowerCase())) return false
    if (jobFilter === 'missed' && !job.deadline_missed) return false
    if (jobFilter === 'unfinished' && job.finish_time) return false
    return true
  }), [jobs, search, jobFilter])

  const metrics: Metrics | null = selected
  const saving = metrics?.percent_saved_vs_run_now_avg_pct
  const comparisonMax = Math.max(1, ...comparison.map((row) => row.emissions_avg_kg || 0))
  const files: Record<string, string> = selected ? { summary: 'summary.csv', per_job: jobFile } : {}
  const loaded = Boolean(summary)

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
          <div className="side-info"><span className="pulse" /> RESULTS FILES<br /><strong>{loaded ? `${summary?.length} runs loaded` : 'Waiting for results'}</strong></div>
          <p>Explore the carbon impact of when training jobs run.</p>
          <div className="side-version">TRACE-DRIVEN RESEARCH TOOL <span>v2.0</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Carbon scheduler <span>/</span> <strong>{view === 'overview' ? 'Overview' : view === 'jobs' ? 'Job explorer' : 'Compare policies'}</strong></div>
          <div className={`connection-pill ${loaded ? 'online' : ''}`}><span /> {loaded ? 'Reading saved sweep results' : 'Results not loaded'}</div>
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

          {error && <div className="error-banner" role="alert"><strong>Could not load results.</strong> {error}</div>}

          <section className="section-heading"><div><span className="section-index">01 / SELECT</span><h2>Choose a sweep result</h2></div><p>Every combination below comes from the saved sweep. Nothing is re-simulated.</p></section>
          <div className="setup-grid">
            <div className="panel data-panel">
              <div className="panel-top"><div className="panel-icon">↗</div><div><h3>Saved results</h3><p>Read from results/summary.csv and the per-job CSV files.</p></div></div>
              <div className="local-data-fields">
                {zones.length > 0 && <div className="field-group">
                  <label htmlFor="zone">Grid zone</label>
                  <select id="zone" value={zone} onChange={(event) => setZone(event.target.value)}>{zones.map((name) => <option key={name} value={name}>{name}</option>)}</select>
                </div>}
                <div className={`data-status ${loaded ? 'good' : 'warn'}`}><span />{loaded ? `${summary?.length} runs found in summary.csv` : 'Run the sweep to create results/summary.csv'}</div>
              </div>
            </div>

            <div className="panel scenario-panel">
              <div className="panel-top"><div className="panel-icon accent">◷</div><div><h3>Scenario</h3><p>Pick the scheduling rule and its operating limits.</p></div></div>
              <div className="scenario-fields">
                <div className="field-group policy-field"><label htmlFor="policy">Scheduling policy</label><select id="policy" value={policy} onChange={(event) => setPolicy(event.target.value as Policy)}>{POLICIES.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><span className="field-hint">{POLICIES.find((item) => item.id === policy)?.description}</span></div>
                <div className="field-group"><label htmlFor="window">Deferral window</label><select id="window" value={k} onChange={(event) => setK(Number(event.target.value))}>{ks.map((value) => <option key={value} value={value}>{value} hours</option>)}</select></div>
                <div className="field-group"><label htmlFor="capacity">GPU capacity</label><select id="capacity" value={capacity} onChange={(event) => setCapacity(event.target.value)}>{capacities.map((value) => <option key={value} value={value}>{value === 'unlimited' ? 'Unlimited' : `${value} GPUs`}</option>)}</select></div>
              </div>
            </div>
          </div>

          {view === 'overview' && <>
            <section className="section-heading results-heading"><div><span className="section-index">02 / ANALYZE</span><h2>Simulation results</h2></div><p>{selected ? `${POLICY_NAME[policy]} · ${zone ? `${zone} · ` : ''}${number(metrics?.total_jobs, 0)} jobs analyzed` : 'Pick a combination above.'}</p></section>
            {selected && metrics ? <>
              <div className="metrics-grid">
                <MetricCard label="Average emissions" value={number(metrics.emissions_avg_kg)} unit="kg CO₂" note="Actual carbon intensity" tone="metric-featured" />
                <MetricCard label="Saved vs. run now" value={number(saving)} unit="%" note={saving == null ? 'Requires completed runs' : saving >= 0 ? 'Lower emissions than baseline' : 'Higher emissions than baseline'} tone={saving != null && saving < 0 ? 'metric-negative' : ''} />
                <MetricCard label="Jobs completed" value={number(metrics.completed_jobs, 0)} unit={`/ ${number(metrics.total_jobs, 0)}`} note={`${number(metrics.unfinished_jobs, 0)} unfinished jobs`} />
                <MetricCard label="Deadline misses" value={number(metrics.deadline_misses, 0)} unit="jobs" note={`${number(metrics.deadline_observed_jobs, 0)} observed deadlines`} />
              </div>
              <div className="analysis-grid">
                <div className="panel chart-panel"><div className="panel-header"><div><span className="section-index">EMISSIONS</span><h3>Carbon comparison</h3></div><span className="panel-badge">Actual CI · kg CO₂</span></div><EmissionsChart metrics={metrics} policy={policy} /><p className="chart-caption">Perfect CI greedy is a foresight benchmark, not a globally optimal schedule.</p></div>
                <div className="panel detail-panel"><div className="panel-header"><div><span className="section-index">OPERATIONS</span><h3>Workload outcomes</h3></div></div><div className="detail-list"><div><span>Mean added delay</span><strong>{number(metrics.mean_added_delay_hours)} h</strong></div><div><span>95th percentile delay</span><strong>{number(metrics.p95_added_delay_hours)} h</strong></div><div><span>GPU utilisation</span><strong>{metrics.gpu_utilisation == null ? '—' : `${number(metrics.gpu_utilisation * 100)}%`}</strong></div><div><span>Total energy</span><strong>{number(metrics.total_kwh)} kWh</strong></div><div><span>Gap to perfect-CI greedy</span><strong>{number(metrics.gap_to_oracle_avg_kg)} kg</strong></div></div></div>
              </div>
              <div className="graphs-grid">
                <div className="panel graph-panel"><div className="panel-header"><div><span className="section-index">DEFERRAL WINDOW</span><h3>Carbon saved vs. k</h3></div></div><SavingsChart rows={zoneCapacityRows} ks={ks} /><p className="chart-caption">Same zone and GPU capacity as selected. Gaps appear where a run left jobs unfinished.</p></div>
                <div className="panel graph-panel"><div className="panel-header"><div><span className="section-index">JOB DELAY</span><h3>Start wait distribution</h3></div></div>{jobs ? <WaitChart jobs={jobs} /> : <p className="graph-empty">{jobsError || 'Loading per-job file…'}</p>}<p className="chart-caption">Counts include jobs that started; jobs without a start time are excluded.</p></div>
              </div>
              <SavedFiles files={files} />
              <div className="next-action"><div><strong>See what happened to each job</strong><p>Inspect arrival, start, finish, wait time, and deadline outcomes.</p></div><button onClick={() => setView('jobs')}>Open job explorer <span>→</span></button></div>
            </> : <div className="empty-panel"><div className="empty-symbol">↗</div><h3>No result for this combination</h3><p>{loaded ? 'This zone, policy, k and capacity are not in summary.csv.' : 'Run the sweep and start python api.py, then reload this page.'}</p></div>}
          </>}

          {view === 'jobs' && <>
            <section className="section-heading results-heading"><div><span className="section-index">02 / EXPLORE</span><h2>Job explorer</h2></div><p>{jobFile ? `Loaded from ${jobFile}` : 'Follow each job from arrival to completion.'}</p></section>
            {jobs ? <div className="panel jobs-panel"><div className="table-toolbar"><div><h3>Per-job outcomes</h3><p>{filteredJobs.length} of {jobs.length} jobs</p></div><div className="table-controls"><input aria-label="Search job ID" placeholder="Search job ID…" value={search} onChange={(event) => { setSearch(event.target.value); setShownJobs(12) }} /><select aria-label="Filter jobs" value={jobFilter} onChange={(event) => { setJobFilter(event.target.value as typeof jobFilter); setShownJobs(12) }}><option value="all">All jobs</option><option value="missed">Deadline missed</option><option value="unfinished">Unfinished</option></select></div></div><div className="table-scroll"><table><thead><tr><th>Job ID</th><th>Arrival</th><th>Start</th><th>Finish</th><th>Wait</th><th>GPU hours</th><th>Status</th></tr></thead><tbody>{filteredJobs.slice(0, shownJobs).map((job) => <tr key={job.job_id}><td className="mono strong">{job.job_id}</td><td>{date(job.arrival)}</td><td>{date(job.start_time)}</td><td>{date(job.finish_time)}</td><td>{number(job.start_wait_hours)} h</td><td>{number(job.busy_gpu_hours)}</td><td><span className={`status-tag ${job.deadline_missed ? 'bad' : !job.finish_time ? 'waiting' : 'good'}`}>{job.deadline_missed ? 'Missed' : !job.finish_time ? 'Unfinished' : 'Completed'}</span></td></tr>)}</tbody></table>{filteredJobs.length === 0 && <div className="table-empty">No jobs match your filter.</div>}</div>{filteredJobs.length > shownJobs && <button className="load-more" onClick={() => setShownJobs((count) => count + 24)}>Show more jobs ↓</button>}</div> : <div className="empty-panel"><div className="empty-symbol">≡</div><h3>{jobsError ? 'Per-job file not found' : 'No job results yet'}</h3><p>{jobsError || 'Select a zone, policy, k and capacity above.'}</p><button className="secondary-button" onClick={() => setView('overview')}>Back to overview</button></div>}
          </>}

          {view === 'compare' && <>
            <section className="section-heading results-heading"><div><span className="section-index">02 / COMPARE</span><h2>Policy comparison</h2></div><p>Same zone, deferral window, and GPU capacity for every policy.</p></section>
            {comparison.length ? <div className="panel compare-panel"><div className="panel-header"><div><span className="section-index">AVERAGE EMISSIONS</span><h3>Side-by-side policies</h3></div><span className="panel-badge">Lower is better</span></div><div className="compare-bars">{comparison.map((row) => <div className="compare-item" key={row.policy}><div className="compare-label"><strong>{POLICY_NAME[row.policy]}</strong><span>{number(row.emissions_avg_kg)} kg CO₂</span></div><div className="bar-track"><div className={`bar-fill ${row.policy === 'run_now' ? 'gray' : row.policy === 'oracle' ? 'blue' : 'green'}`} style={{ width: `${Math.max(0, ((row.emissions_avg_kg || 0) / comparisonMax) * 100)}%` }} /></div><div className="compare-detail"><span>{number(row.percent_saved_vs_run_now_avg_pct)}% saved</span><span>{number(row.deadline_misses, 0)} deadline misses</span><span>{number(row.mean_added_delay_hours)} h mean delay</span></div></div>)}</div><p className="chart-caption">Savings show as unavailable when a run leaves jobs unfinished. Oracle uses actual future CI as a forecast benchmark.</p></div> : <div className="empty-panel"><div className="empty-symbol">▥</div><h3>No comparison available</h3><p>No runs in summary.csv match this zone, k and capacity.</p></div>}
          </>}

          <footer className="page-footer"><span>Carbon Scheduler Studio</span><span>Results are read from the saved sweep in results/; nothing is re-simulated here.</span></footer>
        </div>
      </main>
    </div>
  )
}

export default App
