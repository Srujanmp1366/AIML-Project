export type Policy = 'run_now' | 'threshold' | 'forecast_greedy' | 'oracle'

export interface Metrics {
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
  gap_to_oracle_avg_kg: number | null
}

export interface CompareRow extends Metrics {
  zone: string
  policy: Policy
  k: number
  gpus: string
}

export interface JobRow {
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

export type Row = Record<string, string>

/** Minimal CSV parser: quoted fields, escaped quotes, LF or CRLF line endings. */
export function parseCsv(text: string): Row[] {
  const records: string[][] = []
  let field = ''
  let record: string[] = []
  let quoted = false
  const input = text.replace(/^\uFEFF/, '')
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (quoted) {
      if (ch === '"' && input[i + 1] === '"') { field += '"'; i++ }
      else if (ch === '"') quoted = false
      else field += ch
    } else if (ch === '"') quoted = true
    else if (ch === ',') { record.push(field); field = '' }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && input[i + 1] === '\n') i++
      record.push(field); field = ''
      if (record.some((value) => value !== '')) records.push(record)
      record = []
    } else field += ch
  }
  if (field !== '' || record.length) { record.push(field); if (record.some((value) => value !== '')) records.push(record) }
  if (!records.length) return []
  const header = records[0].map((name) => name.trim())
  return records.slice(1).map((values) => Object.fromEntries(header.map((name, index) => [name, (values[index] ?? '').trim()])))
}

function num(value: string | undefined): number | null {
  if (value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function bool(value: string | undefined): boolean {
  return ['true', '1', 'yes'].includes((value ?? '').trim().toLowerCase())
}

function text(value: string | undefined): string | null {
  return value === undefined || value === '' ? null : value
}

const POLICY_IDS = new Set<string>(['run_now', 'threshold', 'forecast_greedy', 'oracle'])

/** Convert summary.csv rows; later rows replace earlier ones with the same zone/policy/k/gpus. */
export function toSummaryRows(rows: Row[]): CompareRow[] {
  const byKey = new Map<string, CompareRow>()
  for (const row of rows) {
    const k = num(row.k)
    if (!POLICY_IDS.has(row.policy) || k == null || !row.gpus) continue
    const entry: CompareRow = {
      zone: row.zone ?? '',
      policy: row.policy as Policy,
      k,
      gpus: row.gpus,
      total_kwh: num(row.total_kwh),
      emissions_avg_kg: num(row.emissions_avg_kg),
      emissions_marg_kg: num(row.emissions_marg_kg),
      total_jobs: num(row.total_jobs),
      completed_jobs: num(row.completed_jobs),
      unfinished_jobs: num(row.unfinished_jobs),
      deadline_misses: num(row.deadline_misses),
      deadline_observed_jobs: num(row.deadline_observed_jobs),
      mean_added_delay_hours: num(row.mean_added_delay_hours),
      p95_added_delay_hours: num(row.p95_added_delay_hours),
      gpu_utilisation: num(row.gpu_utilisation),
      run_now_avg_kg: num(row.run_now_avg_kg),
      percent_saved_vs_run_now_avg_pct: num(row.percent_saved_vs_run_now_avg_pct),
      greedy_perfect_ci_avg_kg: num(row.greedy_perfect_ci_avg_kg),
      gap_to_oracle_avg_kg: num(row.gap_to_oracle_avg_kg),
    }
    byKey.set(`${entry.zone}|${entry.policy}|${entry.k}|${entry.gpus}`, entry)
  }
  return [...byKey.values()]
}

export function toJobRows(rows: Row[]): JobRow[] {
  return rows.map((row) => ({
    job_id: row.job_id,
    arrival: text(row.arrival),
    start_time: text(row.start_time),
    finish_time: text(row.finish_time),
    deadline: text(row.deadline),
    deferrable: bool(row.deferrable),
    deadline_observed: bool(row.deadline_observed),
    deadline_missed: bool(row.deadline_missed),
    infeasible_on_arrival: bool(row.infeasible_on_arrival),
    start_wait_hours: num(row.start_wait_hours),
    added_delay_hours: num(row.added_delay_hours),
    busy_gpu_hours: num(row.busy_gpu_hours),
  }))
}

/** Same naming rule as run.py: <zone slug>_<policy>_k<k>_g<gpus>.csv */
export function zoneSlug(zone: string): string {
  return zone.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '') || 'zone'
}

export function jobFileName(row: { zone: string; policy: string; k: number; gpus: string }): string {
  const base = `${row.policy}_k${row.k}_g${row.gpus}.csv`
  return row.zone ? `${zoneSlug(row.zone)}_${base}` : base
}
