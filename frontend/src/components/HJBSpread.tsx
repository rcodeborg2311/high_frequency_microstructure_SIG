import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart,
  ComposedChart,
} from 'recharts'
import type { HJBData } from '../types'

interface Props {
  hjb: HJBData | null
  inventory: number
  gamma: number
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const q = payload[0]?.payload?.q
  return (
    <div className="bg-card border border-border rounded px-3 py-2 text-[11px] font-mono shadow-xl">
      <div className="text-muted mb-1">q = {q >= 0 ? `+${q}` : q}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color }} className="flex justify-between gap-4">
          <span>{p.name}</span>
          <span className="tabular-nums">{p.value != null ? p.value.toFixed(5) : '—'}</span>
        </div>
      ))}
    </div>
  )
}

export function HJBSpread({ hjb, inventory, gamma }: Props) {
  if (!hjb) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="text-muted text-xs font-mono animate-pulse">Computing HJB…</span>
      </div>
    )
  }

  const data = hjb.q_arr.map((q, i) => ({
    q,
    'δ*bid': hjb.bid_spreads[i],
    'δ*ask': hjb.ask_spreads[i],
  }))

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 pt-3 pb-1 flex items-center justify-between">
        <span className="text-[11px] font-mono uppercase tracking-widest text-muted">
          HJB Optimal Spread
        </span>
        <div className="flex gap-3 text-[10px] font-mono">
          <span className="text-muted">γ=</span>
          <span className="text-yellow">{gamma.toFixed(3)}</span>
          <span className="text-muted ml-2">{(hjb.t_frac * 100).toFixed(0)}% of day</span>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 4, right: 16, bottom: 20, left: 8 }}>
            <CartesianGrid stroke="#1a2744" strokeWidth={0.5} />
            <XAxis
              dataKey="q"
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={{ stroke: '#1a2744' }}
              tickLine={false}
              label={{ value: 'Inventory q', position: 'insideBottom', offset: -8, fill: '#4a5568', fontSize: 9 }}
            />
            <YAxis
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v?.toFixed(4) ?? ''}
              label={{ value: 'Half-spread δ*', angle: -90, position: 'insideLeft', fill: '#4a5568', fontSize: 9 }}
            />

            {/* Current inventory marker */}
            <ReferenceLine x={inventory} stroke="rgba(255,255,255,0.3)" strokeDasharray="4 2" strokeWidth={1.5} />

            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', paddingTop: 4 }}
            />

            <Line dataKey="δ*bid" stroke="#00c853" strokeWidth={2} dot={{ r: 2, fill: '#00c853' }} isAnimationActive={false} connectNulls={false} />
            <Line dataKey="δ*ask" stroke="#ff3b3b" strokeWidth={2} dot={{ r: 2, fill: '#ff3b3b' }} isAnimationActive={false} connectNulls={false} />

            {/* Current inventory point highlights */}
            {hjb.cur_bid_sp != null && (
              <Scatter
                data={[{ q: inventory, '★bid': hjb.cur_bid_sp }]}
                dataKey="★bid"
                fill="#00c853"
                shape={(props: any) => (
                  <circle cx={props.cx} cy={props.cy} r={6} fill="#00c853" stroke="#060a14" strokeWidth={1.5} />
                )}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Current spread callout */}
      {hjb.cur_bid_sp != null && hjb.cur_ask_sp != null && (
        <div className="px-4 pb-2 grid grid-cols-2 gap-2">
          <div className="bg-green/10 border border-green/20 rounded px-3 py-1.5 text-[11px] font-mono">
            <span className="text-muted">δ*bid  </span>
            <span className="text-green font-semibold">{hjb.cur_bid_sp.toFixed(5)}</span>
          </div>
          <div className="bg-red/10 border border-red/20 rounded px-3 py-1.5 text-[11px] font-mono">
            <span className="text-muted">δ*ask  </span>
            <span className="text-red font-semibold">{hjb.cur_ask_sp.toFixed(5)}</span>
          </div>
        </div>
      )}
    </div>
  )
}
