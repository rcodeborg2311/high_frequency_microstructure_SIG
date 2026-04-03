import { clsx } from 'clsx'
import type { MarketState } from '../types'

interface StatProps {
  label: string
  value: string
  color?: string
  pulse?: boolean
}

function Stat({ label, value, color = 'text-text', pulse }: StatProps) {
  return (
    <div className="flex flex-col gap-0.5 px-4 border-r border-border last:border-0">
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted">{label}</span>
      <span className={clsx('text-sm font-mono font-medium tabular-nums', color, pulse && 'animate-pulse')}>
        {value}
      </span>
    </div>
  )
}

interface Props {
  state: MarketState
}

export function StatsBar({ state }: Props) {
  const pnlColor = state.pnl >= 0 ? 'text-green' : 'text-red'
  const ofiColor = state.ofi > 0.5 ? 'text-accent' : state.ofi < -0.5 ? 'text-red' : 'text-text'
  const vpinColor = state.vpin > 0.7 ? 'text-red' : state.vpin > 0.55 ? 'text-yellow' : 'text-green'
  const sharpeColor = state.sharpe > 1 ? 'text-green' : state.sharpe < -1 ? 'text-red' : 'text-text'

  return (
    <div className="flex items-stretch bg-surface border-b border-border overflow-x-auto">
      <Stat
        label="Mid Price"
        value={state.mid.toFixed(4)}
        color="text-text"
      />
      <Stat
        label="Spread"
        value={state.spread.toFixed(5)}
        color="text-muted"
      />
      <Stat
        label="OFI"
        value={state.ofi >= 0 ? `+${state.ofi.toFixed(3)}` : state.ofi.toFixed(3)}
        color={ofiColor}
      />
      <Stat
        label="VPIN"
        value={state.vpin.toFixed(3)}
        color={vpinColor}
        pulse={state.vpin > 0.75}
      />
      <Stat
        label="Kyle λ"
        value={Number.isFinite(state.kyle) ? state.kyle.toExponential(2) : '—'}
        color="text-text"
      />
      <Stat
        label="Cum P&L"
        value={(state.pnl >= 0 ? '+' : '') + state.pnl.toFixed(4)}
        color={pnlColor}
      />
      <Stat
        label="Sharpe"
        value={Number.isFinite(state.sharpe) ? state.sharpe.toFixed(2) : '—'}
        color={sharpeColor}
      />
      <Stat
        label="Fill Rate"
        value={`${state.fill_rate.toFixed(1)}%`}
        color="text-text"
      />
      <Stat
        label="Fills"
        value={state.n_fills.toLocaleString()}
        color="text-accent"
      />
    </div>
  )
}
