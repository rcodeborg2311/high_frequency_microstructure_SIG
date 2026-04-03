import { Activity, Radio, Wifi, WifiOff } from 'lucide-react'
import { clsx } from 'clsx'

interface Props {
  connected: boolean
  source: string
  tick: number
}

export function Header({ connected, source, tick }: Props) {
  const isLive = source.includes('Coinbase')

  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-border bg-surface">
      {/* Logo + Title */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <Activity className="w-5 h-5 text-accent" />
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-green animate-pulse" />
        </div>
        <div>
          <h1 className="font-sans font-semibold text-sm tracking-wide text-text">
            HF Market Microstructure
          </h1>
          <p className="font-mono text-[10px] text-muted tracking-widest uppercase">
            Signal Intelligence Platform
          </p>
        </div>
      </div>

      {/* Center — tick counter */}
      <div className="hidden md:flex items-center gap-6 text-xs font-mono">
        <span className="text-muted">TICK</span>
        <span className="text-accent font-medium tabular-nums">
          {tick.toLocaleString().padStart(6, '0')}
        </span>
      </div>

      {/* Right — connection badge */}
      <div className="flex items-center gap-3">
        {/* WS status — primary badge */}
        <div className={clsx(
          'flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-medium border',
          connected
            ? 'border-green/30 bg-green/10 text-green'
            : 'border-red/30 bg-red/10 text-red'
        )}>
          {connected
            ? <Wifi className="w-3 h-3" />
            : <WifiOff className="w-3 h-3" />}
          {connected ? 'CONNECTED' : 'RECONNECTING'}
        </div>

        {/* Source badge — secondary, muted */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono border border-dim text-muted">
          {isLive ? <Radio className="w-3 h-3" /> : <Activity className="w-3 h-3" />}
          {isLive ? 'LIVE · BTC-USD' : 'SYNTHETIC'}
        </div>
      </div>
    </header>
  )
}
