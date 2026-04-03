import { clsx } from 'clsx'

interface SliderProps {
  label: string
  sub?: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
  format?: (v: number) => string
  color?: string
}

function Slider({ label, sub, min, max, step, value, onChange, format, color = 'accent' }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="flex flex-col gap-2 flex-1 min-w-[160px] max-w-[280px]">
      <div className="flex items-baseline justify-between">
        <div>
          <span className="text-[11px] font-mono uppercase tracking-widest text-muted">{label}</span>
          {sub && <span className="ml-1.5 text-[10px] text-dim">{sub}</span>}
        </div>
        <span className={clsx('text-sm font-mono font-semibold tabular-nums',
          color === 'accent' ? 'text-accent' : 'text-yellow'
        )}>
          {format ? format(value) : value}
        </span>
      </div>
      <div className="relative h-1.5 rounded-full bg-dim cursor-pointer">
        <div
          className={clsx('absolute h-full rounded-full transition-all',
            color === 'accent' ? 'bg-accent' : 'bg-yellow'
          )}
          style={{ width: `${pct}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div
          className={clsx(
            'absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full border-2 pointer-events-none',
            color === 'accent'
              ? 'bg-bg border-accent shadow-glow'
              : 'bg-bg border-yellow'
          )}
          style={{ left: `calc(${pct}% - 7px)` }}
        />
      </div>
      <div className="flex justify-between text-[9px] font-mono text-dim">
        <span>{format ? format(min) : min}</span>
        <span>{format ? format(max) : max}</span>
      </div>
    </div>
  )
}

interface Props {
  inventory: number
  gamma: number
  onInventoryChange: (v: number) => void
  onGammaChange: (v: number) => void
}

export function Controls({ inventory, gamma, onInventoryChange, onGammaChange }: Props) {
  return (
    <div className="flex items-center gap-8 px-5 py-3 bg-surface border-b border-border">
      <div className="text-[10px] font-mono uppercase tracking-widest text-muted whitespace-nowrap">
        Parameters
      </div>
      <Slider
        label="Inventory q"
        sub="(units)"
        min={-10}
        max={10}
        step={1}
        value={inventory}
        onChange={v => onInventoryChange(Math.round(v))}
        format={v => (v >= 0 ? `+${v}` : `${v}`)}
        color="accent"
      />
      <Slider
        label="Risk Aversion γ"
        sub="(Avellaneda-Stoikov)"
        min={0.001}
        max={0.1}
        step={0.001}
        value={gamma}
        onChange={onGammaChange}
        format={v => v.toFixed(3)}
        color="yellow"
      />

      {/* Legend */}
      <div className="hidden lg:flex items-center gap-4 ml-auto text-[10px] font-mono">
        {[
          { color: 'bg-accent', label: 'OFI' },
          { color: 'bg-yellow', label: 'VPIN' },
          { color: 'bg-green', label: 'Kyle λ' },
          { color: 'bg-muted', label: 'Mid' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5 text-muted">
            <div className={`w-4 h-0.5 rounded ${color}`} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
