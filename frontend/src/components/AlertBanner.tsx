import { AlertTriangle, Zap, TrendingUp, TrendingDown } from 'lucide-react'
import { clsx } from 'clsx'

interface Props {
  alerts: string[]
}

const ALERT_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  VPIN_HIGH: {
    label: '⚠ VPIN SPIKE — Informed trading detected',
    color: 'border-red/40 bg-red/10 text-red',
    icon: <AlertTriangle className="w-3.5 h-3.5" />,
  },
  OFI_BUY: {
    label: '⚡ OFI BUY PRESSURE — Net buy order flow',
    color: 'border-accent/40 bg-accent/10 text-accent',
    icon: <TrendingUp className="w-3.5 h-3.5" />,
  },
  OFI_SELL: {
    label: '⚡ OFI SELL PRESSURE — Net sell order flow',
    color: 'border-red/40 bg-red/10 text-red',
    icon: <TrendingDown className="w-3.5 h-3.5" />,
  },
}

export function AlertBanner({ alerts }: Props) {
  const recent = alerts.slice(-3).reverse()
  if (recent.length === 0) return null

  return (
    <div className="flex gap-2 px-4 py-1.5 flex-wrap bg-bg border-b border-border">
      {recent.map((alert, i) => {
        const cfg = ALERT_CONFIG[alert]
        if (!cfg) return null
        return (
          <div
            key={`${alert}-${i}`}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-0.5 rounded-full text-[11px] font-mono border',
              cfg.color
            )}
          >
            {cfg.icon}
            {cfg.label}
          </div>
        )
      })}
    </div>
  )
}
