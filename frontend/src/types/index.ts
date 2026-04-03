export interface LOBLevel {
  px: number
  vol: number
}

export interface HJBData {
  q_arr: number[]
  bid_spreads: (number | null)[]
  ask_spreads: (number | null)[]
  cur_bid_sp: number | null
  cur_ask_sp: number | null
  t_frac: number
  gamma: number
  q: number
}

export interface MarketState {
  tick: number
  source: string
  ready: boolean
  mid: number
  spread: number
  ofi: number
  vpin: number
  kyle: number
  pnl: number
  n_fills: number
  sharpe: number
  fill_rate: number
  mids: number[]
  ofis: number[]
  vpins: number[]
  kyles: number[]
  pnls: number[]
  lob: { bids: LOBLevel[]; asks: LOBLevel[] }
  alerts: string[]
  event_times: number[]
}
