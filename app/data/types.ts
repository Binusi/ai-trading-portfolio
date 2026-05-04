// Types mirror the JSON schema produced by
// risk-return-analysis/src/export_app_data.py. If you change one, change the
// other.

export type ProfileKey = 'conservative' | 'balanced' | 'aggressive';

export type AssetClass =
  | 'us_equity'
  | 'bond_etf'
  | 'equity_index_etf'
  | 'commodity_etf'
  | 'international_etf';

export type ProfileMeta = {
  key: ProfileKey;
  name: string;
  description: string;
  targets: Record<AssetClass, number>;
  default: boolean;
};

export type ComparisonRow = {
  profile_key: ProfileKey;
  profile_name: string;
  tilt_enabled: boolean;
  data_file: string;
  final_portfolio_value: number;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  max_drawdown: number;
  max_drawdown_date: string | null;
  total_trades: number;
  total_transaction_costs: number;
};

export type BenchmarkDailyPoint = {
  date: string;
  value: number;
};

export type Benchmark = {
  name: string;
  description: string;
  metrics: {
    initial_capital: number;
    final_portfolio_value: number;
    total_return: number;
    annualized_return: number;
    max_drawdown: number;
  };
  daily: BenchmarkDailyPoint[];
};

export type SimulationConfig = {
  start_date: string;
  end_date: string;
  initial_capital: number;
  transaction_cost_bps: number;
  rebalance_cadence: string;
  tilt_cap_pct: number;
  equity_universe: string[];
  simulation_tickers: string[];
};

export type MlModelInfo = {
  model_name: string;
  target: string;
  asset_group: string;
  validation_sharpe: number | null;
};

export type Summary = {
  schema_version: number;
  generated_at: string;
  disclaimer: string;
  simulation: SimulationConfig;
  ml_model: MlModelInfo;
  profiles: ProfileMeta[];
  asset_class_labels: Record<string, string>;
  comparison: ComparisonRow[];
  benchmark: Benchmark;
  default_view: {
    profile_key: ProfileKey;
    tilt_enabled: boolean;
  };
};

export type DailyPoint = {
  date: string;
  value: number;
  daily_return: number;
  asset_class_weights: Partial<Record<AssetClass, number>>;
};

export type Tilt = {
  ticker: string;
  score: number;
  base_weight: number;
  tilt: number;
  final_weight: number;
  direction: 'overweight' | 'underweight' | 'neutral';
};

export type Trade = {
  ticker: string;
  action: 'buy' | 'sell';
  shares: number;
  price: number;
  dollar_value: number;
  transaction_cost: number;
};

export type RebalanceEvent = {
  date: string;
  rationale_text: string;
  tilt_applied: boolean;
  target_weights: Record<string, number>;
  asset_class_weights: Partial<Record<AssetClass, number>>;
  tilts: Tilt[];
  trades: Trade[];
  trade_count: number;
  transaction_cost: number;
  portfolio_value_before_rebalance: number | null;
  portfolio_value_after_rebalance: number | null;
};

export type ProfileMetrics = {
  initial_capital: number;
  final_portfolio_value: number;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  max_drawdown: number;
  max_drawdown_date: string | null;
  total_transaction_costs: number;
  total_trades: number;
  win_rate: number;
  best_day_return: number;
  worst_day_return: number;
  avg_asset_class_weights: Partial<Record<AssetClass, number>>;
};

export type ProfileDetail = {
  profile: ProfileMeta;
  tilt_enabled: boolean;
  metrics: ProfileMetrics;
  daily: DailyPoint[];
  rebalance_events: RebalanceEvent[];
};
